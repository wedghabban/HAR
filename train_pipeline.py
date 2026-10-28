import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ==========================================
# 0. HARDWARE & SEED CONFIGURATION
# ==========================================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--> Active Compute Device: {device} ({torch.cuda.get_device_name(0)})")

os.makedirs("saved_models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ==========================================
# 1. DATA AUDITING & VALIDATION
# ==========================================
CSV_PATH = "data/Training_set.csv"
IMG_DIR = "data/train"

if not os.path.exists(CSV_PATH) or not os.path.exists(IMG_DIR):
    raise FileNotFoundError(f"Missing '{CSV_PATH}' or '{IMG_DIR}'. Check folder paths!")

df = pd.read_csv(CSV_PATH)
df['full_path'] = df['filename'].apply(lambda x: os.path.join(IMG_DIR, x))
df = df[df['full_path'].apply(os.path.exists)].reset_index(drop=True)

valid_indices = []
print("Auditing image integrity...")
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Auditing"):
    try:
        with Image.open(row['full_path']) as img:
            img.verify()
        valid_indices.append(idx)
    except Exception:
        pass

df = df.iloc[valid_indices].reset_index(drop=True)
classes = sorted(df['label'].unique())
class_to_idx = {name: i for i, name in enumerate(classes)}
idx_to_class = {i: name for i, name in enumerate(classes)}
df['target'] = df['label'].map(class_to_idx)
num_classes = len(classes)
print(f"Verified Samples: {len(df)} across {num_classes} action classes.")

# ==========================================
# 2. STRATIFIED 70:15:15 SPLIT
# ==========================================
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['target'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['target'], random_state=42)

weights = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=train_df['target'].values)
class_weights = torch.tensor(weights, dtype=torch.float).to(device)

# ==========================================
# 3. RESOLUTION (288x288) & AUGMENTATIONS
# ==========================================
IMG_SIZE = (288, 288)
BATCH_SIZE = 16

class HARDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, 'full_path']
        img = Image.open(path).convert('RGB')
        label = self.df.loc[idx, 'target']
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

train_transforms = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), value='random')
])

eval_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_loader = DataLoader(HARDataset(train_df, train_transforms), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(HARDataset(val_df, eval_transforms), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(HARDataset(test_df, eval_transforms), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ==========================================
# 4. CONVNEXT-TINY ARCHITECTURE
# ==========================================
model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)

in_features = model.classifier[2].in_features
model.classifier[2] = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features, 512),
    nn.LayerNorm(512),
    nn.GELU(),
    nn.Dropout(p=0.3),
    nn.Linear(512, num_classes)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.08)
scaler = torch.amp.GradScaler('cuda')

def train_one_epoch(loader, optimizer):
    model.train()
    total_loss, correct = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def evaluate_loader(loader):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# ==========================================
# 5. PHASE 1: WARMUP HEAD (25 EPOCHS)
# ==========================================
print("\n--- PHASE 1: WARMUP HEAD (25 Epochs, High LR) ---")
for param in model.features.parameters():
    param.requires_grad = False

optimizer_p1 = torch.optim.AdamW(model.classifier.parameters(), lr=7e-4, weight_decay=1e-3)
scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_p1, T_max=25, eta_min=1e-5)

p1_history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
best_val_acc = 0.0

for epoch in range(25):
    tr_loss, tr_acc = train_one_epoch(train_loader, optimizer_p1)
    vl_loss, vl_acc = evaluate_loader(val_loader)
    scheduler_p1.step()
    
    p1_history['train_acc'].append(tr_acc)
    p1_history['val_acc'].append(vl_acc)
    p1_history['train_loss'].append(tr_loss)
    p1_history['val_loss'].append(vl_loss)
    
    print(f"P1 [{epoch+1:02d}/25] Train Acc: {tr_acc*100:.2f}% | Val Acc: {vl_acc*100:.2f}% | Val Loss: {vl_loss:.4f}")
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save(model.state_dict(), "saved_models/best_phase1_convnext.pth")

model.load_state_dict(torch.load("saved_models/best_phase1_convnext.pth"))

# ==========================================
# 6. PHASE 2: DISCRIMINATIVE FINE-TUNING (25 EPOCHS)
# ==========================================
print("\n--- PHASE 2: DISCRIMINATIVE FINE-TUNING (25 Epochs, Stages 2-3 Unfrozen) ---")
for param in model.features.parameters():
    param.requires_grad = True

# ConvNeXt features have 8 stages (0, 2, 4, 6 are downsamplers; 1, 3, 5, 7 are blocks)
# Freeze stages up through stage 4 (first two downsample-block groups)
for stage_idx in range(5):
    for param in model.features[stage_idx].parameters():
        param.requires_grad = False

optimizer_p2 = torch.optim.AdamW([
    {'params': [p for p in model.features.parameters() if p.requires_grad], 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-4}
], weight_decay=1e-3)

scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=25, eta_min=1e-7)

p2_history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

for epoch in range(25):
    tr_loss, tr_acc = train_one_epoch(train_loader, optimizer_p2)
    vl_loss, vl_acc = evaluate_loader(val_loader)
    scheduler_p2.step()
    
    p2_history['train_acc'].append(tr_acc)
    p2_history['val_acc'].append(vl_acc)
    p2_history['train_loss'].append(tr_loss)
    p2_history['val_loss'].append(vl_loss)
    
    print(f"P2 [{epoch+1:02d}/25] Train Acc: {tr_acc*100:.2f}% | Val Acc: {vl_acc*100:.2f}% | Val Loss: {vl_loss:.4f}")
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save(model.state_dict(), "saved_models/best_final_convnext_90plus.pth")
        print(f"--> Checkpoint Saved: {best_val_acc*100:.2f}%")

# ==========================================
# 7. EVALUATION WITH TEST-TIME AUGMENTATION (TTA)
# ==========================================
print("\n--- RUNNING TEST-TIME AUGMENTATION (TTA) ON PURE TEST SET ---")
model.load_state_dict(torch.load("saved_models/best_final_convnext_90plus.pth"))
model.eval()

y_preds, y_trues = [], []
y_probs_all = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast('cuda'):
            out1 = torch.softmax(model(imgs), dim=1)
            imgs_flipped = torch.flip(imgs, dims=[3])
            out2 = torch.softmax(model(imgs_flipped), dim=1)
            
        final_probs = (out1 + out2) / 2.0
        preds = final_probs.argmax(dim=1).cpu().numpy()
        
        y_preds.extend(preds)
        y_trues.extend(labels.numpy())
        y_probs_all.extend(final_probs.cpu().numpy())

y_preds = np.array(y_preds)
y_trues = np.array(y_trues)
y_probs_all = np.array(y_probs_all)
final_test_acc = (y_preds == y_trues).mean()

print(f"\n==============================================")
print(f"FINAL UNSEEN TEST ACCURACY (WITH TTA): {final_test_acc * 100:.2f}%")
print(f"==============================================\n")

report_dict = classification_report(y_trues, y_preds, target_names=classes, output_dict=True)
report_str = classification_report(y_trues, y_preds, target_names=classes)
print(report_str)
with open("outputs/classification_report_90plus.txt", "w") as f:
    f.write(report_str)

# ==========================================
# 8. VISUALIZATION SUITE (SAVED TO /outputs)
# ==========================================
print("\n--> Generating and saving visualization outputs...")

# 8.1 Training & Validation Progression Curves
epochs_p1 = len(p1_history['train_acc'])
epochs_p2 = len(p2_history['train_acc'])
total_epochs = epochs_p1 + epochs_p2
x_axis = range(1, total_epochs + 1)

full_tr_acc = p1_history['train_acc'] + p2_history['train_acc']
full_val_acc = p1_history['val_acc'] + p2_history['val_acc']
full_tr_loss = p1_history['train_loss'] + p2_history['train_loss']
full_val_loss = p1_history['val_loss'] + p2_history['val_loss']

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(x_axis, [x * 100 for x in full_tr_acc], label='Train Accuracy', color='#1f77b4', lw=2)
plt.plot(x_axis, [x * 100 for x in full_val_acc], label='Val Accuracy', color='#2ca02c', lw=2)
plt.axvline(x=epochs_p1, color='red', linestyle='--', label='Phase 2 Fine-Tuning Start')
plt.title('Accuracy Progression Across Phases', fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_axis, full_tr_loss, label='Train Loss', color='#d62728', lw=2)
plt.plot(x_axis, full_val_loss, label='Val Loss', color='#ff7f0e', lw=2)
plt.axvline(x=epochs_p1, color='red', linestyle='--', label='Phase 2 Fine-Tuning Start')
plt.title('Loss Progression Across Phases', fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss (Label Smoothed CE)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig("outputs/training_progress_curves.png", dpi=300)
plt.close()

# 8.2 Normalized Confusion Matrix
cm = confusion_matrix(y_trues, y_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(13, 11))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=True)
plt.xlabel('Predicted Action', fontsize=12)
plt.ylabel('Actual Action', fontsize=12)
plt.title(f'Normalized Test Confusion Matrix (TTA Acc: {final_test_acc * 100:.2f}%)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/normalized_confusion_matrix.png", dpi=300)
plt.close()

# 8.3 Per-Class F1, Precision, and Recall Bar Plot
class_metrics = []
for c in classes:
    class_metrics.append({
        'Class': c,
        'Precision': report_dict[c]['precision'],
        'Recall': report_dict[c]['recall'],
        'F1-Score': report_dict[c]['f1-score']
    })
metrics_df = pd.DataFrame(class_metrics).sort_values(by='F1-Score', ascending=False)

fig, ax = plt.subplots(figsize=(14, 7))
bar_width = 0.25
indices = np.arange(len(classes))

ax.bar(indices - bar_width, metrics_df['Precision'], width=bar_width, label='Precision', color='#4c72b0')
ax.bar(indices, metrics_df['Recall'], width=bar_width, label='Recall', color='#55a868')
ax.bar(indices + bar_width, metrics_df['F1-Score'], width=bar_width, label='F1-Score', color='#c44e52')

ax.set_xlabel('Action Class', fontsize=12)
ax.set_ylabel('Score (0.0 to 1.0)', fontsize=12)
ax.set_title('Per-Class Precision, Recall, and F1-Score on Unseen Test Set', fontsize=14)
ax.set_xticks(indices)
ax.set_xticklabels(metrics_df['Class'], rotation=45, ha='right')
ax.set_ylim([0.0, 1.05])
ax.grid(axis='y', linestyle=':', alpha=0.6)
ax.legend()
plt.tight_layout()
plt.savefig("outputs/per_class_performance.png", dpi=300)
plt.close()

# 8.4 Multi-Class ROC / AUC Curves
y_test_bin = label_binarize(y_trues, classes=np.arange(num_classes))
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_probs_all[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_probs_all.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure(figsize=(12, 9))
plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.3f})',
         color='deeppink', linestyle=':', linewidth=3)

colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=1.5, label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Multi-Class One-vs-Rest ROC Curves', fontsize=14)
plt.legend(loc="lower right", fontsize=9, ncol=2)
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("outputs/roc_auc_curves.png", dpi=300)
plt.close()

# ==========================================
# 9. PYTORCH NATIVE GRAD-CAM VISUALIZATION
# ==========================================
print("--> Generating Grad-CAM visual explainability maps...")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class):
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        loss = output[0, target_class]
        loss.backward()
        
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1).squeeze(0)
        cam = torch.clamp(cam, min=0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()

# For ConvNeXt, features[-1] is the final norm layer; features[-2] holds the final 7x7 convolutional block
target_layer = model.features[-2]
grad_cam = GradCAM(model, target_layer)

num_samples = 4
fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

for idx in range(num_samples):
    sample = test_df.iloc[idx]
    orig_img = cv2.imread(sample['full_path'])
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    h, w, _ = orig_img.shape
    
    resized_img = cv2.resize(orig_img, IMG_SIZE)
    tensor_img = eval_transforms(Image.fromarray(resized_img)).unsqueeze(0).to(device)
    
    label_idx = sample['target']
    heatmap = grad_cam.generate(tensor_img, label_idx)
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    overlay = np.uint8(0.4 * heatmap_colored + 0.6 * orig_img)
    
    axes[idx, 0].imshow(orig_img)
    axes[idx, 0].set_title(f"Target: {sample['label']}", fontsize=11)
    axes[idx, 0].axis('off')
    
    axes[idx, 1].imshow(heatmap_resized, cmap='jet')
    axes[idx, 1].set_title("Grad-CAM Activation", fontsize=11)
    axes[idx, 1].axis('off')
    
    axes[idx, 2].imshow(overlay)
    axes[idx, 2].set_title("Overlay Attention Map", fontsize=11)
    axes[idx, 2].axis('off')

plt.tight_layout()
plt.savefig("outputs/gradcam_explainability.png", dpi=300)
plt.close()

print("--> All visual artifacts exported successfully to 'outputs/' directory.")