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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--> Compute Device: {device} ({torch.cuda.get_device_name(0)})")

# ==========================================
# 1. LOAD TEST DATASET (EXACT SAME SPLIT)
# ==========================================
CSV_PATH = "data/Training_set.csv"
IMG_DIR = "data/train"

df = pd.read_csv(CSV_PATH)
df['full_path'] = df['filename'].apply(lambda x: os.path.join(IMG_DIR, x))
df = df[df['full_path'].apply(os.path.exists)].reset_index(drop=True)

# Filter corrupted images
valid_indices = []
for idx, row in df.iterrows():
    try:
        with Image.open(row['full_path']) as img:
            img.verify()
        valid_indices.append(idx)
    except Exception:
        pass
df = df.iloc[valid_indices].reset_index(drop=True)

classes = sorted(df['label'].unique())
class_to_idx = {name: i for i, name in enumerate(classes)}
df['target'] = df['label'].map(class_to_idx)
num_classes = len(classes)

# Exact same random state 42
_, temp_df = train_test_split(df, test_size=0.30, stratify=df['target'], random_state=42)
_, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['target'], random_state=42)
test_df = test_df.reset_index(drop=True)
print(f"Loaded Unseen Test Set: {len(test_df)} samples across {num_classes} classes.")

# ==========================================
# 2. BUILD ARCHITECTURES & LOAD WEIGHTS
# ==========================================
print("\n--> Loading Pretrained Checkpoints...")

# Model A: ConvNeXt-Tiny (Input 288x288)
model_conv = models.convnext_tiny()
in_features_conv = model_conv.classifier[2].in_features
model_conv.classifier[2] = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features_conv, 512),
    nn.LayerNorm(512),
    nn.GELU(),
    nn.Dropout(p=0.3),
    nn.Linear(512, num_classes)
)
conv_path = "saved_models/best_final_convnext_90plus.pth"
model_conv.load_state_dict(torch.load(conv_path, map_location=device))
model_conv = model_conv.to(device)
model_conv.eval()
print(f"Loaded ConvNeXt-Tiny from: {conv_path}")

# Model B: EfficientNet-B2 (Input 260x260)
model_eff = models.efficientnet_b2()
in_features_eff = model_eff.classifier[1].in_features
model_eff.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features_eff, 512),
    nn.BatchNorm1d(512),
    nn.SiLU(),
    nn.Dropout(p=0.3),
    nn.Linear(512, num_classes)
)
eff_path = "saved_models/best_final_b2_90plus.pth"
model_eff.load_state_dict(torch.load(eff_path, map_location=device))
model_eff = model_eff.to(device)
model_eff.eval()
print(f"Loaded EfficientNet-B2 from: {eff_path}")

# ==========================================
# 3. MULTI-SCALE TEST-TIME AUGMENTATION (TTA)
# ==========================================
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def get_tta_tensors(img_pil, size):
    # Multi-Scale & Multi-Crop TTA (Original, Flipped, Scaled Up & Center Cropped)
    t1 = norm(transforms.ToTensor()(transforms.Resize(size)(img_pil)))
    t2 = torch.flip(t1, dims=[2])
    
    # Scale crop (+10% zoom to catch micro gestures like texting/calling)
    zoom_size = (int(size[0] * 1.1), int(size[1] * 1.1))
    t3 = norm(transforms.ToTensor()(transforms.CenterCrop(size)(transforms.Resize(zoom_size)(img_pil))))
    t4 = torch.flip(t3, dims=[2])
    
    return torch.stack([t1, t2, t3, t4]).to(device)

# ==========================================
# 4. ENSEMBLE INFERENCE (WEIGHTED SOFT-VOTING)
# ==========================================
print("\n--> Running Deep Ensemble + Multi-Crop TTA on Test Set...")

y_trues = []
ensemble_probs = []

# ConvNeXt gets 65% weight (higher individual accuracy), EfficientNet gets 35%
W_CONV = 0.65
W_EFF = 0.35

with torch.no_grad():
    for idx in tqdm(range(len(test_df)), desc="Evaluating Ensemble"):
        row = test_df.iloc[idx]
        img = Image.open(row['full_path']).convert('RGB')
        label = row['target']
        y_trues.append(label)
        
        # ConvNeXt TTA forward passes (288x288)
        batch_conv = get_tta_tensors(img, (288, 288))
        with torch.amp.autocast('cuda'):
            out_conv = torch.softmax(model_conv(batch_conv), dim=1)
        prob_conv = out_conv.mean(dim=0).cpu().numpy()
        
        # EfficientNet-B2 TTA forward passes (260x260)
        batch_eff = get_tta_tensors(img, (260, 260))
        with torch.amp.autocast('cuda'):
            out_eff = torch.softmax(model_eff(batch_eff), dim=1)
        prob_eff = out_eff.mean(dim=0).cpu().numpy()
        
        # Soft voting blend
        blended_prob = (W_CONV * prob_conv) + (W_EFF * prob_eff)
        ensemble_probs.append(blended_prob)

y_trues = np.array(y_trues)
ensemble_probs = np.array(ensemble_probs)
y_preds = ensemble_probs.argmax(axis=1)

ensemble_acc = (y_preds == y_trues).mean()

print("\n" + "="*50)
print(f"🔥 FINAL ENSEMBLE UNSEEN TEST ACCURACY: {ensemble_acc * 100:.2f}% 🔥")
print("="*50 + "\n")

# ==========================================
# 5. GENERATE FINAL REPORT & VISUALS
# ==========================================
os.makedirs("outputs", exist_ok=True)

report_str = classification_report(y_trues, y_preds, target_names=classes)
print(report_str)
with open("outputs/ensemble_classification_report.txt", "w") as f:
    f.write(report_str)

# Normalized Confusion Matrix
cm = confusion_matrix(y_trues, y_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(13, 11))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Action', fontsize=12)
plt.ylabel('Actual Action', fontsize=12)
plt.title(f'Ensemble Test Confusion Matrix (Acc: {ensemble_acc * 100:.2f}%)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("outputs/ensemble_confusion_matrix.png", dpi=300)
plt.close()

# Micro & Macro ROC Curves
y_test_bin = label_binarize(y_trues, classes=np.arange(num_classes))
fpr, tpr, _ = roc_curve(y_test_bin.ravel(), ensemble_probs.ravel())
micro_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='deeppink', lw=2.5, label=f'Ensemble Micro-average ROC (AUC = {micro_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
plt.title(f'Ensemble ROC Curve (AUC = {micro_auc:.3f})', fontsize=14)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("outputs/ensemble_roc_curve.png", dpi=300)
plt.close()

print("--> All Ensemble diagnostic reports saved to 'outputs/'!")