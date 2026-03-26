import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob

# Import your model from wherever you saved it
# from my_model_file import MILModel_ResNet50 
# uses best model weights that we get after running extract_features.py
# --- CONFIGURATION ---
FEATURES_DIR = "./extracted_features"
# Example target labels: [IDH1_mutated, ATRX_mutated, P53_mutated]
# 1.0 = Mutated/Positive, 0.0 = Wildtype/Negative
CSV_PATH = "./labels.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# THE MIL MODEL (Upgraded for ResNet50)
# ==========================================
class MILModel_ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        # Separate attention for each biomarker (2048 dims for ResNet50)
        self.attention_IDH1 = self._make_attention()
        self.attention_ATRX = self._make_attention()
        self.attention_P53 = self._make_attention()

        # Classifier
        self.classifier = nn.Linear(2048, 3)

    def _make_attention(self):
        return nn.Sequential(
            nn.Linear(2048, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, feats, return_attention=False):
        # feats is shape (N, 2048) representing all patches in ONE slide
        A_idh1 = torch.softmax(self.attention_IDH1(feats), dim=0)
        A_atrx = torch.softmax(self.attention_ATRX(feats), dim=0)
        A_p53  = torch.softmax(self.attention_P53(feats), dim=0)

        # Aggregate features
        M_idh1 = torch.sum(A_idh1 * feats, dim=0)
        M_atrx = torch.sum(A_atrx * feats, dim=0)
        M_p53  = torch.sum(A_p53 * feats, dim=0)

        # Stack features
        M = torch.stack([M_idh1, M_atrx, M_p53], dim=0)

        # Classifier
        out = self.classifier(M.mean(dim=0))

        if return_attention:
            return out, [A_idh1, A_atrx, A_p53]

        return out
        
# ==========================================
# 1. THE MIL DATASET
# ==========================================
class WSIFeatureDataset(Dataset):
    def __init__(self, features_dir, csv_path):
        # FIX 1: Force Pandas to read slide_id as a string so it keeps the "000"
        self.labels_df = pd.read_csv(csv_path, dtype={'slide_id': str})
        
        # Set the slide ID as the index so we can look up rows instantly
        self.labels_df.set_index('slide_id', inplace=True)
        
        # Find all extracted feature files
        all_files = glob.glob(os.path.join(features_dir, "*.pt"))
        
        # SAFETY FILTER
        self.feature_files = []
        for file_path in all_files:
            slide_name = os.path.basename(file_path).replace(".pt", "")
            
            if slide_name in self.labels_df.index:
                self.feature_files.append(file_path)
            else:
                print(f"  [Warning] Found {slide_name}.pt but it is missing from the CSV. Skipping.")

        print(f"Successfully matched {len(self.feature_files)} slides with CSV labels.")

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        file_path = self.feature_files[idx]
        slide_name = os.path.basename(file_path).replace(".pt", "")
        
        # Load the [N, 2048] feature tensor
        features = torch.load(file_path, weights_only=True)
        
        # Lookup the specific row for this slide
        row = self.labels_df.loc[slide_name]
        
        # FIX 2: Exact column names from your CSV
        label_list = [
            row['IDH1R132H'], 
            row['ATRX'], 
            row['P53']
        ]
        
        # Convert to a PyTorch tensor
        label = torch.tensor(label_list, dtype=torch.float32)
        
        return features, label, slide_name
# 2. THE TRAINING LOOP
# ==========================================
def train_model():
    print(f"Initializing Training on {DEVICE}...")
    
    # 1. Setup Data
    dataset = WSIFeatureDataset(FEATURES_DIR, CSV_PATH)
    
    # CRITICAL: batch_size MUST be 1 so PyTorch doesn't try to stack 
    # WSIs with different numbers of patches!
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 2. Setup Model
    model = MILModel_ResNet50().to(DEVICE)
    model.train()
    
    # 3. Setup Loss and Optimizer
    # Since you have 3 separate binary biomarkers (IDH1, ATRX, P53), 
    # BCEWithLogitsLoss is the mathematically correct loss function.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # ... (Keep your dataset and optimizer setup the same) ...
    
    epochs = 10
    best_loss = float('inf') # Set initial best loss to infinity
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        epoch_loss = 0.0
        model.train() # Ensure model is in training mode
        
        for batch_idx, (features, label, slide_name) in enumerate(dataloader):
            features = features.squeeze(0).to(DEVICE)
            label = label.to(DEVICE)
            
            # Forward Pass
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions.unsqueeze(0), label)
            
            # Backward Pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 2 == 0:
                print(f"  Processed {slide_name[0]} | Loss: {loss.item():.4f}")
                
        # Calculate the average loss for this entire epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")
        
        # --- THE CHECKPOINT LOGIC ---
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_path = "best_mil_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  ⭐ Loss improved! Saved new best weights to {save_path}")

    print("\nTraining Complete. Best model weights are safely stored on disk.")

if __name__ == "__main__":
    train_model()
