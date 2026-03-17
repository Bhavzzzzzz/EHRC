import os
import glob
import torch
import pandas as pd
import cv2

class WSIDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, labels_csv, transform=None, max_patches=500):
        self.root_dir = root_dir
        self.transform = transform
        self.max_patches = max_patches

        df = pd.read_csv(labels_csv)
        self.slides = df['slide_id'].values
        self.labels = df[['IDH1R132H','ATRX','P53']].values.astype(float)

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, idx):
        slide = self.slides[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        patch_paths = glob.glob(os.path.join(self.root_dir, slide, "*.png"))
        patch_paths = patch_paths[:self.max_patches]

        patches = []
        for p in patch_paths:
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.tensor(img).permute(2,0,1).float()/255.0
            patches.append(img)

        if len(patches) == 0:
            patches.append(torch.zeros(3,224,224))

        patches = torch.stack(patches)

        return patches, label