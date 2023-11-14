import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from torchvision.io import read_image, ImageReadMode

import numpy as np
import random
import yaml
from PIL import Image

class FitzpatrickDataset(Dataset):
    def __init__(self, df, transform=None, skin_type='multi'):
        self.df = df
        self.transform = transform
        self.skin_type = skin_type
        self.classes = self.get_num_classes()
        self.class_to_idx = self._get_class_to_idx()

    def __len__(self):
        return len(self.df)
    
    def get_num_classes(self):
        #return self.df['label_idx'].unique()
        return self.df['binary_label'].unique()
    
    def _get_class_to_idx(self):
        return {'benign': 0, 'malignant': 1, 'non-neoplastic': 2}
    
    def __getitem__(self, idx):
        image = read_image(self.df.iloc[idx]['Path'], mode=ImageReadMode.RGB)
        image = T.ToPILImage()(image)
        #label = self.df.iloc[idx]['label_idx']
        label = self.df.iloc[idx]['binary_label']

        if(self.skin_type == 'multi'):
            sens_attribute = self.df.iloc[idx]['skin_type']
        elif(self.skin_type == 'binary'):
            sens_attribute = self.df.iloc[idx]['skin_binary']

        if self.transform:
            image = self.transform(image)
        
        return image, label, sens_attribute