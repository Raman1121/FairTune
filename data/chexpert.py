import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from torchvision.io import read_image, ImageReadMode

import numpy as np
import random
import yaml
from PIL import Image

class ChexpertDataset(Dataset):
    def __init__(self, df, sens_attribute=None, transform=None, age_type='binary'):

        assert sens_attribute is not None

        self.df = df
        self.transform = transform
        self.sens_attribute = sens_attribute
        self.age_type = age_type
        self.classes = self.get_num_classes()
        self.class_to_idx = self._get_class_to_idx()

    def __len__(self):
        return len(self.df)
    
    def get_num_classes(self):
        return self.df['No Finding'].unique()
    
    def _get_class_to_idx(self):
        return {'Healthy':0,
                'Not Healthy':1,
                }
    
    def __getitem__(self, idx):
        image = read_image(self.df.iloc[idx]['Path'], mode=ImageReadMode.RGB)
        image = T.ToPILImage()(image)
        label = torch.tensor(self.df.iloc[idx]['No Finding']).to(torch.int64)

        if(self.sens_attribute == 'gender'):
            sens_attribute = self.df.iloc[idx]['Sex']
        elif(self.sens_attribute == 'age'):
            if(self.age_type == 'multi'):
                sens_attribute = self.df.iloc[idx]['Age_multi']
            elif(self.age_type == 'binary'):
                sens_attribute = self.df.iloc[idx]['Age_binary']
        elif(self.sens_attribute == 'age_sex'):
            sens_attribute = self.df.iloc[idx]['Multi_age_multi_sens']

        if self.transform:
            image = self.transform(image)
        
        return image, label, sens_attribute