import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from torchvision.io import read_image, ImageReadMode

import numpy as np
import random
import yaml
from PIL import Image


class OASISDataset(Dataset):
    def __init__(self, df, sens_attribute=None, transform=None, age_type=None):
        assert sens_attribute is not None
        assert age_type is not None

        self.df = df
        self.transform = transform
        self.sens_attribute = sens_attribute
        self.age_type = age_type
        self.classes = self.get_num_classes()
        self.class_to_idx = self._get_class_to_idx()

    def __len__(self):
        return len(self.df)

    def get_num_classes(self):
        return self.df["CDR"].unique()

    def _get_class_to_idx(self):
        # return {0:'Non-Demented',
        #         1:'Very Mild-Dementia',
        #         2:'Mild-Dementia',
        #         3:'Moderate-Dementia'
        #         }
        return {
            0: "Non-Demented",
            1: "Dementia",
        }

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["Path"]
        image = np.load(img_path).astype(np.uint8)
        image = Image.fromarray(image).convert("RGB")
        label = torch.tensor(self.df.iloc[idx]["CDR"]).to(torch.int64)

        if self.sens_attribute == "gender":
            sens_attribute = self.df.iloc[idx]["Gender"]
        elif self.sens_attribute == "age":
            if self.age_type == "multi":
                sens_attribute = self.df.iloc[idx]["Age_multi"]
            elif self.age_type == "binary":
                sens_attribute = self.df.iloc[idx]["Age_binary"]
            else:
                raise NotImplementedError("Age type not implemented")

        if self.transform:
            image = self.transform(image)

        return image, label, sens_attribute
