import os
from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import re

class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.transform = transform
        self.img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

        entries = []

        for fname in os.listdir(img_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in self.img_extensions:
                match = re.search(r'(\d+)', fname)
                frame_no = int(match.group(1)) if match else -1
                entries.append((os.path.join(img_dir, fname),
                                fname[:-len(ext)],
                                frame_no))

        entries.sort(key=lambda x: x[2])
        self.img_paths, self.img_names, _ = zip(*entries)

        # self.label_encoder = LabelEncoder()
        # self.encoded_labels = self.label_encoder.fit_transform(self.labels)


    def get_label_encoder(self):
      return self.label_encoder

    def __len__(self):
        return len(self.img_paths)

    # def get_encoded_labels(self):
    #     return self.encoded_labels

    def get_img_names(self):
        return list(self.img_names)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        # label = self.encoded_labels[idx]
        name  = self.img_names[idx]

        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, name

