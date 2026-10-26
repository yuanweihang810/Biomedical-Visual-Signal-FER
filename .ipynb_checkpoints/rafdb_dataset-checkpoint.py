# paper2/rafdb_dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset

# RAF-DB label → 我们的 EMOTIONS 索引
RAF2OUR = {1: 5, 2: 2, 3: 1, 4: 3, 5: 4, 6: 0, 7: 6}


class RAFDBDataset(Dataset):
    def __init__(self, items, img_dir, tf):
        self.items = items      # list of (filename, label)
        self.img_dir = img_dir
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fn, lab = self.items[idx]
        img = Image.open(os.path.join(self.img_dir, fn)).convert('RGB')
        if self.tf:
            img = self.tf(img)
        return img, lab


def load_rafdb(data_root):
    img_dir = os.path.join(data_root, 'Image/aligned')
    label_file = os.path.join(data_root, 'EmoLabel/list_patition_label.txt')
    train_items, test_items = [], []
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            stem, lab = parts[0], int(parts[1])
            fn = stem.replace('.jpg', '_aligned.jpg')
            our_lab = RAF2OUR[lab]
            if stem.startswith('train'):
                train_items.append((fn, our_lab))
            else:
                test_items.append((fn, our_lab))
    return train_items, test_items, img_dir