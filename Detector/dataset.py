import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle

class DetData(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = sorted([os.path.join(data_dir, "images", fname) for fname in os.listdir(os.path.join(data_dir, "images")) if fname.endswith(".jpg")])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = img_path.replace('/images', '/labels').replace('.jpg', '.txt')

        # Load image
        img = Image.open(img_path)

        # Load label
        boxes = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                _, x, y, w, h = map(float, line.split())
                # Convert from YOLO format to [x_min, y_min, x_max, y_max]
                x_min = x - w / 2
                y_min = y - h / 2
                x_max = x + w / 2
                y_max = y + h / 2
                boxes.append([x_min, y_min, x_max, y_max])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)

        return img, boxes


class DetBoxData(Dataset):
    def __init__(self, data_dir, transform=None, split='train', split_ratio=0.8):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.split_ratio = split_ratio
        self.image_paths = sorted([os.path.join(data_dir, "images", fname) for fname in os.listdir(os.path.join(data_dir, "images")) if fname.endswith(".jpg")])

        if split in ['train', 'val']:
            self._perform_split()

    def _perform_split(self):
        total_size = len(self.image_paths)
        split_index = int(total_size * self.split_ratio)  # Calculate the split index based on the ratio

        # Use the calculated split_index for slicing
        if self.split == 'train':
            self.image_paths = self.image_paths[:split_index]
        elif self.split == 'val':
            self.image_paths = self.image_paths[split_index:]


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = img_path.replace('/images', '/labels').replace('.jpg', '.txt')

        # Load image
        img = Image.open(img_path)

        # Load label
        x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                _, x, y, w, h = map(float, line.split())
                # Convert from YOLO format to [x_min, y_min, x_max, y_max]
                x_min = x - w / 2
                y_min = y - h / 2
                x_max = x + w / 2
                y_max = y + h / 2
                x_mins.append(x_min)
                y_mins.append(y_min)
                x_maxs.append(x_max)
                y_maxs.append(y_max)

        # Calculate the big bounding box that includes all objects
        label = [min(x_mins), min(y_mins), max(x_maxs), max(y_maxs)]
        label = torch.as_tensor(label, dtype=torch.float32)

        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)

        return img, label


class DetLineData(Dataset):
    def __init__(self, data_dir, transform=None, split='train', split_ratio=0.8):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.split_ratio = split_ratio
        self.image_paths = sorted([os.path.join(data_dir, "images", fname) for fname in os.listdir(os.path.join(data_dir, "images")) if fname.endswith(".jpg")])

        if split in ['train', 'val']:
            self._perform_split()

    def _perform_split(self):
        total_size = len(self.image_paths)
        split_index = int(total_size * self.split_ratio)  # Calculate the split index based on the ratio

        # Use the calculated split_index for slicing
        if self.split == 'train':
            self.image_paths = self.image_paths[:split_index]
        elif self.split == 'val':
            self.image_paths = self.image_paths[split_index:]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = img_path.replace('/images', '/labels').replace('.jpg', '.txt')

        # Load image
        img = Image.open(img_path)

        # Load label
        centers = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                _, x, y, w, h = map(float, line.split())
                centers.append((x, y))

        centers_sorted = sorted(centers, key=lambda x: x[0])
        left_center = centers_sorted[0]
        right_center = centers_sorted[-1]
        label = torch.tensor([*left_center, *right_center], dtype=torch.float32)

        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)

        return img, label

if __name__=='__main__':
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2))
    ])

    # Instantiate and create data loader
    data_dir = "/home/myid/zw63397/Projects/Crop_Detect/data/sim_data"
    dataset = DetLineData(data_dir=data_dir, transform=transform, split='val')

    print(len(dataset))
    img, label = dataset[47]
    print(img.shape, label.shape)
    print(label)
    image_width, image_height = img.shape[2], img.shape[1]

    fig, ax = plt.subplots(1)
    ax.imshow(img.permute(1,2,0))
    ax.axis('off')

    ax.plot([label[0] * image_width, label[2] * image_width],
            [label[1] * image_height, label[3] * image_height], 'r-')
    
    # x_min, y_min, x_max, y_max = label
    # x_min *= image_width
    # y_min *= image_height
    # x_max *= image_width
    # y_max *= image_height
    # rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
    # ax.add_patch(rect)

    plt.show()