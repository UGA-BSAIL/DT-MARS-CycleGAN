import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
from matplotlib.patches import Rectangle
# import torchvision.transforms as transforms
import json 

from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train',rate=1.0):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/real_block' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/simu_pure_1k' % mode) + '/*.*'))
        self.files_A = random.sample(self.files_A,int(len(self.files_A)*rate))
        self.files_B = random.sample(self.files_B,int(len(self.files_B)*rate))

        with open(os.path.join(root, '%s/label_pure_1k.json'%mode),'r', encoding='UTF-8') as f:
            self.label_B = json.load(f)

    def __getitem__(self, index):
        item_A = np.load(self.files_A[index % len(self.files_A)])
        item_B = np.load(self.files_B[index % len(self.files_B)],allow_pickle=True)
        file_name = self.files_B[index%len(self.files_B)]
        file_name = file_name.split('/')[-1]
        label_b = self.label_B[file_name]
        return {'A': item_A, 'B': item_B, 'B_label':label_b}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    

class CropDataset(Dataset):
    def __init__(self, root, transforms_=None, rate=1.0):
        self.transform = transforms_

        self.files_A = sorted(glob.glob(os.path.join(root, 'real_data/real_org') + '/*.jpg'))
        # self.files_A = self.files_A[:2400]
        self.files_B = sorted(glob.glob(os.path.join(root, 'sim_data/images') + '/*.jpg'))
        self.files_A = random.sample(self.files_A,int(len(self.files_A)*rate))
        self.files_B = random.sample(self.files_B,int(len(self.files_B)*rate))
        # print(len(self.files_A), len(self.files_B))

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    
    def __getitem__(self, index):
        item_A = self.readimage(self.files_A[index % len(self.files_A)])
        item_B = self.readimage(self.files_B[index % len(self.files_B)])

        if self.transform:
            item_A = self.transform(item_A)
            item_B = self.transform(item_B)

        file_name = self.files_B[index%len(self.files_B)]
        lbl_name = file_name.replace('/images/', '/labels/').replace('.jpg', '.txt')
        label_b = self.readlabels(lbl_name)
        A_fname = self.files_A[index % len(self.files_A)]
        return {'A': item_A, 'B': item_B, 'B_label':label_b, 'B_fname':file_name, 'A_fname':A_fname}

    def readimage(self, path):
        img = Image.open(path).convert("RGB")  # Open an image file and ensure it has 3 channels
        return img

    def readlabels(self, path):
        x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
        with open(path, 'r') as file:
            for line in file:
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
        return label



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision import transforms

    # Initialize the dataset
    root = "/home/myid/zw63397/Projects/Crop_Detect/data"  # replace with your dataset root directory
    transforms_ = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2))
    ])

    dataset = CropDataset(root, transforms_=transforms_)
    print(len(dataset))

    # Get one sample
    sample = dataset[0]

    # Print the sample
    print("Image A Shape: ", sample['A'].shape)  # If ToTensor() is used, it will print the shape of the tensor.
    print("Image B Shape: ", sample['B'].shape)  # If ToTensor() is used, it will print the shape of the tensor.
    print("Image B Labels: ", sample['B_label'])
    print('B_name', sample['B_fname'])
    image_width, image_height = sample['B'].shape[2], sample['B'].shape[1]

    plt.subplot(1, 2, 1)
    plt.imshow(sample['A'].permute(1, 2, 0))  # Need to permute to (Height, Width, Channels) for visualization
    plt.title('Image A')

    plt.subplot(1, 2, 2)
    plt.imshow(sample['B'].permute(1, 2, 0))  # Need to permute to (Height, Width, Channels) for visualization
    x_min, y_min, x_max, y_max = sample['B_label']
    x_min *= image_width
    y_min *= image_height
    x_max *= image_width
    y_max *= image_height
    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    plt.title('Image B')

    plt.show()
