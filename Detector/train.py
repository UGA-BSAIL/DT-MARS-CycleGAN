import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import DetData, DetBoxData, DetLineData
from model import DetModel, DetLineModel
from loss import DetLossBatch
from tqdm import tqdm

# Configuration
data_dir = '/home/myid/zw63397/Projects/Crop_Detect/data/sim_data'  # Update this path
num_epochs = 50
batch_size = 256
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2))
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset and DataLoader
train_dataset = DetBoxData(data_dir=data_dir, transform=train_transform, split='train')
val_dataset = DetBoxData(data_dir=data_dir, transform=val_transform, split='val')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
print('dataset len:', len(train_loader), len(val_loader))

# Model
model = DetLineModel().to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
detloss = torch.nn.SmoothL1Loss()

best_val_loss = float('inf')

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        preds = model(images)
        # print(preds.shape)
        # print(labels.shape)

        # Compute loss
        loss = detloss(10*preds, 10*labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    train_loss = running_loss / len(train_loader.dataset)

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            preds = model(images)

            # Compute loss
            loss = detloss(10*preds, 10*labels)
            val_running_loss += loss.item() * images.size(0)
        val_loss = val_running_loss / len(val_loader.dataset)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Save the model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model_save_path = 'models/box/det_model_{}_{:.4f}.pth'.format(epoch+1, best_val_loss)
        torch.save(model.state_dict(), model_save_path)
        print(f'Best model saved to {model_save_path}')

