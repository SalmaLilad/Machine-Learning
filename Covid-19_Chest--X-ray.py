import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import urllib.request

# Download datasets
urllib.request.urlretrieve('http://www-users.math.umn.edu/~jwcalder/MCFAM/train_images64.npz','train_images64.npz')
urllib.request.urlretrieve('http://www-users.math.umn.edu/~jwcalder/MCFAM/train_labels.npz','train_labels.npz')

# Load image and label data
images = np.load('train_images64.npz')['train_images']
labels = np.load('train_labels.npz')['train_labels']

print("Image shape:", images.shape)
print("Label shape:", labels.shape)

# Custom dataset for binary COVID classification
class CustomNumpyDataset(Dataset):
    def __init__(self, data_array, labels_array):
        self.data = torch.from_numpy(data_array).float()
        self.labels = torch.from_numpy(labels_array).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx].unsqueeze(0)  # Add channel dimension
        label_idx = torch.argmax(self.labels[idx]).item()
        binary_label = 1 if label_idx in [2, 3] else 0  # 1: COVID, 0: No-COVID
        return sample, binary_label

# Train/test split
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3, random_state=42)
train_ds = CustomNumpyDataset(train_images, train_labels)
test_ds = CustomNumpyDataset(test_images, test_labels)

# DataLoaders
batch_size = 100
test_batch_size = 500

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, num_workers=1, pin_memory=True)

# CNN model with 2 output classes
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        w = (32, 64, 128, 256)
        self.conv1 = nn.Conv2d(1, w[0], 3, 1)
        self.bn1 = nn.BatchNorm2d(w[0])
        self.conv2 = nn.Conv2d(w[0], w[1], 3, 1)
        self.bn2 = nn.BatchNorm2d(w[1])
        self.conv3 = nn.Conv2d(w[1], w[2], 3, 1)
        self.bn3 = nn.BatchNorm2d(w[2])
        self.conv4 = nn.Conv2d(w[2], w[3], 3, 1)
        self.bn4 = nn.BatchNorm2d(w[3])

        self._to_linear = None
        self._get_linear_input_size()

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.dropout = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(512, 2)  # Binary classification

    def _get_linear_input_size(self):
        x = torch.zeros(1, 1, 64, 64)
        x = self.conv1(x); x = self.bn1(x); x = F.relu(x); x = F.max_pool2d(x, 2)
        x = self.conv2(x); x = self.bn2(x); x = F.relu(x); x = F.max_pool2d(x, 2)
        x = self.conv3(x); x = self.bn3(x); x = F.relu(x); x = F.max_pool2d(x, 2)
        x = self.conv4(x); x = self.bn4(x); x = F.relu(x); x = F.max_pool2d(x, 2)
        self._to_linear = x.numel()

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = F.relu(x); x = F.max_pool2d(x, 2)
        x = self.conv2(x); x = self.bn2(x); x = F.relu(x); x = F.max_pool2d(x, 2)
        x = self.conv3(x); x = self.bn3(x); x = F.relu(x); x = F.max_pool2d(x, 2)
        x = self.conv4(x); x = self.bn4(x); x = F.relu(x); x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Testing function with accuracy return
def test(model, device, test_loader, scheduler):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(target)

    test_loss /= total
    accuracy = 100. * correct / total

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%) [COVID Detection]\n')
    scheduler.step(test_loss)
    return accuracy

# Set device and seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

# Initialize model, optimizer, scheduler
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.6)

# Training loop
epochs = 10  # You can increase for better performance
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    acc = test(model, device, test_loader, scheduler)

# Final accuracy
print(f"Final Test Accuracy: {acc:.2f}%")

# Save the model
torch.save(model.state_dict(), "COVID_binary_classifier.pth")
