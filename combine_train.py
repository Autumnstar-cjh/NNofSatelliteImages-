import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from PIL import Image
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.init as init



# Residual Block
loss_list = []
residuals = []
l1_loss = nn.L1Loss()
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, targets, transform=None):
        self.root_dir = root_dir
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, f'{idx + 1}.jpg')
        image = Image.open(image_path).convert('RGB')

        target = self.targets[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, target

# Train the model
def trainloop(train_loader):
    num_epoch = 50
    for epoch in range(num_epoch):
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            l1_reg = 0.0
            for param in net.parameters():
                l1_reg += l1_loss(param, torch.zeros_like(param))
            l1_lambda = 0.001
            loss += l1_lambda * l1_reg
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loss_list.append(loss.item())

        # Print the training statistics
        print(
            f"Epoch {epoch + 1}/{num_epoch} - Loss: {running_loss / len(train_loader):.4f} ")

    predicted_train_labels = []
    true_train_labels = []
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            outputs = net(inputs)
            predicted_train_labels += outputs.squeeze().tolist()
            true_train_labels += labels.tolist()
    train_r2 = r2_score(true_train_labels, predicted_train_labels)
    print("Train R-squared:", train_r2)


if __name__ == '__main__':

    # Read Excel file and extract the labels
    train_label_df = pd.read_excel('trainlabel_travel_walk_ratio.xlsx')
    train_labels = train_label_df['target'].tolist()

    # Read Excel file and extract the labels
    test_label_df = pd.read_excel('testlabel.xlsx')
    test_labels = test_label_df['target'].tolist()

    # check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the dataset and transform it
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # randomly flip the image horizontally
        transforms.RandomRotation(10),  # randomly rotate the image by up to 10 degrees
        transforms.ToTensor(),  # convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # randomly flip the image horizontally
        transforms.RandomRotation(10),  # randomly rotate the image by up to 10 degrees
        transforms.ToTensor(),  # convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CustomDataset(root_dir='trainset_zoom=13STseg', targets=train_labels, transform=train_transform)
    test_dataset = CustomDataset(root_dir='testset_zoom=13STseg', targets=test_labels, transform=test_transform)


    # create data loaders to load the data in batches
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8,
                                               shuffle=True, num_workers=8)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8,
                                             shuffle=False, num_workers=8)

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

        def forward(self, x):
            residual = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(residual)
            out = F.relu(out)
            return out

    class ResNet(nn.Module):
        def __init__(self, num_layers=4, num_channels=64, num_classes=1):
            super(ResNet, self).__init__()
            self.in_channels = num_channels
            self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(num_channels)
            self.layer1 = self.make_layer(64, num_layers)
            self.layer2 = self.make_layer(128, num_layers, stride=2)
            self.layer3 = self.make_layer(256, num_layers, stride=2)
            self.layer4 = self.make_layer(512, num_layers, stride=2)
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

        def make_layer(self, out_channels, num_blocks, stride=1):
            layers = []
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
            for _ in range(1, num_blocks):
                layers.append(ResidualBlock(out_channels, out_channels))
            return nn.Sequential(*layers)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out

    # Initialize the model
    net = ResNet().to(device)

    # Define the loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.01,momentum = 0.9)

    trainloop(train_loader)


    def r_squared(y_pred, y_true):
        total_sum_of_squares = torch.sum((y_true - torch.mean(y_true)) ** 2)
        residual_sum_of_squares = torch.sum((y_true - y_pred) ** 2)
        r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
        return r2


    # Test the network on the test dataset
    net.eval()
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            outputs = net(inputs)
            true_labels += labels.tolist()
            predicted_labels += outputs.squeeze().tolist()

    for i in range(len(predicted_labels)):
        if predicted_labels[i] >= 1:
            predicted_labels[i] = 1
        if predicted_labels[i] <= 0.5:
            predicted_labels[i] = 0.5


    predicted_labels = torch.tensor(predicted_labels)
    true_labels = torch.tensor(true_labels)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(true_labels, predicted_labels)
    print("MAE:", mae)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(true_labels, predicted_labels)
    print("MSE:", mse)

    # Calculate R-squared (R^2)
    r2 = r_squared(predicted_labels, true_labels)
    print("R-squared:", r2)

    # Calculate the residuals
    residuals = np.array(true_labels) - np.array(predicted_labels)

    plt.plot(loss_list)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()

    # Plot the residuals
    plt.scatter(true_labels, residuals, s=5)
    plt.xlabel('True Labels')
    plt.ylabel('Prediction Residuals')
    plt.title('Model Prediction Residuals')
    plt.show()


    # Calculate and plot regression line
    regression_line = np.polyfit(true_labels, predicted_labels, 1)
    regression_line_fn = np.poly1d(regression_line)
    plt.scatter(true_labels, predicted_labels, s=5)
    plt.plot(true_labels, regression_line_fn(true_labels), color='red')
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.title('Regression Line')
    plt.show()