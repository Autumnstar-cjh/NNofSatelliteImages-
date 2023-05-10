import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd
import multiprocessing as mp
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report



loss_list = []
accuracy_trainging_list = []
# Train the model
def trainloop(train_loader):
    for epoch in range(30):
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        for i, data in enumerate(train_loader, 0):
            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            l1_reg = 0
            for param in net.parameters():
                l1_reg += torch.sum(torch.abs(param))
            loss += l1_lambda * l1_reg
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            loss_list.append(loss.item())
            if i % 40 == 39:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        accuracy = 100.0 * total_correct / total_samples
        accuracy_trainging_list.append(accuracy)
        print('Accuracy on epoch %d: %d %%' % (epoch + 1, accuracy))
    print('Finished Training')

if __name__ == '__main__':

    # Read Excel file and extract the labels
    train_label_df = pd.read_excel('trainlabel.xlsx')
    train_labels = train_label_df['label'].tolist()

    # Read Excel file and extract the labels
    test_label_df = pd.read_excel('testlabel.xlsx')
    test_labels = test_label_df['label'].tolist()

    # check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the dataset and transform it
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # resize the image to (256, 256)
        transforms.RandomHorizontalFlip(),  # randomly flip the image horizontally
        transforms.RandomRotation(10),  # randomly rotate the image by up to 10 degrees
        transforms.ToTensor(),  # convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    test_transform = transforms.Compose(
        [transforms.Resize((256, 256)),  # resize the image to (256, 256)
        transforms.RandomHorizontalFlip(),  # randomly flip the image horizontally
        transforms.RandomRotation(10),  # randomly rotate the image by up to 10 degrees
        transforms.ToTensor(),  # convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    # load the training set
    train_dataset = datasets.ImageFolder(root='trainset',
                                          transform=train_transform)

    # load the test set
    test_dataset = datasets.ImageFolder(root='testset',
                                       transform=test_transform)


    # create data loaders to load the data in batches
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16,
                                               shuffle=True, num_workers=8)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16,
                                             shuffle=False, num_workers=8)
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(32 * 64 * 64, 256)
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            x = self.pool(nn.functional.relu(self.conv1(x)))
            x = self.pool(nn.functional.relu(self.conv2(x)))
            x = x.view(-1, 32 * 64 * 64)
            x = nn.functional.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Initialize the model
    net = Net().to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)
    l1_lambda = 0.001

    trainloop(train_loader)

    # Test the network on the test dataset
    net.eval()
    total = 0
    correct = 0
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            true_labels += labels.tolist()
            predicted_labels += predicted.tolist()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test accuracy:', correct / total)

    # Calculate precision and recall
    report = classification_report(true_labels, predicted_labels, zero_division=1)

    print(report)


    plt.plot(loss_list)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()

    plt.plot(accuracy_trainging_list)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Training accuracy Curve')
    plt.show()