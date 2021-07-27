import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils import data
from torch.autograd import Variable


dataset = dsets.ImageFolder(root='E:\\ai\\exam_data\\train',
                            transform=transforms.Compose([
                                transforms.Resize(size=(224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))

n = len(dataset)
a = int(n * 0.8)
b = n - a
train_dataset, test_dataset = data.random_split(dataset, (a, b))


num_classes = 10
batch_size = 32

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3)
        self.relu1 = nn.ReLU()

        self.c2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3)
        self.relu2 = nn.ReLU()

        self.c3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3)
        self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(in_features=256 * 7 * 7, out_features=1024)
        self.relu4 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.relu5 = nn.ReLU()

        self.fc3 = nn.Linear(in_features=512, out_features=256)
        self.relu6 = nn.ReLU()

        self.fc4 = nn.Linear(in_features=256, out_features=412)
        self.relu7 = nn.ReLU()

        self.fc5 = nn.Linear(in_features=412, out_features=9)

    def forward(self, x):

        x = self.c1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)



        x = self.c2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)



        x = self.c3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)



        x = x.view(x.size(0), -1)


        x = self.fc1(x)
        x = self.relu4(x)

        x = self.fc2(x)
        x = self.relu5(x)

        x = self.fc3(x)
        x = self.relu6(x)

        x = self.fc4(x)
        x = self.relu7(x)

        x = self.fc5(x)

        return x


model = CNNModel()

if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

total_step = len(train_loader)

loss_list = []
acc_list = []
num_epochs = 12
for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):

        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)


        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        if torch.cuda.is_available():
            correct = (predicted.cpu() == labels.cpu()).sum()
        else:
            correct = (predicted == labels).sum()
        acc_list.append(correct // total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (100 * correct // total)))
MODEL_STORE_PATH = "E:\\ai\\Models\\"
torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.pth')
model.eval()


with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        if torch.cuda.is_available():
            correct += (predicted.cpu() == labels.cpu()).sum()
        else:
            correct += (predicted == labels).sum()

    print('Test Accuracy of the model: {} %'.format((100 * correct // total)))




