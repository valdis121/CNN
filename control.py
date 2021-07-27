import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from PIL import Image
import torch.nn as nn
import csv
import os
import natsort
from torch.utils.data import Dataset

FileName='result.csv'




class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

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

transform=transforms.Compose([
                                transforms.Resize(size=(224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])
dataset = CustomDataSet("E:\\ai\\exam_data\\val",transform)
test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=1,
                                          shuffle=False)

model=CNNModel()
model.load_state_dict(torch.load("E:\\ai\\Models\\conv_net_model.pth"))

model.eval()
labels={
    '0':"casseteplayer",
    '1':"chainsaw",
    '2':"church",
    '3':"englishspringer",
    '4':"frenchhorn",
    '5':"garbagetruck",
    '6':"gaspump",
    '7':"golfball",
    '8':"parachute",
}

answers = dict()
i=0

with open(FileName, mode='w',newline='') as employee_file:
    employee_writer = csv.writer(employee_file)
    employee_writer.writerow(["fname","class"])

for images in dataset:
    images = images.unsqueeze(0)
    output = model(images)
    pred = torch.argmax(output, 1)
    if i%100==0:
        print(i)
    with open(FileName, mode='a',newline='') as employee_file:
        employee_writer = csv.writer(employee_file)
        employee_writer.writerow([str(dataset.total_imgs[i]), str(labels[str(int(pred))])])
    i+=1






