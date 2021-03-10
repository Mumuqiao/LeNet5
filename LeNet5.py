# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data as Data
import cv2
TrainData = torchvision.datasets.MNIST('./mnist',train=True,
                                        transform=torchvision.transforms.ToTensor(),download=True)
TestData = torchvision.datasets.MNIST('./mnist',train=False,
                                       transform=torchvision.transforms.ToTensor())
# print("train_data:", TrainData.train_data.size())
# print("train_labels:", TrainData.train_labels.size())
# print("test_data:", TestData.test_data.size())

TrainLoader = Data.DataLoader(dataset=TrainData,batch_size=64,shuffle=True)
TestLoader = Data.DataLoader(dataset=TestData,batch_size=64)

# Show the image

# images, lables = next(iter(TrainLoader))
# img = torchvision.utils.make_grid(images, nrow = 50)
# img = img.numpy().transpose(1, 2, 0)
# cv2.imshow('img', img)
# cv2.waitKey(0)


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        #kernel
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        #Linear
        self.fc1 = nn.Linear(16 * 4 *4, 120)    #in the original page, using 32*32 image, but I derectly using 28*28
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)


    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    def num_flat_features(self,x):
        size = x.size()[1:]
        result = 1
        for sizez in size:
            result *= sizez
        return result

def main():
    device = torch.device('cuda')
    net = LeNet5().to(device)
    Closs = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters())
    epochs = 30
    # print(net)
    for epo in range(epochs):
        sum_loss = 0.0
        train_correct = 0
        for data in TrainLoader:
            inputs, labels = data
            inputs, labels = torch.autograd.Variable(inputs).cuda(), torch.autograd.Variable(labels).cuda()
            optim.zero_grad()
            output = net(inputs)
            loss = Closs(output, labels)
            loss.backward()
            optim.step()
            _, id = torch.max(output.data, 1)
            sum_loss += loss.data
            train_correct += torch.sum(id == labels.data)

        print('[%d,%d] loss:%.03f' % (epo + 1, epochs, sum_loss / len(TrainLoader)))
        print('        correct:%.03f%%' % (100 * train_correct / len(TrainData)))
    net.eval()
    test_correct = 0
    for data in TestLoader:
        inputs, lables = data
        inputs, lables = torch.autograd.Variable(inputs).cuda(), torch.autograd.Variable(lables).cuda()
        outputs = net(inputs)
        _, id = torch.max(outputs.data, 1)
        test_correct += torch.sum(id == lables.data)
    print("correct:%.3f%%" % (100 * test_correct / len(TestData)))
    torch.save(net.state_dict(), "parameter.pkl")    # save

if __name__ == '__main__':
    # main()                            #train
    net = LeNet5().to(device='cuda')    #or you can choose cpu
    net.load_state_dict(torch.load('parameter.pkl'))   # load
    net.eval()
    test_correct = 0
    rs = 0
    for data in TestLoader:
        inputs, lables = data
        #print(inputs.size())
        temp = inputs
        inputs, lables = torch.autograd.Variable(inputs).cuda(), torch.autograd.Variable(lables).cuda()
        outputs = net(inputs)
        #print(outputs.data)
        _, id = torch.max(outputs.data, 1)
        print(id)
        img = torchvision.utils.make_grid(temp,nrow=8)    #show the test image
        img = img.numpy().transpose(1,2,0)
        cv2.imshow('img'+str(rs), img)
        cv2.waitKey(0)
        ++rs
        test_correct += torch.sum(id == lables.data)
    print("correct:%.3f%%" % (100 * test_correct / len(TestData)))