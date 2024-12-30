import torch
from torch import nn
from torch.nn import init
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data as data
from scipy.io import loadmat

batch_size = 1
num_epoch = 2


class Mydataset(data.Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.idx = list()
        for item in x:
            self.idx.append(item)
        pass

    def __getitem__(self, index):
        input_data = self.idx[index]
        target = self.y[index]
        return input_data, target

    def __len__(self):
        return len(self.idx)

    # 定义网络结构
    # net = nn.Sequential(
    #     nn.Conv1d(in_channels=2001, out_channels=100, kernel_size=1, stride=1, padding=)
    # )


class CNN_AE(nn.Module):
    def __init__(self):
        super(CNN_AE, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding='same', bias=True),
            # 1*2001 -> 32*2001
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2)  # 32*2001 -> 32*1000
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same', bias=True),
            # 32*2000 -> 32*2000
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # 32*2000 -> 32*1000
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same', bias=True),
            # 32*1000 -> 64*1000
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2)  # 64*1000 -> 64*500
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same', bias=True),
            # 64*1000 -> 64*1000
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # 64*1000 -> 64*500
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32000, 16000),
            nn.ReLU(),
            nn.Linear(16000, 8000),
            nn.ReLU(),
            nn.Linear(8000, 2001),
        )

    def forward(self, X):
        y = self.conv1(X)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = y.view(-1, y.size()[0] * y.size()[1])
        y = self.fc(y)
        return y


net = CNN_AE()
for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)
use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    model = net.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-6)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


file = 'E:\\ML_code\\1dCNN_AE\\pythonProject\\data_X.mat'
data = loadmat(file, mat_dtype=True)
x_dat_train = data['train_data_X']
x_dat_train = x_dat_train.astype(np.float32)

file = 'E:\\ML_code\\1dCNN_AE\\pythonProject\\data_y.mat'
data2 = loadmat(file, mat_dtype=True)
y_dat_train = data2['train_data_y']
y_dat_train = y_dat_train.astype(np.float32)
train_datasets = Mydataset(x_dat_train, y_dat_train)

train_data = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size)


def train(net, train_iter, loss, num_epochs, batch_size, optimizer, params=None):
    for epoch in range(num_epochs):
        # 训练网络
        for X, y in train_iter:
            # X = X.view(batch_size, 1, -1)
            y_hat = net(X)
            total_loss = loss(y_hat, y).sum()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            print('epoch %d, train_loss %.10f' % (epoch + 1, total_loss.item()))


train(net, train_data, criterion, num_epoch, batch_size, optimizer, None)

# torch.save(net, 'network_test.pth')


def show_result(n):
    test_x = torch.tensor(x_dat_train[n, :])
    true_x = x_dat_train[n, :]
    test_x = test_x.view(1, 1, -1)
    test_y = net(test_x)
    test_y = test_y.view(-1, 1)
    test_y = test_y.detach().numpy()
    plt.figure()
    plt.subplot(3, 1, 1)
    y = y_dat_train[n, :]
    plt.plot(test_y)
    plt.subplot(3, 1, 2)
    plt.plot(y)
    plt.subplot(3, 1, 3)
    plt.plot(true_x)


show_result(50)

# input = torch.randn(1, 2001)
# output = net(input)

# for X, y in train_data:
#     # X = X.view(batch_size, 1, -1)
#     print(X.size())
#     # print(y.size())
#     print('-'*100)