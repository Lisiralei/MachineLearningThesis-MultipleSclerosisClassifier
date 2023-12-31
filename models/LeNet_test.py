import torch
import matplotlib.pyplot as plt
import random
import numpy as np
import torchvision


class LeNet5(torch.nn.Module):
    def __init__(self, activation='tanh', pooling='avg', conv_size=5, use_batch_norm=False):
        super(LeNet5, self).__init__()

        self.conv_size = conv_size
        self.use_batch_norm = use_batch_norm

        if activation == 'tanh':
            activation_function = torch.nn.Tanh()
        elif activation == 'relu':
            activation_function = torch.nn.ReLU()
        else:
            raise NotImplementedError

        if pooling == 'avg':
            pooling_layer = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        elif pooling == 'max':
            pooling_layer = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError

        if conv_size == 5:
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=1)
            self.conv2 = self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=1)
        elif conv_size == 3:
            self.conv1_1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
            self.conv1_2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
            self.conv2_1 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
            self.conv2_2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        else:
            raise NotImplementedError

        self.act1 = activation_function
        self.bn1 = torch.nn.BatchNorm2d(num_features=16)
        self.pool1 = pooling_layer

        self.act2 = activation_function
        self.bn2 = torch.nn.BatchNorm2d(num_features=32)
        self.pool2 = pooling_layer

        self.fc1 = torch.nn.Linear(8 * 8 * 32, 256)
        self.act3 = activation_function

        self.fc2 = torch.nn.Linear(256, 32)
        self.act4 = activation_function

        self.fc3 = torch.nn.Linear(32, 10)

    def forward(self, x):
        if self.conv_size == 5:
            x = self.conv1(x)
        elif self.conv_size == 3:
            x = self.conv1_2(self.conv1_1(x))
        x = self.act1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.pool1(x)

        if self.conv_size == 5:
            x = self.conv2(x)
        elif self.conv_size == 3:
            x = self.conv2_2(self.conv2_1(x))
        x = self.act2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.act4(x)
        x = self.fc3(x)

        return x


def train(net, X_train, y_train, X_test, y_test, batch_size):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0e-3)

    batch_size = batch_size

    test_accuracy_history = []
    test_loss_history = []

    X_test = X_test.to(device)
    y_test = y_test.to(device)

    for epoch in range(30):
        order = np.random.permutation(len(X_train))
        for start_index in range(0, len(X_train), batch_size):
            optimizer.zero_grad()
            net.train()

            batch_indexes = order[start_index:start_index + batch_size]

            X_batch = X_train[batch_indexes].to(device)
            y_batch = y_train[batch_indexes].to(device)

            preds = net.forward(X_batch)

            loss_value = loss(preds, y_batch)
            loss_value.backward()

            optimizer.step()

        net.eval()
        test_preds = net.forward(X_test)
        test_loss_history.append(float(loss(test_preds, y_test).data.cpu()))
        net.zero_grad()

        accuracy = (test_preds.argmax(dim=1) == y_test).float().mean().data.cpu()
        test_accuracy_history.append(float(accuracy))

        print(accuracy)
    print('---------------')
    return test_accuracy_history, test_loss_history


def run_lenet(batch_size=100):
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    CIFAR_train = torchvision.datasets.CIFAR10('./', download=True, train=True)
    CIFAR_test = torchvision.datasets.CIFAR10('./', download=True, train=False)

    X_train = torch.FloatTensor(CIFAR_train.data)
    y_train = torch.LongTensor(CIFAR_train.targets)
    X_test = torch.FloatTensor(CIFAR_train.data)
    y_test = torch.LongTensor(CIFAR_train.targets)

    len(y_train), len(y_test)

    #plt.imshow(X_train[0, :, :])
    #plt.show()
    print(y_train[0])

    X_train /= 255
    X_test /= 255

    print(f'{X_train.shape}, {y_train.shape}')

    plt.figure(figsize=(20, 2))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow(X_train[i])
        print(y_train[i], end=' ')

    X_train = X_train.permute(0, 3, 1, 2)
    X_test = X_test.permute(0, 3, 1, 2)
    print(X_train.shape)

    accuracies = {}
    losses = {}

    accuracies['tanh'], losses['tanh'] = \
        train(LeNet5(activation='tanh', conv_size=5),
              X_train, y_train, X_test, y_test, batch_size)

    accuracies['relu'], losses['relu'] = \
        train(LeNet5(activation='relu', conv_size=5),
              X_train, y_train, X_test, y_test, batch_size)

    #accuracies['relu_3'], losses['relu_3'] = \
    #    train(LeNet5(activation='relu', conv_size=3),
    #          X_train, y_train, X_test, y_test, batch_size)

    #accuracies['relu_3_max_pool'], losses['relu_3_max_pool'] = \
        #train(LeNet5(activation='relu', conv_size=3, pooling='max'),
        #      X_train, y_train, X_test, y_test, batch_size)

    #accuracies['relu_3_max_pool_bn'], losses['relu_3_max_pool_bn'] = \
        #train(LeNet5(activation='relu', conv_size=3, pooling='max', use_batch_norm=True),
         #     X_train, y_train, X_test, y_test, batch_size)

    plt.figure(figsize=(10,10))
    for experiment_id in accuracies.keys():
        plt.plot(accuracies[experiment_id], label=experiment_id)
    plt.legend()
    plt.title('Validation Accuracy')

    for experiment_id in losses.keys():
        plt.plot(losses[experiment_id], label=experiment_id)
    plt.legend()
    plt.title('Validation Loss')
