import torch
import matplotlib.pyplot as plt
import time
import numpy as np
import torchvision


class MNISTNet(torch.nn.Module):

    def __init__(self, n_hidden_neurons):
        super(MNISTNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, n_hidden_neurons)
        self.ac1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ac1(x)
        x = self.fc2(x)
        return x


def run_mnist():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('The current device is:', device)

    mnist_net = MNISTNet(100)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mnist_net.parameters(), lr=1.0e-3)
    batch_size = 100

    test_accuracy_history_cpu = []
    test_loss_history_cpu = []
    test_accuracy_history_gpu = []
    test_loss_history_gpu = []

    MNIST_train = torchvision.datasets.MNIST('./', download=True, train=True)
    MNIST_test = torchvision.datasets.MNIST('./', download=True, train=False)
    X_train = MNIST_train.train_data
    y_train = MNIST_train.train_labels
    X_test = MNIST_test.test_data
    y_test = MNIST_test.test_labels
    print(X_train.dtype, y_train.dtype)
    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    X_test = X_test.float()
    X_train = X_train.float()
    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    X_train = X_train.reshape([-1, 28 * 28])
    X_test = X_test.reshape([-1, 28 * 28])

    cpu_start = time.time()
    for epoch in range(10):
        start = time.time()
        order = np.random.permutation(len(X_train))

        for start_index in range(0, len(X_train), batch_size):
            optimizer.zero_grad()

            batch_indexes = order[start_index:start_index + batch_size]

            X_batch = X_train[batch_indexes]  # .to(device)
            y_batch = y_train[batch_indexes]  # .to(device)

            preds = mnist_net.forward(X_batch)

            loss_value = loss(preds, y_batch)
            loss_value.backward()

            optimizer.step()

        test_preds = mnist_net.forward(X_test)
        test_loss_history_cpu.append(loss(test_preds, y_test))
        accuracy = (test_preds.argmax(dim=1) == y_test).float().mean()
        test_accuracy_history_cpu.append(accuracy)
        end = time.time()
        print(f" Epoch # {epoch}, accuracy = {accuracy.data}, time took = {end - start}")
    cpu_end = time.time()
    print(f"Time taken for all epochs on cpu: {cpu_end - cpu_start}")

    mnist_net = mnist_net.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    X_test_gpu = X_test.to(device)
    y_test_gpu = y_test.to(device)
    list(mnist_net.parameters())
    list(optimizer.state.values())

    gpu_start = time.time()
    for epoch in range(10):
        start = time.time()
        order = np.random.permutation(len(X_train))

        for start_index in range(0, len(X_train), batch_size):
            optimizer.zero_grad()

            batch_indexes = order[start_index:start_index + batch_size]

            X_batch = X_train[batch_indexes].to(device)
            y_batch = y_train[batch_indexes].to(device)

            preds = mnist_net.forward(X_batch)

            loss_value = loss(preds, y_batch)
            loss_value.backward()

            optimizer.step()

        test_preds = mnist_net.forward(X_test_gpu)
        test_loss_history_gpu.append(loss(test_preds, y_test_gpu))
        accuracy = (test_preds.argmax(dim=1) == y_test_gpu).float().mean()
        test_accuracy_history_gpu.append(accuracy)
        end = time.time()
        print(f" Epoch # {epoch}, accuracy = {accuracy.data}, time took = {end - start}")
    gpu_end = time.time()
    print(f"Time taken for all epochs on gpu: {gpu_end - gpu_start}")

    test_acc_from_gpu = [item.cpu().detach().numpy() for item in test_accuracy_history_gpu]
    test_loss_from_gpu = [item.cpu().detach().numpy() for item in test_loss_history_gpu]
    test_loss_from_cpu = [item.cpu().detach().numpy() for item in test_loss_history_cpu]
    cpu_time = cpu_end - cpu_start
    gpu_time = gpu_end - gpu_start
    print(f"Time taken for all epochs on cpu: {cpu_time:.3f}s")
    print(f"Time taken for all epochs on gpu: {gpu_time:.3f}s")
    print(f"Time difference for all epochs on cpu and gpu: {abs(gpu_time - cpu_time):.3f}s")
    print(
        f"Time difference ratio for all epochs on cpu and gpu:"
        f" {(max(gpu_time, cpu_time) / min(gpu_time, cpu_time)):.3f}"
    )
    plt.figure()
    plt.plot(test_accuracy_history_cpu)
    plt.plot(test_loss_from_cpu)
    plt.figure()
    plt.plot(test_acc_from_gpu)
    plt.plot(test_loss_from_gpu)
