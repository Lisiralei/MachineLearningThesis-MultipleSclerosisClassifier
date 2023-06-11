import random
from datetime import datetime
import os

import torch
import matplotlib.pyplot as plt
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import dataloader


class SclerosisCNN(torch.nn.Module):
    def __init__(self, pooling='avg', conv_size=5, use_batch_norm=False, verbosity_level=0):
        super(SclerosisCNN, self).__init__()

        self.conv_size = conv_size
        self.use_batch_norm = use_batch_norm
        self.verbosity_level = verbosity_level

        activation_function = torch.nn.ReLU()

        if pooling == 'avg':
            pooling_layer = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        elif pooling == 'max':
            pooling_layer = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError

        if conv_size == 5:
            self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=1)
            self.conv2 = self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=1)
        elif conv_size == 3:
            self.conv1_1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
            self.conv1_2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
            self.conv2_1 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.conv2_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        else:
            raise NotImplementedError

        self.act1 = activation_function
        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.pool1 = pooling_layer

        self.act2 = activation_function
        self.bn2 = torch.nn.BatchNorm2d(num_features=64)
        self.pool2 = pooling_layer

        self.fc1 = torch.nn.Linear(262144, 256)
        self.act3 = activation_function

        self.fc2 = torch.nn.Linear(256, 32)
        self.act4 = activation_function

        self.fc3 = torch.nn.Linear(32, 2)

    def forward(self, x):
        if self.conv_size == 5:
            x = self.conv1(x)
        elif self.conv_size == 3:
            x = self.conv1_2(self.conv1_1(x))
        if self.verbosity_level > 2:
            print('After First convolutional layer:', x.shape)

        x = self.act1(x)
        if self.verbosity_level > 2:
            print('After First activational layer:', x.shape)

        if self.use_batch_norm:
            x = self.bn1(x)
        if self.verbosity_level > 2:
            print('After First batch normalizing layer:', x.shape)
        x = self.pool1(x)
        if self.verbosity_level > 2:
            print('After First pooling layer:', x.shape)

        if self.conv_size == 5:
            x = self.conv2(x)
        elif self.conv_size == 3:
            x = self.conv2_2(self.conv2_1(x))
        if self.verbosity_level > 2:
            print('After Second convolutional layer:', x.shape)

        x = self.act2(x)

        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.pool2(x)
        if self.verbosity_level > 2:
            print('After Second pooling layer:', x.shape)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        if self.verbosity_level > 2:
            print('After Transform:', x.shape)

        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.act4(x)
        x = self.fc3(x)

        return x

    def initialize_train(
            self, train_loader,
            epochs, batch_size,
            save_frequency, on_cuda=False,
            validation_loader=None):
        if on_cuda:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            if self.verbosity_level > 0:
                print('Cuda is available:', torch.cuda.is_available())
            self.to(device)
        else:
            device = torch.device('cpu')

        loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=1.0e-2)

        now = datetime.now()
        n_total_steps = len(train_loader)
        accuracy_history = list()
        current_abs_path = os.path.abspath('./')
        if not os.path.exists(os.path.join(current_abs_path, 'model_cache')):
            os.mkdir(os.path.join(current_abs_path, 'model_cache'))
        PATH = os.path.join(current_abs_path, f'model_cache\\{now.strftime("%Y_%m_%d_%H-%M-%S")}')
        if not os.path.exists(PATH):
            os.mkdir(PATH)
        for epoch in range(epochs):
            if self.verbosity_level > 0:
                print(
                    f'Epoch [{epoch + 1}/{epochs}]',
                    ', Batches: ' if self.verbosity_level > 1 else ' processing;',
                    sep=''
                )
            for i, (images, labels) in enumerate(train_loader):

                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = self(images)
                loss_value = loss(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()

                if self.verbosity_level > 1:
                    if (i + 1) % 10 == 0:
                        print(f'\tBatch: [{i + 1}/{n_total_steps}], Loss: {loss_value.item():.4f}')
                    elif (i + 1) == len(train_loader):
                        print(f'\tBatch: [{i + 1}/{n_total_steps}], Loss: {loss_value.item():.4f}')

            if (epoch + 1) % save_frequency == 0:
                filename = f'epoch-{epoch+1}_cnn.pth'
                torch.save(self.state_dict(), os.path.join(PATH, filename))
                if self.verbosity_level > 0:
                    print(
                        f'Model weights saved for epoch {epoch+1}/{epochs},\n',
                        f'path = \n {PATH}' if self.verbosity_level > 1 else ''
                        )

            if validation_loader:
                accuracy_history.append(
                    self.test_accuracy(validation_loader, on_cuda=on_cuda)
                )

        print('---------------')
        print('Finished Training')
        filename = 'epoch-final_cnn.pth'
        torch.save(self.state_dict(), os.path.join(PATH, filename))
        if self.verbosity_level > 0:
            print(
                f'Model weights saved for the final model, '
                f'epochs completed: {epochs},\n',
                f'path = \n {PATH}' if self.verbosity_level > 1 else ''
            )

        return accuracy_history

    def test_accuracy(self, test_loader, classes=('no', 'yes'), on_cuda=False):
        if on_cuda:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.to(device)
        else:
            device = torch.device('cpu')

        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for i in range(len(classes))]
            n_class_samples = [0 for i in range(len(classes))]
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = self(images)
                # max returns (value ,index)
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

                for i in range(len(images)):
                    label = labels[i]
                    prediction = predicted[i]
                    if label == prediction:
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1

            acc = dict()
            acc['general'] = 100.0 * n_correct / n_samples
            if self.verbosity_level > 0:
                print(f'Accuracy of the network: {acc["general"]} %')

            for i in range(len(classes)):
                acc[classes[i]] = 100.0 * n_class_correct[i] / n_class_samples[i]
                if self.verbosity_level > 0:
                    print(f'Accuracy of {classes[i]}: {acc[classes[i]]} %')

        return classes, acc


def run_cnn(dataset_path, epochs=10, batch_size=100, save_frequency=5, verbosity=0, with_independent_test=False):
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

    data_transform = transforms.Compose([
        # Resize the images to 128x128
        transforms.Resize(size=(256, 256)),
        transforms.Grayscale(),
        # Turn the image into a torch.Tensor
        transforms.ToTensor()  # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0

    ])

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    dataset_path = dataset_path

    msclr_train_data = datasets.ImageFolder(dataset_path + "\\train", transform=data_transform)
    msclr_train_dataloader = dataloader.DataLoader(msclr_train_data, shuffle=True, batch_size=batch_size)

    msclr_validation_data = datasets.ImageFolder(dataset_path + "\\validation", transform=data_transform)
    msclr_validation_dataloader = dataloader.DataLoader(msclr_validation_data, shuffle=True,  batch_size=batch_size)

    if with_independent_test:
        msclr_test_data = datasets.ImageFolder(dataset_path + "\\test", transform=data_transform)
        msclr_test_dataloader = dataloader.DataLoader(msclr_test_data, shuffle=False, batch_size=batch_size)

    if verbosity > 1:
        print('Train dataloader: ', len(msclr_train_dataloader), ' batches with ~', batch_size, ' items each', sep='')
        print('Validation dataloader: ', len(msclr_validation_dataloader), ' batches with ~', batch_size, ' items each', sep='')
        if with_independent_test:
            print('Test dataloader: ', len(msclr_test_dataloader), ' batches with ~', batch_size, ' items each', sep='')

    cnn_instance = SclerosisCNN(pooling='avg', conv_size=3, use_batch_norm=True, verbosity_level=verbosity)

    accuracy_in_training = cnn_instance.initialize_train(
        msclr_train_dataloader, epochs,
        validation_loader=msclr_validation_dataloader,
        save_frequency=save_frequency,
        batch_size=batch_size, on_cuda=True
    )

    independent_test_accuracy = None
    if with_independent_test:
        independent_test_accuracy = cnn_instance.test_accuracy(
            msclr_test_dataloader, on_cuda=True
        )

    # Freeing the instance and cache after learning
    del cnn_instance
    with torch.no_grad():
        torch.cuda.empty_cache()

    return accuracy_in_training, independent_test_accuracy



