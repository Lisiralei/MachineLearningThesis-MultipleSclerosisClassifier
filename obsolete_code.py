"""
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)
"""

"""
    X_train, y_train = next(iter(msclr_train_dataloader))
    #print(X_train)
    print(y_train)
    #y_train = torch.LongTensor(msclr_train.targets)
    #X_test, y_test = next(iter(msclr_test_dataloader))
    #y_test = torch.LongTensor(msclr_test.targets)

    #len(y_train), len(y_test)

    #print(y_train[0])

    #X_train /= 255
    #X_test /= 255

    #print(f'{X_train.shape}, {y_train.shape}')
    print('Initial shape:', X_train.shape)

    #X_train = X_train.permute(0, 3, 2, 1)
    #X_test = X_test.permute(0, 3, 2, 1)
    print('Transformed shape:', X_train.shape)
    plt.figure(figsize=(20, 2))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow(X_train[i])
        print(y_train[i], end=' ')
    #plt.show()

    #X_train = X_train.permute(0, 3, 2, 1)
    #X_test = X_test.permute(0, 3, 2, 1)
    print('To turn back:',X_train.shape)

    #X_train = X_train.permute(0, 3, 1, 2)
    #X_test = X_test.permute(0, 3, 1, 2)
    #print('Target shape:',X_train.shape)
"""

"""        for epoch in range(epochs):
            print('Epoch: ', epoch)
            order = np.random.permutation(len(X_train))
            for start_index in range(0, len(X_train), batch_size):
                print('New Batch')

                self.train()

                batch_indexes = order[start_index:start_index + batch_size]

                X_batch = X_train[batch_indexes].to(device)
                y_batch = y_train[batch_indexes].to(device)

                predictions = self.forward(X_batch)

                loss_value = loss(predictions, y_batch)
                optimizer.zero_grad()

                loss_value.backward()
                optimizer.step()

            self.eval()
            test_preds = self.forward(X_test)
            test_loss_history.append(float(loss(test_preds, y_test).data.cpu()))
            self.zero_grad()
"""
