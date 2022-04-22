from torchvision import datasets, transforms

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Download and load the training data
train_set = datasets.MNIST('DATA_MNIST/', download=True, train=True, transform=transforms.ToTensor())
test_set = datasets.MNIST('DATA_MNIST/', download=True, train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

'''
training_data = enumerate(train_loader)
batch_idx, (images, labels) = next(training_data)
print(type(images)) # Checking the datatype
print(images.shape) # the size of the image
print(labels.shape) # the size of the labels
'''

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        # Convolutional Neural Network Layer
        self.convolutaional_neural_network_layers = nn.Sequential(
            nn.Conv2d(
              in_channels=1,
              out_channels=15,
              kernel_size=5
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=15,
                out_channels=30,
                kernel_size=5
            ),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )

        # Linear layer
        self.linear_layers = nn.Sequential(
          nn.Linear(in_features=480, out_features=64),
          nn.ReLU(),
          nn.Dropout(p=0.5),
          nn.Linear(in_features=64, out_features=10),
          nn.LogSoftmax()
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.convolutaional_neural_network_layers(x)
        # After we get the output of our convolutional layer we must flatten it or rearrange the output into a vector
        x = x.view(x.size(0), -1)
        # Then pass it through the linear layer
        x = self.linear_layers(x)
        return x


if __name__ == '__main__':
    model = Network()

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    epochs = 20
    train_loss, val_loss = [], []
    accuracy_total_train, accuracy_total_val = [], []

    for epoch in range(epochs):

        total_train_loss = 0
        total_val_loss = 0

        model.train()

        total = 0
        # training our model
        for idx, (image, label) in enumerate(train_loader):

            # image, label = image.to(device), label.to(device)

            optimizer.zero_grad()

            pred = model(image)

            loss = criterion(pred, label)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

            pred = torch.nn.functional.softmax(pred, dim=1)
            for i, p in enumerate(pred):
                if label[i] == torch.max(p.data, 0)[1]:
                    total = total + 1

        accuracy_train = total / len(train_set)
        accuracy_total_train.append(accuracy_train)

        total_train_loss = total_train_loss / (idx + 1)
        train_loss.append(total_train_loss)

        # validating our model
        model.eval()
        total = 0
        for idx, (image, label) in enumerate(test_loader):
            # image, label = image.cuda(), label.cuda()
            pred = model(image)
            loss = criterion(pred, label)
            total_val_loss += loss.item()

            pred = torch.nn.functional.softmax(pred, dim=1)
            for i, p in enumerate(pred):
                if label[i] == torch.max(p.data, 0)[1]:
                    total = total + 1

        accuracy_val = total / len(test_set)
        accuracy_total_val.append(accuracy_val)

        total_val_loss = total_val_loss / (idx + 1)
        val_loss.append(total_val_loss)

        if epoch % 1 == 0:
            print("Epoch: {}/{}  ".format(epoch, epochs),
                  "Training loss: {:.4f}  ".format(total_train_loss),
                  "Testing loss: {:.4f}  ".format(total_val_loss),
                  "Train accuracy: {:.4f}  ".format(accuracy_train),
                  "Test accuracy: {:.4f}  ".format(accuracy_val))

