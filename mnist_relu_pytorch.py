import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

BATCH_SIZE = 20

trainset = torchvision.datasets.MNIST(root='./dataset', train=True,
                                      transform=transform, download=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True)

testset = torchvision.datasets.MNIST(root='./dataset', train=False,
                                     transform=transform, download=True)

testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False)


class Net(torch.nn.Module):
    def __init__(self, INPUT_FEATURES, HIDDEN, OUTPUT_FEATURES):
        super().__init__()
        self.fc1 = torch.nn.Linear(INPUT_FEATURES, HIDDEN)
        self.fc2 = torch.nn.Linear(HIDDEN, OUTPUT_FEATURES)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


INPUT_FEATURES = 28 * 28
HIDDEN = 12
OUTPUT_FEATURES = 10
EPOCHS = 2

net = Net(INPUT_FEATURES, HIDDEN, OUTPUT_FEATURES)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(1, EPOCHS + 1):
    running_loss = 0.0
    for count, item in enumerate(trainloader, 1):
        inputs, labels = item
        inputs = inputs.reshape(-1, 28 * 28)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if count % 500 == 0:
            print(f'#{epoch}, data: {count * 20}, running_loss: {running_loss / 500:1.3f}')
            running_loss = 0.0

print('Finished')

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs = inputs.reshape(-1, 28 * 28)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += len(outputs)
        correct += (predicted == labels).sum().item()

print(f'correct: {correct}, accuracy: {correct} / {total} = {correct / total}')
