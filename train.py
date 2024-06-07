import torch
from model import MyDataset, Resnet
from torch.utils.data import DataLoader
from torch import nn


def train():
    dataset = MyDataset(512, 512)
    model = Resnet()
    model.train()
    epoch = 50

    optimizer = torch.optim.SGD(model.parameters(), lr=7e-3, momentum=0.9)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, )
    criterion = nn.CrossEntropyLoss()
    for e in range(epoch):
        for b, (x, y) in enumerate(loader):
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            print("epoch:", e, "batch:", b, loss.item())
        scheduler.step()
    torch.save(model.state_dict(), "resnet_epoch_50.pkl")


if __name__ == '__main__':
    train()