import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size),
                                     nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(hidden_size, input_size),
                                     nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def anomaly_score(x, model):
    x = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        loss = model(x) - x
        loss = torch.mean(torch.pow(loss, 2), dim=1)
    return loss


# 训练模型
def train_autoencoder(data_loader, learning_rate, num_epochs):
    input_size = data_loader.dataset.tensors[0].shape[1]
    hidden_size = 32  # 隐藏层节点数
    model = Autoencoder(input_size, hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for data in data_loader:
            inputs, _ = data
            optimizer.zero_grad()

            outputs = model(inputs.float())
            loss = criterion(outputs, inputs.float())
            loss.backward()
            optimizer.step()

        print('Epoch [{}/{}], Loss {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    return model
