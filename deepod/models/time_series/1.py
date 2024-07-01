import torch
import torch.nn as nn
import torch.optim as optim
import random


# Encoder类定义
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, src):
        # src形状：(batch_size, seq_len, input_dim)
        outputs, (hidden, cell) = self.lstm(src)
        return hidden, cell


# Decoder类定义
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell):
        # input形状：(batch_size, 1, input_dim)
        outputs, (hidden, cell) = self.lstm(input, (hidden, cell))
        predictions = self.fc(outputs.squeeze(1))
        return predictions, hidden, cell


# Seq2Seq整体模型定义
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        output_dim = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, output_dim).to(self.device)

        hidden, cell = self.encoder(src)

        input = trg[:, 0].unsqueeze(1)

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            input = (trg[:, t].unsqueeze(1) if teacher_force else output)

        return outputs


# 主程序
def main():
    # 设定参数
    INPUT_DIM = 10
    OUTPUT_DIM = 10
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 实例化模型
    encoder = Encoder(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS)
    decoder = Decoder(HIDDEN_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # 假设数据
    src = torch.rand(32, 10, 10).to(DEVICE)  # (batch_size, seq_len, input_dim)
    trg = torch.rand(32, 10, 10).to(DEVICE)  # (batch_size, seq_len, output_dim)

    # 模型训练过程
    model.train()
    optimizer.zero_grad()
    output = model(src, trg)
    loss = criterion(output, trg)
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item()}")


if __name__ == "__main__":
    main()
