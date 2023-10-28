import torch
import torch.nn as nn
from torchsummary import summary


# 多层感知机
class MLP(nn.Module):
    def __init__(self):

        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, 512)
        self.fc6 = nn.Linear(512, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))

        x = self.fc6(x)
        return x


class SimpleRNN(nn.Module):
    ###########################设置全局变量##################################
    num_time_steps = 16    
    input_size = 4 
    hidden_size = 16
    output_size = 1
    num_layers = 1
    def __init__(self, input_size=4, hidden_size=16, num_layers=1, output_size=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        for p in self.rnn.parameters():
          nn.init.normal_(p, mean=0.0, std=0.001)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):
       out, hidden_prev = self.rnn(x, hidden_prev)
       # [b, seq, h]
       out = out.view(-1, 16)
       out = self.linear(out)#[seq,h] => [seq,3]
       out = out.unsqueeze(dim=0)  # => [1,seq,3]
       return out, hidden_prev




if __name__ == '__main__':
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = SimpleRNN(input_size=4,hidden_size=16,output_size=1).to(device)
    hidden_prev = torch.zeros(1, 1, 16).to(device)
    inputs = torch.randn([1, 1, 4]).to(device)

    outputs, hidden_prev = model(inputs, hidden_prev)

    summary()