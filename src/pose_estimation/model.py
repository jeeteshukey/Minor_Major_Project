import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=60, hidden_size=256, num_classes=3):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,          # deeper model
            batch_first=True,
            dropout=0.3
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),       # 🔥 slightly reduced (better generalization)
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, features)

        x = x.unsqueeze(1)  
        # → (batch_size, 1, input_size)

        lstm_out, (hn, cn) = self.lstm(x)

        # Take last layer hidden state
        out = hn[-1]

        out = self.fc(out)

        return out