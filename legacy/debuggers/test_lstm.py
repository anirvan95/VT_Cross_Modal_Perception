import torch
import torch.nn as nn
import numpy as np


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.fc(output[:, -1, :])  # Take the last output
        return output, hidden


# Generate some dummy data
num_samples = 1000
seq_len = 10
input_size = 3
hidden_size = 50
output_size = 2

X = torch.rand(num_samples, seq_len, input_size)
y = torch.rand(num_samples, output_size)

# Instantiate the model
model = LSTMModel(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(10):
    optimizer.zero_grad()
    vals = []
    hidden = (torch.zeros(1, 1, hidden_size),
            torch.zeros(1, 1, hidden_size))
    for i in range(seq_len):
        output, hidden = model(X[:, i:i+1, :], hidden)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

# Test the model
test_input = torch.rand(1, seq_len, input_size)  # 1 sample
hidden = (torch.zeros(1, 1, hidden_size),
            torch.zeros(1, 1, hidden_size))
model.eval()
predictions = []
for i in range(seq_len):
    output, hidden = model(test_input[:, i:i + 1, :], hidden)
    predictions.append(output)

print("Prediction:", predictions)
