'''Supervised Model for open information extraction task.'''

import torch.nn as nn
from TorchCRF import CRF


class BiLSTM_CRF(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiLSTM_CRF, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True, num_layers=2)
        
        # Additional FC layers for more complex feature extraction
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

        self.activation = nn.ReLU()  # Activation function
        self.dropout = nn.Dropout(0.5)  # Regularization
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)

        # CRF layer
        self.crf = CRF(output_dim, batch_first=True)

    def forward(self, inputs, tags=None, mask=None):
        lstm_out, _ = self.lstm(inputs)
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.batch_norm(lstm_out.transpose(1, 2)).transpose(1, 2)

        # Use more non-linear transformations before passing to CRF
        fc_output = self.activation(self.fc1(lstm_out))
        fc_output = self.activation(self.fc2(fc_output))
        emissions = self.fc3(fc_output)  # Final emission scores for CRF

        if tags is not None:  # If tags are provided, calculate the loss
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        
        # If tags are not provided, return the predicted tags
        return self.crf.decode(emissions, mask=mask)
