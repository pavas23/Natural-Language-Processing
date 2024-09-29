'''Supervised Model for open information extraction task.'''

import torch.nn as nn
from TorchCRF import CRF


class BiLSTM_CRF(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiLSTM_CRF, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirection
        self.crf = CRF(output_dim, batch_first=True)

    def forward(self, inputs, tags=None, mask=None):
        lstm_out, _ = self.lstm(inputs)  # Shape: (batch_size, seq_len, hidden_dim * 2)
        emissions = self.hidden2tag(lstm_out)  # Shape: (batch_size, seq_len, output_dim)

        if tags is not None:  # If tags are provided, calculate the loss
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        
        # If tags are not provided, return the predicted tags
        return self.crf.decode(emissions, mask=mask)
    
