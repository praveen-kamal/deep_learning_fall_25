# rnn_model.py
import torch
import torch.nn as nn
from spatiotemporal_lstm_cell import SpatioTemporalLSTMCell  # Importing the LSTM cell


class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs, num_classes=36):
        super(RNN, self).__init__()

        self.configs = configs
        self.frame_channel = (
            configs.patch_size * configs.patch_size * configs.img_channel
        )
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(
                    in_channel,
                    num_hidden[i],
                    width,
                    configs.filter_size,
                    configs.stride,
                    configs.layer_norm,
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

        # Classification layer instead of frame prediction
        # Global pooling followed by a fully connected layer to output class logits
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(
            num_hidden[-1], num_classes
        )  # Fully connected layer for classification

    def forward(self, input_seq):
        # Input: sequence of frames (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, _, height, width = input_seq.size()

        h_t = []
        c_t = []
        m_t = []

        # Initialize hidden, cell, and memory states for all layers
        for i in range(self.num_layers):
            h_t.append(
                torch.zeros(batch_size, self.num_hidden[i], height, width).to(
                    input_seq.device
                )
            )
            c_t.append(
                torch.zeros(batch_size, self.num_hidden[i], height, width).to(
                    input_seq.device
                )
            )
            m_t.append(
                torch.zeros(batch_size, self.num_hidden[i], height, width).to(
                    input_seq.device
                )
            )

        # Process each frame in the input sequence
        for t in range(seq_len):
            x_t = input_seq[:, t]  # Current frame in the sequence
            for i in range(self.num_layers):
                if i == 0:
                    h_t[i], c_t[i], m_t[i] = self.cell_list[i](
                        x_t, h_t[i], c_t[i], m_t[i]
                    )
                else:
                    h_t[i], c_t[i], m_t[i] = self.cell_list[i](
                        h_t[i - 1], h_t[i], c_t[i], m_t[i]
                    )

        # Final hidden state from the last layer is used for classification
        final_output = h_t[-1]

        # Apply global average pooling
        pooled_output = self.global_avg_pool(final_output).view(batch_size, -1)

        # Apply the fully connected layer to get class logits
        logits = self.fc(pooled_output)

        return logits
