"""
model.py

The MusicLSTM architecture, unchanged from the original design:
3-layer LSTM (512 units) -> dense (256) -> separate pitch/duration heads.

This used to be defined independently inside both the training script and
the prediction script. That's what caused the real errors I hit in 2025:

    size mismatch for lstm1.weight_ih_l0:
    copying a param with shape torch.Size([2048, 1]) from checkpoint,
    the shape in current model is torch.Size([2048, 2])

    size mismatch for pitch_output.weight:
    copying a param with shape torch.Size([294, 256]) from checkpoint,
    the shape in current model is torch.Size([238, 256])

The first happened when the LSTM's input size changed (1 feature -> 2,
pitch-only -> pitch+duration) between when a checkpoint was saved and when
it was loaded. The second happened because the pitch/duration vocab size
depends on whatever MIDI files were parsed that run, and nothing forced
training and generation to agree on it. Both are solved by (1) defining
the model once, imported by both scripts, and (2) saving the vocab inside
the checkpoint itself (see train.py).
"""

import torch.nn as nn


class MusicLSTM(nn.Module):
    def __init__(self, n_vocab, n_durations):
        super().__init__()

        self.lstm1 = nn.LSTM(
            2, 512, num_layers=3, dropout=0.3, batch_first=True
        )

        self.batch_norm1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.3)

        self.dense1 = nn.Linear(512, 256)
        self.relu = nn.ReLU()

        self.batch_norm2 = nn.BatchNorm1d(256)

        self.pitch_output = nn.Linear(256, n_vocab)
        self.duration_output = nn.Linear(256, n_durations)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = x[:, -1, :]

        x = self.batch_norm1(x)
        x = self.dropout(x)

        x = self.dense1(x)
        x = self.relu(x)

        x = self.batch_norm2(x)
        x = self.dropout(x)

        pitch = self.pitch_output(x)
        duration = self.duration_output(x)

        return pitch, duration
