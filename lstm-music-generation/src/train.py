"""
train.py

Matches the original train_network() / train() flow. Three real changes
from the 2025 version:

1. MusicLSTM is imported from model.py instead of being redefined here,
   so it can never drift out of sync with generate.py.
2. The checkpoint saves pitchnames/durations/note_to_int/duration_to_int/
   n_vocab/n_durations alongside the weights. The original only saved
   {epoch, model_state_dict, loss}, which is what caused the pitch_output
   294-vs-238 / duration_output 22-vs-36 shape mismatches whenever
   generation was run against a differently-parsed MIDI set.
3. The best-loss checkpoint (section 24 of my notes -- I wrote the
   intended code for this but never actually wired it into the real
   training loop, which only saved on the final epoch) is now real.

Usage:
    python src/train.py --data mozart_sonatas --epochs 200
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data import get_notes, prepare_sequences
from model import MusicLSTM

if torch.cuda.is_available():
    print("CUDA is available!")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")


def train_network(data_folder, epochs=200, batch_size=512, checkpoint_path="checkpoints/best_model.pth"):
    notes = get_notes(data_folder)

    pitchnames = sorted(set(item[0] for item in notes))
    n_vocab = len(pitchnames)

    (
        network_input,
        network_output_pitch,
        network_output_duration,
        durations,
        note_to_int,
        duration_to_int,
    ) = prepare_sequences(notes, pitchnames, n_vocab)

    n_durations = len(durations)

    model = MusicLSTM(n_vocab, n_durations).to(device)

    train(
        model,
        network_input,
        network_output_pitch,
        network_output_duration,
        pitchnames,
        durations,
        note_to_int,
        duration_to_int,
        n_vocab,
        n_durations,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_path=checkpoint_path,
    )


def train(
    model,
    network_input,
    network_output_pitch,
    network_output_duration,
    pitchnames,
    durations,
    note_to_int,
    duration_to_int,
    n_vocab,
    n_durations,
    epochs=200,
    batch_size=512,
    checkpoint_path="checkpoints/best_model.pth",
):
    print("training network")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    criterion_pitch = nn.CrossEntropyLoss()
    criterion_duration = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters())

    network_input_t = torch.FloatTensor(network_input).to(device)
    network_output_pitch_t = torch.LongTensor(
        np.argmax(network_output_pitch, axis=1)
    ).to(device)
    network_output_duration_t = torch.LongTensor(
        np.argmax(network_output_duration, axis=1)
    ).to(device)

    dataset = TensorDataset(network_input_t, network_output_pitch_t, network_output_duration_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0
        print(f"Starting Epoch {epoch + 1}")

        for batch_input, batch_pitch, batch_duration in dataloader:
            optimizer.zero_grad()

            pitch_pred, duration_pred = model(batch_input)

            loss_pitch = criterion_pitch(pitch_pred, batch_pitch)
            loss_duration = criterion_duration(duration_pred, batch_duration)
            loss = loss_pitch + loss_duration

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {average_loss}")

        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "loss": best_loss,
                    "pitchnames": pitchnames,
                    "durations": durations,
                    "note_to_int": note_to_int,
                    "duration_to_int": duration_to_int,
                    "n_vocab": n_vocab,
                    "n_durations": n_durations,
                    "sequence_length": 100,
                },
                checkpoint_path,
            )
            print(f"  -> saved new best checkpoint ({best_loss:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Folder of .mid files, e.g. mozart_sonatas")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--checkpoint-path", default="checkpoints/best_model.pth")
    args = parser.parse_args()

    train_network(
        data_folder=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint_path,
    )
