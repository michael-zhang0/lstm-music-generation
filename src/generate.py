"""
generate.py

Matches the original generate() / generate_notes() / create_midi() flow,
with the following real fixes:

1. Duration normalization bug (confirmed in two places in the old code):
       prediction_duration_input = ... / float(n_vocab)
   should have been:
       prediction_duration_input = ... / float(n_durations)
   This was flagged as a likely cause of the generated music sounding
   "faster" and less rhythmically varied than the source pieces.

2. Vocabulary is loaded from the checkpoint instead of being recomputed
   from whatever data/notes pickle happens to be sitting on disk. The
   old version rebuilt pitchnames/durations from the currently-loaded
   notes file, which is exactly what caused the 294-vs-238 pitch and
   22-vs-36 duration mismatches when generation was run against a
   different dataset than the one a checkpoint was trained on.

3. The pitch-jump constraint is implemented the way it was actually
   intended (see my notes, section 19) but never correctly shipped:
   mask out-of-range pitch probabilities, renormalize, THEN sample.
   The original attempt picked the single highest-probability pitch
   that passed the constraint, which collapsed generation into
   repeating the same note (usually C5) instead of preserving
   diversity. The 4/4-time "adjusted_output" experiment from that same
   era computed a value it never returned, so it's left out here
   rather than carried over as dead code.

Usage:
    python src/generate.py --checkpoint checkpoints/best_model.pth --output generated/sample.mid
"""

import argparse
import random

import numpy as np
import torch
from music21 import instrument, note, stream, chord, duration as m21duration

from model import MusicLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def note_name_to_midi(pitch_str):
    """MIDI note number for a single pitch name, for pitch-jump distance checks.
    Chord strings (e.g. "0.4.7") use the first pitch class as a stand-in."""
    try:
        return note.Note(pitch_str.split(".")[0]).pitch.midi
    except Exception:
        return 60  # fall back to middle C


def apply_temperature(predictions, temperature):
    predictions = np.array(predictions)
    predictions = np.log(predictions + 1e-10) / temperature
    exp_predictions = np.exp(predictions)
    return exp_predictions / np.sum(exp_predictions)


def top_k_sampling(probabilities, k=5):
    top_k_indices = np.argsort(probabilities)[-k:]
    top_k_probs = probabilities[top_k_indices]
    top_k_probs = top_k_probs / np.sum(top_k_probs)
    return int(np.random.choice(top_k_indices, p=top_k_probs))


def mask_pitch_jump(pitch_predictions, previous_pitch, int_to_note, max_jump=7):
    """Zero out pitches more than `max_jump` semitones from the previous
    note, renormalize, and let the caller sample from what's left --
    instead of deterministically picking the single best valid candidate."""
    previous_midi = note_name_to_midi(previous_pitch) if previous_pitch else 60

    mask = np.zeros_like(pitch_predictions)
    for idx in range(len(pitch_predictions)):
        candidate_midi = note_name_to_midi(int_to_note[idx])
        if abs(candidate_midi - previous_midi) <= max_jump:
            mask[idx] = pitch_predictions[idx]

    if mask.sum() > 0:
        return mask / mask.sum()
    return pitch_predictions  # nothing passed the constraint, fall back


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = MusicLSTM(checkpoint["n_vocab"], checkpoint["n_durations"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def generate_notes(
    model,
    checkpoint,
    num_notes=500,
    pitch_temperature=0.65,
    duration_temperature=0.75,
    duration_top_k=5,
    max_pitch_jump=7,
):
    pitchnames = checkpoint["pitchnames"]
    durations = checkpoint["durations"]
    note_to_int = checkpoint["note_to_int"]
    duration_to_int = checkpoint["duration_to_int"]
    sequence_length = checkpoint.get("sequence_length", 100)

    n_vocab = len(pitchnames)
    n_durations = len(durations)

    int_to_note = {i: n for n, i in note_to_int.items()}
    int_to_duration = {i: float(d) for d, i in duration_to_int.items()}

    # Seed with a random window rather than requiring the caller to supply
    # one -- the original relied on re-deriving this from data/notes,
    # which is exactly the coupling that caused the vocab mismatches.
    pitch_pattern = [random.randint(0, n_vocab - 1) for _ in range(sequence_length)]
    duration_pattern = [random.randint(0, n_durations - 1) for _ in range(sequence_length)]

    prediction_output = []
    previous_pitch = None

    for _ in range(num_notes):
        prediction_pitch_input = (
            torch.FloatTensor(pitch_pattern).unsqueeze(0).unsqueeze(2) / float(n_vocab)
        )
        # Fixed: was `/ float(n_vocab)` in the original, which corrupted
        # the duration signal fed back into the model each step.
        prediction_duration_input = (
            torch.FloatTensor(duration_pattern).unsqueeze(0).unsqueeze(2) / float(n_durations)
        )

        prediction_input = torch.cat(
            (prediction_pitch_input, prediction_duration_input), dim=2
        ).to(device)

        with torch.no_grad():
            pitch_pred, duration_pred = model(prediction_input)

        pitch_predictions = torch.softmax(pitch_pred, dim=1).cpu().numpy()[0]
        pitch_predictions = apply_temperature(pitch_predictions, pitch_temperature)
        pitch_predictions = mask_pitch_jump(
            pitch_predictions, previous_pitch, int_to_note, max_jump=max_pitch_jump
        )

        duration_predictions = torch.softmax(duration_pred, dim=1).cpu().numpy()[0]
        duration_predictions = apply_temperature(duration_predictions, duration_temperature)

        # Random sampling for pitch (not argmax -- argmax collapses to
        # repeating the single most likely note)
        pitch_index = int(np.random.choice(len(pitch_predictions), p=pitch_predictions))

        # Top-k sampling for duration
        duration_index = top_k_sampling(duration_predictions, k=duration_top_k)

        pitch_result = int_to_note.get(pitch_index, "C4")
        if not isinstance(pitch_result, str):
            pitch_result = "C4"

        duration_result = int_to_duration.get(duration_index, 1.0)

        prediction_output.append((pitch_result, duration_result))

        pitch_pattern = pitch_pattern[1:] + [pitch_index]
        duration_pattern = duration_pattern[1:] + [duration_index]
        previous_pitch = pitch_result

    return prediction_output


def create_midi(prediction_output, output_path="generated/sample.mid"):
    offset = 0
    output_notes = []

    for pattern, note_duration in prediction_output:
        if not isinstance(pattern, str) or not pattern[0].isalpha():
            print(f"Invalid pitch {pattern}, using default C4")
            pattern = "C4"

        if "." in pattern:
            # Chord: normalOrder pitch classes -> real Note objects
            try:
                pitch_classes = [int(p) for p in pattern.split(".")]
                new_element = chord.Chord(pitch_classes)
            except ValueError:
                new_element = note.Note("C4")
        else:
            new_element = note.Note(pattern)

        new_element.offset = offset

        try:
            new_element.duration = m21duration.Duration(float(note_duration))
        except Exception:
            print(f"Invalid duration {note_duration}, using default 1")
            new_element.duration = m21duration.Duration(1)

        new_element.storedInstrument = instrument.Piano()
        output_notes.append(new_element)

        offset += float(note_duration)

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp=output_path)
    print(f"Saved generated MIDI to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="generated/sample.mid")
    parser.add_argument("--num-notes", type=int, default=500)
    parser.add_argument("--max-pitch-jump", type=int, default=7)
    args = parser.parse_args()

    model, checkpoint = load_checkpoint(args.checkpoint)
    prediction_output = generate_notes(
        model, checkpoint, num_notes=args.num_notes, max_pitch_jump=args.max_pitch_jump
    )
    create_midi(prediction_output, output_path=args.output)
