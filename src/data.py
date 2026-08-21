"""
data.py

MIDI parsing and sequence preparation, matching the original get_notes()
and prepare_sequences() functions. The only structural change is that
get_notes() takes a folder argument instead of a hardcoded
"mozart_sonatas/*.mid" glob, so the same code trains on Bach or Mozart.
"""

import glob
import pickle

import numpy as np
from music21 import converter, note, chord


def get_notes(midi_folder, cache_path="data/notes"):
    notes = []

    for file in glob.glob(f"{midi_folder}/*.mid"):
        try:
            print(f"\nAttempting to parse: {file}")
            midi = converter.parse(file)
            print(f"Parsing {file}")

            if midi is None:
                continue

            notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(
                        (str(element.pitch), element.duration.quarterLength)
                    )
                elif isinstance(element, chord.Chord):
                    notes.append(
                        (
                            ".".join(str(n) for n in element.normalOrder),
                            element.duration.quarterLength,
                        )
                    )

            print(f"Successfully processed {file}")

        except Exception as e:
            print(f"Error processing {file}: {e}")

    if len(notes) == 0:
        raise ValueError("No notes were extracted")

    print(f"\nTotal notes extracted: {len(notes)}")

    with open(cache_path, "wb") as filepath:
        pickle.dump(notes, filepath)

    return notes


def prepare_sequences(notes, pitchnames, n_vocab, sequence_length=100):
    note_to_int = {n: i for i, n in enumerate(pitchnames)}

    durations = sorted(set(item[1] for item in notes))
    duration_to_int = {d: i for i, d in enumerate(durations)}
    n_durations = len(durations)

    network_input = []
    network_output_pitch = []
    network_output_duration = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]

        pitch_seq = [note_to_int[char[0]] for char in sequence_in]
        duration_seq = [duration_to_int[char[1]] for char in sequence_in]

        network_input.append(list(zip(pitch_seq, duration_seq)))
        network_output_pitch.append(note_to_int[sequence_out[0]])
        network_output_duration.append(duration_to_int[sequence_out[1]])

    network_input = np.array(network_input, dtype=np.float32)

    # Pitch normalized by n_vocab, duration normalized by n_durations.
    # This line was correct in the original training script -- the
    # duration/n_vocab bug only ever showed up on the prediction side
    # (see generate.py).
    network_input /= [float(n_vocab), float(n_durations)]

    network_output_pitch = np.eye(n_vocab)[network_output_pitch].astype(np.float32)
    network_output_duration = np.eye(n_durations)[network_output_duration].astype(np.float32)

    return (
        network_input,
        network_output_pitch,
        network_output_duration,
        durations,
        note_to_int,
        duration_to_int,
    )
