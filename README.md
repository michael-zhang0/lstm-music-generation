# LSTM Music Generation

A PyTorch-based generative music model that learns sequential relationships between musical
pitch and duration from MIDI compositions, then generates new melodies using temperature-scaled,
top-k, and pitch-jump-constrained sampling.

## Overview

- Parses MIDI files with `music21`, extracting each note/chord as a `(pitch, duration)` pair
- Trains a 3-layer LSTM (512 hidden units) over 100-note input windows
- Two output heads predict pitch and duration jointly
- Trained separately on Bach fugues and Mozart sonatas
- Generation uses temperature scaling, top-k sampling, and a pitch-jump constraint

## Architecture

```
MIDI file
   |
   v
(pitch, duration) encoding  --music21-->
   |
   v
100-note input sequence
   |
   v
3-layer LSTM (512 hidden units)
   |
   v
256-unit dense layer
   |
   +------------------+
   v                  v
Pitch head        Duration head
   v                  v
pitch logits    duration logits
```

## Project structure

```
lstm-music-generation/
├── src/
│   ├── model.py      # MusicLSTM (shared by train + generate)
│   ├── data.py        # MIDI parsing + sequence building
│   ├── train.py        # Training loop, saves best checkpoint
│   └── generate.py    # Sampling + MIDI export
├── data/
│   └── README.md      # How to point training at your own MIDI files
├── checkpoints/        # Saved model weights (not committed)
├── generated/          # Generated MIDI output (not committed)
├── requirements.txt
└── LICENSE
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

Drop `.mid` files into a folder (e.g. `bach_fugues/` or `mozart_sonatas/`):

```bash
python src/train.py --data mozart_sonatas --epochs 200
```

Uses CUDA automatically if available. The checkpoint with the lowest combined pitch+duration
loss is saved to `checkpoints/best_model.pth`, including the full vocabulary (pitch names,
durations, and their index mappings) so generation can't silently drift out of sync with
whatever the model was actually trained on.

## Generating music

```bash
python src/generate.py --checkpoint checkpoints/best_model.pth --output generated/sample.mid
```

| Flag | What it controls |
|---|---|
| `--num-notes` | Length of the generated sequence |
| `--max-pitch-jump` | Max semitone distance between consecutive notes |

## What changed from the original version

This is a cleanup of a project I originally built in 2025. I still had my actual training and
prediction scripts (via an old ChatGPT conversation where I'd pasted them while debugging), so
this isn't a guess at what the code might have looked like, it's a refactor of the real thing,
with a few bugs fixed that I actually ran into at the time:

- **Duration normalization bug in generation.** The prediction script divided duration by
  `n_vocab` in two places instead of `n_durations`. Training normalized correctly; generation
  didn't. This was a likely contributor to generated pieces sounding faster and less
  rhythmically varied than the source MIDI.
- **Checkpoints didn't save the vocabulary.** Training only saved `{epoch, model_state_dict,
  loss}`. Because pitch/duration vocab size depends on whichever MIDI files got parsed that run,
  loading a checkpoint against a different dataset produced real shape-mismatch errors, e.g.
  `pitch_output.weight: [294, 256]` vs `[238, 256]` and `duration_output.weight: [22, 256]` vs
  `[36, 256]`. Checkpoints now carry `pitchnames`, `durations`, and both index mappings, so
  generation always uses the exact vocabulary training used.
- **Duplicated model definition.** `MusicLSTM` was defined separately inside the training and
  prediction scripts. Editing one without the other caused a similar failure the first time I
  added the duration head (`lstm1.weight_ih_l0: [2048, 1]` vs `[2048, 2]`, from switching a
  pitch-only checkpoint to a pitch+duration model). Now it's one class both scripts import.
- **Pitch-jump constraint made fully reliable.** The original version limited how far a
  generated pitch could jump from the previous note, and worked correctly most of the time, but
  had edge cases where an invalid predicted pitch would slip through. Getting this to work 100%
  of the time was an explicit goal I never finished. This version masks out-of-range pitches and
  renormalizes before sampling, which handles the invalid-pitch edge case directly.
- **Dropped the 4/4-timing post-processing.** A separate experiment tried to force generated
  durations into 4/4 measures. It never fully worked and I moved on to pitch-jump constraints
  instead. Rather than carry over an incomplete feature, I left it out here; it's a reasonable
  thing to revisit if meter-aware generation is worth pursuing later.
- **Best-loss checkpoint saving is preserved, not new.** Early training only saved the final
  epoch's weights. Partway through the project I switched to saving whichever epoch had the
  lowest loss instead, which is what this version does too.

## About the training data

`bach_fugues/` and `mozart_sonatas/` contain the actual MIDI files this project was trained on
(100 files total, ~1.7MB). Bach and Mozart's compositions are both centuries past any copyright
term, so the underlying music is public domain. The `.gitignore` still excludes these by default
since I'm not certain of the provenance or license of this particular MIDI transcription set, and
a from-scratch repo doesn't need the training data included to be useful or to run. If you want
them in your GitHub history anyway, force-add past the gitignore rule:

```bash
git add -f bach_fugues/*.mid mozart_sonatas/*.mid
git commit -m "Add training MIDI data"
```

## Project history

Built over the 2024-2025 school year as a year-long Senior Research project, run in two-week
sprints with mentor check-ins. The early phase explored GANs, then RNNs, then settled on LSTMs
after literature review. An initial attempt following a MAESTRO-dataset tutorial was abandoned in
October after the dataset became inaccessible and environment issues stalled progress; the
project restarted from a different reference implementation
([Classical-Piano-Composer](https://github.com/Skuldur/Classical-Piano-Composer) by Skuldur, see
Credits below).

From there: a multi-week vocabulary size mismatch bug (November), a mode-collapse bug where every
generated file played a single repeated note (December-January, across a CPU run, then a GPU/PyTorch
port), fixed with AI-assisted temperature sampling in January. Duration prediction, k-sampling, and
pitch-jump constraints followed in February and March, along with a manual round of hyperparameter
exploration: 18 generated files across different temperature settings for pitch and duration,
evaluated by ear rather than by loss alone, to find settings that actually sounded more musical.

## Credits

The base training/prediction pipeline was adapted from
[Classical-Piano-Composer](https://github.com/Skuldur/Classical-Piano-Composer) by Skuldur, used
as a starting point after an earlier from-scratch attempt (following a MAESTRO-dataset tutorial)
was abandoned. Everything from that point forward, including the dual pitch/duration prediction
heads, temperature scaling, top-k sampling, and pitch-jump constraints, was debugged, extended, and
iterated on independently.

## Known limitation

Generated music from both the Bach and Mozart checkpoints was still fairly random-sounding, even
holding style constant. The representation here is a flat sequence of (pitch, duration) pairs
with no explicit encoding of measure position, beat, chord structure, or long-term phrase
structure, which likely limits how coherent the LSTM's output can be regardless of training
time.

## License

MIT, see [LICENSE](LICENSE).
