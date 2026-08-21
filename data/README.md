# Training data

This project trains on public-domain MIDI transcriptions (Bach fugues, Mozart sonatas).
The raw `.mid` files aren't committed to this repo to keep it lightweight and avoid
redistributing third-party MIDI transcriptions.

To reproduce training:

1. Create a folder (e.g. `bach_fugues/` or `mozart_sonatas/`) at the repo root.
2. Drop in `.mid` files for that composer/dataset.
3. Run `python src/train.py --data bach_fugues`.

`src/data.py` will parse every `.mid` file in the folder you point it at and cache the
extracted (pitch, duration) sequence so re-training doesn't require re-parsing MIDI each time.
