# Handwriting Gen v1

Compare two handwriting generation approaches on a shared EMNIST letters domain:
- Offline image generation: **DCGAN** (`28x28` grayscale letters)
- Online-like sequence generation: **RNN-MDN** over synthetic stroke trajectories derived from EMNIST

The repo supports training both models, generating samples, and running a lightweight comparison pipeline.

## Project Structure

```text
handwriting-gen/
  data/
  src/
    datasets/
      emnist.py
      emnist_to_strokes.py
    models/
      dcgan.py
      rnn_mdn.py
      classifier_cnn.py
    train_dcgan.py
    train_rnn.py
    train_classifier.py
    sample.py
    eval.py
    utils/
      io.py
      seed.py
      render.py
  notebook.ipynb
  README.md
  requirements.txt
```

## Environment Setup

Target runtime is **Python 3.11**.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Training and Evaluation Commands

Run from repository root (`handwriting-gen/`).

1. Train realism classifier:

```bash
python -m src.train_classifier \
  --data-dir data \
  --out-dir runs/classifier \
  --epochs 12 \
  --batch-size 256 \
  --seed 42
```

2. Train DCGAN (unconditional baseline):

```bash
python -m src.train_dcgan \
  --data-dir data \
  --out-dir runs/dcgan \
  --epochs 40 \
  --batch-size 128 \
  --latent-dim 100 \
  --seed 42
```

3. Train conditional DCGAN (optional):

```bash
python -m src.train_dcgan \
  --data-dir data \
  --out-dir runs/dcgan_cond \
  --epochs 40 \
  --batch-size 128 \
  --latent-dim 100 \
  --conditional \
  --num-classes 26 \
  --label-embed-dim 32 \
  --seed 42
```

4. Train RNN-MDN (unconditional baseline):

```bash
python -m src.train_rnn \
  --strokes-path data/processed/emnist_letters_strokes_len160_seed42.npz \
  --data-dir data \
  --out-dir runs/rnn \
  --epochs 60 \
  --batch-size 128 \
  --mixtures 20 \
  --max-len 160 \
  --seed 42
```

5. Train conditional RNN-MDN + force cache rebuild (optional):

```bash
python -m src.train_rnn \
  --strokes-path data/processed/emnist_letters_strokes_len160_seed42.npz \
  --data-dir data \
  --out-dir runs/rnn_cond \
  --epochs 60 \
  --batch-size 128 \
  --mixtures 20 \
  --max-len 160 \
  --conditional \
  --num-classes 26 \
  --class-embed-dim 16 \
  --rebuild-strokes-cache \
  --seed 42
```

6. Generate samples:

```bash
python -m src.sample \
  --model dcgan \
  --ckpt runs/dcgan/best.pt \
  --num-samples 64 \
  --out reports/samples_dcgan.png

python -m src.sample \
  --model rnn \
  --ckpt runs/rnn/best.pt \
  --num-samples 64 \
  --out reports/samples_rnn.png \
  --render
```

For conditional checkpoints, you can optionally pin class labels:

```bash
python -m src.sample \
  --model dcgan \
  --ckpt runs/dcgan_cond/best.pt \
  --num-samples 64 \
  --class-label 3 \
  --out reports/samples_dcgan_cond.png
```

7. Evaluate comparison metrics:

```bash
python -m src.eval \
  --dcgan-ckpt runs/dcgan/best.pt \
  --rnn-ckpt runs/rnn/best.pt \
  --classifier-ckpt runs/classifier/best.pt \
  --num-samples 5000 \
  --out reports/metrics.json
```

`src.eval` auto-detects whether checkpoints are conditional and samples labels accordingly.

## Expected Artifacts

- `runs/classifier/best.pt`: classifier checkpoint
- `runs/dcgan/best.pt`: DCGAN checkpoint
- `runs/rnn/best.pt`: RNN-MDN checkpoint
- `reports/samples_dcgan.png`: GAN sample grid
- `reports/samples_dcgan_interp.png`: GAN latent interpolation strip
- `reports/samples_rnn.png`: rendered stroke sample grid
- `reports/samples_rnn_strokes.png`: raw stroke plots
- `reports/metrics.json`: comparison metrics
- `reports/metrics.csv`: CSV version of metrics

Conditional sampling may also produce:
- `reports/*.labels.npy` for DCGAN sampled labels
- `reports/*.npz` containing RNN sequences (and labels when conditional)

## Metrics in `reports/metrics.json`

| Metric | DCGAN | RNN |
|---|---|---|
| Classifier confidence (mean) | 0.706 | 0.603 |
| Classifier confidence (p80) | 0.952 | - |
| Class entropy | 3.174 | 2.607 |
| Mean pen lifts per sequence | - | 30.4 |
| Mean stroke length | - | 8.76 |
| Stroke smoothness (mean abs turn, rad) | - | 0.868 |

Higher class entropy indicates more diverse class coverage. Higher classifier confidence indicates more letter-like samples.

## Reproducibility

- Deterministic split file: `data/processed/splits_letters_seed42.npz`
- Cached stroke data: `data/processed/emnist_letters_strokes_len160_seed42.npz`
- Stroke cache now stores a version marker and `max_len`; incompatible caches are rebuilt.
- Each run saves `config.json` plus metrics logs in its output directory.

## Notebook

Open `notebook.ipynb` after running training/sample/eval.

The notebook is checkpoint-driven and aligned with `src/` implementations:
- RNN uses `src.models.rnn_mdn` (MDN-based sequence generation)
- GAN uses `src.models.dcgan` (image-space DCGAN)

## Stroke Conversion Pipeline

EMNIST bitmap images are converted to `(dx, dy, pen_lift)` stroke sequences for RNN training:

1. **Binarize & skeletonize**: `skimage.morphology.skeletonize` thins ink to 1-pixel-wide paths.
2. **Graph traversal**: build pixel adjacency graph, extract connected components, split at branch/endpoint nodes.
3. **Biased path ordering**: chain strokes with a deterministic cost (distance + backward penalty + teleport penalty) to favor human-like left-to-right progression.
4. **Delta encoding**: convert absolute `(x, y)` positions to `(dx, dy)`, clip to `[-1, 1]`, append a `pen_lift` flag at each stroke boundary.
5. **Fixed-length padding**: pad/truncate to `max_len=160` steps; store in a compressed `.npz` cache.

## Key Hyperparameters

| | DCGAN | RNN-MDN |
|---|---|---|
| Latent / input dim | 100 | 3 |
| Hidden units | - | 256 |
| LSTM layers | - | 2 |
| MDN mixtures | - | 20 |
| Batch size | 128 | 128 |
| Learning rate | 2e-4 | 1e-3 |
| Epochs | 40 | 60 |
| Optimizer | Adam (beta1=0.5) | Adam |
| Gradient clip | - | 1.0 |
| Dropout | 0.2 (D) | 0.2 |

## Limitations

- Stroke sequences are derived from image skeletons and approximate pen trajectories rather than true online handwriting capture.
- The DCGAN produces 28x28 images; upscaling will show pixelation.
- Conditional generation is optional and class-level only; writer/style conditioning is not implemented.
