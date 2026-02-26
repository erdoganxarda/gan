from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.models.dcgan import Generator
from src.models.rnn_mdn import RNNMDN, generate_unconditional
from src.utils.io import ensure_dir
from src.utils.render import plot_stroke_sequences, render_sequences_to_tensor, save_tensor_grid
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate samples from trained handwriting models.")
    parser.add_argument("--model", choices=["dcgan", "rnn"], required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--max-len", type=int, default=160)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--num-classes", type=int, default=26)
    parser.add_argument("--class-label", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_checkpoint(path: str | Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device)


def _resolve_labels(
    num_samples: int,
    num_classes: int,
    class_label: int,
    device: torch.device,
) -> torch.Tensor:
    if class_label >= 0:
        if class_label >= num_classes:
            raise ValueError(f"class_label={class_label} is out of range for num_classes={num_classes}")
        return torch.full((num_samples,), class_label, device=device, dtype=torch.long)
    return torch.randint(0, num_classes, (num_samples,), device=device, dtype=torch.long)


def _sample_dcgan(args: argparse.Namespace, device: torch.device) -> None:
    ckpt = _load_checkpoint(args.ckpt, device)
    cfg = ckpt.get("config", {})
    latent_dim = int(cfg.get("latent_dim", args.latent_dim))
    conditional = bool(cfg.get("conditional", False))
    num_classes = int(cfg.get("num_classes", args.num_classes))
    label_embed_dim = int(cfg.get("label_embed_dim", 32))

    model = Generator(
        latent_dim=latent_dim,
        conditional=conditional,
        num_classes=num_classes,
        label_embed_dim=label_embed_dim,
    ).to(device)
    model.load_state_dict(ckpt["generator_state"])
    model.eval()

    labels = _resolve_labels(args.num_samples, num_classes, args.class_label, device) if conditional else None
    with torch.no_grad():
        z = torch.randn(args.num_samples, latent_dim, device=device)
        images = model(z, labels=labels)

    save_tensor_grid(images.cpu(), args.out, nrow=int(np.sqrt(args.num_samples)) or 8)

    # Save a simple latent interpolation strip.
    steps = 10
    z0 = torch.randn(1, latent_dim, device=device)
    z1 = torch.randn(1, latent_dim, device=device)
    alphas = torch.linspace(0, 1, steps=steps, device=device).view(-1, 1)
    z_interp = (1.0 - alphas) * z0 + alphas * z1
    interp_labels = None
    if conditional:
        interp_class = args.class_label if args.class_label >= 0 else int(torch.randint(0, num_classes, (1,)).item())
        interp_labels = torch.full((steps,), interp_class, device=device, dtype=torch.long)
    with torch.no_grad():
        interp_images = model(z_interp, labels=interp_labels)

    out_path = Path(args.out)
    interp_path = out_path.with_name(f"{out_path.stem}_interp{out_path.suffix}")
    save_tensor_grid(interp_images.cpu(), interp_path, nrow=steps)

    if conditional:
        labels_path = out_path.with_suffix(".labels.npy")
        np.save(labels_path, labels.detach().cpu().numpy() if labels is not None else np.array([], dtype=np.int64))


def _sample_rnn(args: argparse.Namespace, device: torch.device) -> None:
    ckpt = _load_checkpoint(args.ckpt, device)
    cfg = ckpt.get("config", {})
    conditional = bool(cfg.get("conditional", False))
    num_classes = int(cfg.get("num_classes", args.num_classes))
    class_embed_dim = int(cfg.get("class_embed_dim", 16))

    model = RNNMDN(
        input_dim=3,
        hidden_size=int(cfg.get("hidden_size", 256)),
        num_layers=int(cfg.get("layers", 2)),
        num_mixtures=int(cfg.get("mixtures", 20)),
        dropout=float(cfg.get("dropout", 0.2)),
        conditional=conditional,
        num_classes=num_classes,
        class_embed_dim=class_embed_dim,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    max_len = int(cfg.get("max_len", args.max_len))
    labels = _resolve_labels(args.num_samples, num_classes, args.class_label, device) if conditional else None
    with torch.no_grad():
        sequences = generate_unconditional(
            model=model,
            num_samples=args.num_samples,
            max_len=max_len,
            device=device,
            temperature=args.temperature,
            labels=labels,
        )

    out_path = Path(args.out)
    ensure_dir(out_path.parent)

    npz_path = out_path.with_suffix(".npz")
    payload = {"sequences": sequences.cpu().numpy()}
    if labels is not None:
        payload["labels"] = labels.cpu().numpy()
    np.savez_compressed(npz_path, **payload)

    if args.render:
        images = render_sequences_to_tensor(sequences.cpu().numpy())
        save_tensor_grid(images, out_path, nrow=int(np.sqrt(args.num_samples)) or 8)

    stroke_path = out_path.with_name(f"{out_path.stem}_strokes{out_path.suffix}")
    plot_stroke_sequences([sequences[i].cpu().numpy() for i in range(min(16, args.num_samples))], stroke_path, cols=4)


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "dcgan":
        _sample_dcgan(args, device)
    else:
        _sample_rnn(args, device)


if __name__ == "__main__":
    main()
