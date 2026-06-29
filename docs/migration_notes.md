# SDFusion Migration Notes

This project keeps `SDFusion-master` read-only and implements the refactored training code under `3D-Diffusion`.

## VQ-VAE

The default VQ-VAE config now uses `architecture: legacy`, matching the original SDFusion VQ-VAE layout:

- `encoder.*`
- `decoder.*`
- `quantize.embedding.weight`
- `quant_conv.*`
- `post_quant_conv.*`

Use:

```bash
python tools/convert_legacy_checkpoint.py --component vqvae --input path/to/vqvae.pth --output runs/converted_vqvae.pt
python tools/evaluate_vqvae.py --config config/defaults/vqvae_snet_chair.yaml --ckpt runs/converted_vqvae.pt --out_dir runs/eval_vqvae
```

The loader strips common wrappers such as `module.` and understands checkpoints with `vqvae`, `model`, or raw state dict roots.

## Diffusion

The refactored diffusion stack supports:

- DDPM training loss.
- DDIM sampling.
- PLMS sampling.
- `conditioning_key` values `concat`, `crossattn`, and `hybrid`.
- Classifier-free guidance through sampler arguments.
- Original SDFusion schedule defaults for ShapeNet latent diffusion: `linear_start=0.00085`, `linear_end=0.012`, `scale_factor=0.18215`.

The current `UNet3D` is still a compact refactored network, not a parameter-name-compatible migration of `openai_model_3d.py`. Legacy diffusion checkpoint conversion is therefore best-effort and mainly useful for inspection until the original UNet block layout is ported.

## Training

VQ-VAE training:

```bash
python tools/train_vqvae.py --config config/defaults/vqvae_snet_chair.yaml --out_dir runs/vqvae_chair
```

Diffusion training:

```bash
python tools/train_diffusion.py --config config/defaults/diffusion_snet_chair.yaml --vqvae_ckpt runs/vqvae_chair/checkpoints/vqvae_last.pt --out_dir runs/diffusion_chair
```

Both scripts write `resolved_config.yaml`, `metrics.jsonl`, and checkpoints. VQ-VAE training also exports reconstruction SDF arrays and PLY meshes on save steps.

## Remaining Gaps

- Full original diffusion UNet parameter compatibility is not complete.
- Text/image/partial-shape datasets and feature encoders are not fully migrated.
- Cross-attention in the compact UNet is a context projection path, not the full original transformer attention stack.
