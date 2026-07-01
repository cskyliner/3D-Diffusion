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

The default VQ-VAE loss remains the original lightweight SDF target, L1 reconstruction plus vector-quantization loss. Optional geometry terms are configured under `vqvae_loss`: occupancy BCE, surface-weighted L1, normal alignment, and multiscale L1.

## Diffusion

The refactored diffusion stack supports:

- DDPM training loss.
- DDPM ancestral sampling with clipping, temperature, mask/x0 conditioning, callbacks, intermediates, progress display, and optional reduced-step sampling.
- DDIM sampling with uniform/quad timestep discretization, precomputed sigmas/alphas, eta, clipping, mask/x0 conditioning, score-corrector hook, callbacks, and intermediates.
- PLMS sampling with pseudo improved Euler on the first step, Adams-Bashforth 2/3/4-step updates, deterministic `eta=0` enforcement, callbacks, and intermediates.
- SDFusion/LDM OpenAI-style 3D UNet migration with `diffusion_net.*` state_dict compatibility.
- `conditioning_key` values `concat`, `crossattn`, and `hybrid`.
- Classifier-free guidance through sampler arguments.
- Original SDFusion schedule defaults for ShapeNet latent diffusion: `linear_start=0.00085`, `linear_end=0.012`, `scale_factor=0.18215`.

The default `unet_architecture: legacy_openai` uses the migrated `UNet3DModel` wrapped in `DiffusionUNet`, preserving the original nested `diffusion_net` key layout. The previous compact `UNet3D` remains available as `unet_architecture: compact`.

## Training

VQ-VAE training:

```bash
python tools/train_vqvae.py --config config/defaults/vqvae_snet_chair.yaml --out_dir runs/vqvae_chair
```

Diffusion training:

```bash
python tools/train_diffusion.py --config config/defaults/diffusion_snet_chair.yaml --vqvae_ckpt runs/vqvae_chair/checkpoints/vqvae_last.pt --out_dir runs/diffusion_chair
```

Both scripts write `resolved_config.yaml`, `metrics.jsonl`, and checkpoints. VQ-VAE training exports reconstruction SDF arrays and PLY meshes on save steps. Diffusion training now runs validation loss and sample snapshots, exporting generated SDF arrays, PLY meshes, and `snapshot_evaluation.json` with mesh success rate, SDF statistics, occupancy ratio, and diversity proxy.

## Remaining Gaps

- Text/image/partial-shape datasets and feature encoders are not fully migrated.
- Cross-attention in `legacy_openai` uses the migrated SpatialTransformer3D path when `use_spatial_transformer: true`; the default unconditional chair config does not enable conditioning.
