# 3D-Diffusion

Refactored course-project implementation of an SDFusion-style 3D generative pipeline. The code in this folder is the submitted implementation; `../SDFusion-master` is treated as a read-only reference and is not modified.

## Implemented

- ShapeNet SDF dataset loading for single-category training.
- Legacy-compatible VQ-VAE architecture and checkpoint key layout.
- VQ-VAE training, evaluation, reconstruction export, mesh export, and optional SDF geometry losses.
- Quantized latent diffusion training: `SDF -> encoder -> quantizer -> z_q -> diffusion`.
- SDFusion/LDM OpenAI-style 3D UNet migration with `diffusion_net.*` legacy checkpoint key compatibility.
- DDPM loss and DDPM/DDIM/PLMS samplers with clipping, mask/x0 conditioning, callbacks, intermediates, progress display, and classifier-free guidance hooks.
- Lightweight `concat`, `crossattn`, and `hybrid` conditioning interfaces.
- Dataset inspection, latent statistics, unconditional inference, and generation evaluation tools.

## Relationship To SDFusion

The VQ-VAE path is designed to match the original SDFusion VQ-VAE state dict names:

- `encoder.*`
- `decoder.*`
- `quantize.embedding.weight`
- `quant_conv.*`
- `post_quant_conv.*`

Diffusion now uses a migrated OpenAI-style 3D UNet by default through `unet_architecture: legacy_openai`. The nested `diffusion_net` module preserves the original SDFusion checkpoint layout, so legacy diffusion checkpoints with keys such as `diffusion_net.input_blocks.*` can be loaded through the compatibility loader. A smaller `compact` UNet remains available only as a lightweight fallback.

## Environment

Python 3.10+ is recommended.

```bash
pip install -e .
```

Core dependencies are listed in `pyproject.toml`: `torch`, `numpy`, `h5py`, `PyYAML`, and `scikit-image`.

## Data Format

Default ShapeNet SDF path:

```text
data/
  ShapeNet/
    SDF_v1/
      resolution_64/
        03001627/
          <model_id>/
            ori_sample_grid.h5
```

For chairs, the synset is `03001627`. Each SDF should reshape to `[1, 64, 64, 64]`. The loader also accepts `.npy` and `.npz` SDF files when referenced by a filelist.

Inspect before training:

```bash
python tools/inspect_dataset.py \
  --data_root data \
  --category chair \
  --res 64 \
  --split train \
  --max_samples 2
```

## VQ-VAE Training

```bash
bash scripts/train_vqvae_chair.sh
```

Equivalent explicit command:

```bash
python tools/train_vqvae.py \
  --config config/defaults/vqvae_snet_chair.yaml \
  --out_dir outputs/vqvae_chair \
  --override data.data_root=data \
  --override data.category=chair
```

Outputs:

- `outputs/vqvae_chair/resolved_config.yaml`
- `outputs/vqvae_chair/metrics.jsonl`
- `outputs/vqvae_chair/checkpoints/vqvae_last.pt`
- `outputs/vqvae_chair/reconstructions/step_*/`

The default VQ-VAE objective matches the original lightweight SDFusion training target:

```text
L = L1(reconstruction, sdf) + codebook_weight * codebook_loss
```

Additional SDF losses can be enabled from config or command line:

```bash
python tools/train_vqvae.py \
  --config config/defaults/vqvae_snet_chair.yaml \
  --out_dir outputs/vqvae_chair_geo \
  --override vqvae.occupancy_weight=0.1 \
  --override vqvae.surface_weight=1.0 \
  --override vqvae.normal_weight=0.1 \
  --override vqvae.multiscale_weight=0.25 \
  --override vqvae.multiscale_levels=3
```

These optional terms are occupancy BCE on SDF sign, near-surface weighted L1, near-surface normal alignment, and multiscale L1.

## Latent Scale

After VQ-VAE training, compute latent statistics:

```bash
python tools/compute_latent_stats.py \
  --config config/defaults/vqvae_snet_chair.yaml \
  --vqvae_ckpt outputs/vqvae_chair/checkpoints/vqvae_last.pt \
  --out_dir outputs/vqvae_chair
```

This writes:

- `outputs/vqvae_chair/latent_stats.pt`
- `outputs/vqvae_chair/latent_stats.json`

Use `scale_factor = 1 / std(z_q)`. The diffusion chair script automatically uses this value when `latent_stats.json` exists.

## Diffusion Training

```bash
bash scripts/train_diffusion_chair.sh
```

Equivalent explicit command:

```bash
python tools/train_diffusion.py \
  --config config/defaults/diffusion_snet_chair.yaml \
  --vqvae_ckpt outputs/vqvae_chair/checkpoints/vqvae_last.pt \
  --out_dir outputs/diffusion_chair
```

`--vqvae_ckpt` is required. Training without a trained VQ-VAE is intentionally blocked.

Outputs:

- `outputs/diffusion_chair/resolved_config.yaml`
- `outputs/diffusion_chair/metrics.jsonl`
- `outputs/diffusion_chair/checkpoints/diffusion_last.pt`
- `outputs/diffusion_chair/samples/step_*/sample_*.sdf.npy`
- `outputs/diffusion_chair/samples/step_*/sample_*.ply`
- `outputs/diffusion_chair/samples/step_*/snapshot_evaluation.json`

During training, `train_diffusion.py` logs training loss, validation diffusion loss, and sample snapshot metrics. The snapshot report includes mesh extraction success rate, SDF statistics, occupancy ratio, and an L1 diversity proxy. Configure cadence with `train.eval_every`, `train.sample_every`, `train.sample_num`, `train.sample_steps`, and `train.sample_sampler`.

## Unconditional Inference

```bash
bash scripts/infer_uncond_chair.sh
```

Equivalent explicit command:

```bash
python tools/infer_uncond.py \
  --config config/defaults/diffusion_snet_chair.yaml \
  --vqvae_ckpt outputs/vqvae_chair/checkpoints/vqvae_last.pt \
  --diffusion_ckpt outputs/diffusion_chair/checkpoints/diffusion_last.pt \
  --num_samples 4 \
  --ddim_steps 100 \
  --out_dir outputs/samples_chair
```

Outputs:

- `sample_0000.sdf.npy`
- `sample_0000.ply`
- `sample_0000.metadata.json`
- `inference_summary.json`

## Generation Evaluation

```bash
bash scripts/eval_chair.sh
```

Equivalent explicit command:

```bash
python tools/evaluate_generation.py \
  --sample_dir outputs/samples_chair
```

The report is written to:

```text
outputs/samples_chair/evaluation.json
```

It includes mesh extraction success rate, failed sample count, SDF min/max/mean, occupancy ratio, and an L1 diversity proxy.

Sampler options:

```bash
python tools/infer_uncond.py \
  --config config/defaults/diffusion_snet_chair.yaml \
  --vqvae_ckpt outputs/vqvae_chair/checkpoints/vqvae_last.pt \
  --diffusion_ckpt outputs/diffusion_chair/checkpoints/diffusion_last.pt \
  --sampler plms \
  --ddim_discretize uniform \
  --num_samples 4 \
  --ddim_steps 100 \
  --out_dir outputs/samples_chair
```

`--sampler` can be `ddpm`, `ddim`, or `plms`. PLMS is deterministic and intentionally rejects `eta != 0`.

## Legacy Checkpoint Conversion

```bash
python tools/convert_legacy_checkpoint.py \
  --component vqvae \
  --input path/to/legacy_vqvae.pth \
  --output outputs/converted_vqvae.pt
```

For VQ-VAE, converted keys are intended to load into `architecture: legacy`. For diffusion, converted `df` checkpoints preserve `diffusion_net.*` keys and load into the default `legacy_openai` denoiser.

## One-Command Scripts

All scripts avoid personal absolute paths and use environment variables:

```bash
DATA_ROOT=${DATA_ROOT:-data}
OUT_DIR=${OUT_DIR:-outputs}
CATEGORY=${CATEGORY:-chair}
```

Available scripts:

- `bash scripts/smoke_test_imports.sh`
- `bash scripts/train_vqvae_chair.sh`
- `bash scripts/train_diffusion_chair.sh`
- `bash scripts/infer_uncond_chair.sh`
- `bash scripts/eval_chair.sh`

## Reserved Interfaces

The text, image, partial-shape, and multimodal condition paths are lightweight functional interfaces. They pass conditions into the diffusion model, but they are not full migrations of SDFusion's BERT/CLIP/ResNet/partial-shape conditioning stack.

Multi-class ShapeNet loading is still reserved; current scripts target one category at a time.

## Course Submission Notes

For reproducibility, run:

```bash
bash scripts/smoke_test_imports.sh
python tools/inspect_dataset.py --data_root data --category chair --res 64 --split train --max_samples 2
bash scripts/train_vqvae_chair.sh
python tools/compute_latent_stats.py --config config/defaults/vqvae_snet_chair.yaml --vqvae_ckpt outputs/vqvae_chair/checkpoints/vqvae_last.pt --out_dir outputs/vqvae_chair
bash scripts/train_diffusion_chair.sh
bash scripts/infer_uncond_chair.sh
bash scripts/eval_chair.sh
```

Include `metrics.jsonl`, reconstruction examples, generated `.ply` files, and `evaluation.json` in the experiment report.

## References

- Original SDFusion repository used as a read-only reference.
- Latent Diffusion Models for the DDPM/DDIM training and sampling formulation.
- VQ-VAE for vector quantized autoencoding.
