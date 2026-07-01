from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

TaskName = Literal["uncond", "completion", "txt2shape", "img2shape", "mm2shape"]
ConditioningKey = Optional[Literal["concat", "crossattn", "hybrid"]]
ConditionerName = Optional[Literal["text", "image", "partial", "multimodal"]]


@dataclass
class DataConfig:
    dataset: str = "shapenet_sdf"
    data_root: str = "data"
    category: str = "chair"
    split: str = "train"
    res: int = 64
    max_samples: Optional[int] = None
    filelist: Optional[str] = None
    split_file_root: Optional[str] = None
    trunc_thres: float = 0.2


@dataclass
class VQVAEConfig:
    architecture: Literal["simple", "legacy"] = "legacy"
    in_channels: int = 1
    out_channels: int = 1
    resolution: int = 64
    base_channels: int = 64
    channel_multipliers: list[int] = field(default_factory=lambda: [1, 2, 4])
    z_channels: int = 3
    embed_dim: int = 3
    n_embed: int = 8192
    ddconfig: dict[str, Any] = field(default_factory=dict)
    legacy_quantizer_loss: bool = False
    init_type: str = "normal"
    init_gain: float = 0.02


@dataclass
class VQVAELossConfig:
    codebook_weight: float = 1.0
    occupancy_weight: float = 0.05
    surface_weight: float = 0.1
    normal_weight: float = 0.05
    multiscale_weight: float = 0.1
    surface_band: float = 0.02
    occupancy_temperature: float = 0.02
    multiscale_levels: int = 3


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_schedule: str = "linear"
    linear_start: float = 1.0e-4
    linear_end: float = 2.0e-2
    scale_factor: float = 1.0
    latent_channels: int = 3
    latent_size: int = 16
    unet_base_channels: int = 192
    unet_architecture: Literal["legacy_openai", "compact"] = "legacy_openai"
    unet_params: dict[str, Any] = field(default_factory=dict)
    concat_channels: int = 0
    context_dim: int = 0
    guidance_scale: float = 1.0
    ddim_eta: float = 0.0


@dataclass
class TrainConfig:
    seed: int = 0
    batch_size: int = 4
    num_workers: int = 0
    lr: float = 1.0e-4
    max_steps: int = 10000
    log_every: int = 50
    save_every: int = 1000
    eval_every: int = 1000
    eval_batches: int = 8
    sample_every: int = 1000
    sample_num: int = 4
    sample_steps: int = 100
    sample_sampler: Literal["ddim", "ddpm", "plms"] = "ddim"
    grad_clip_norm: float = 1.0
    device: str = "cuda"


@dataclass
class SDFusionConfig:
    project: str = "SDFusion-Refactored"
    task: TaskName = "uncond"
    implemented: bool = True
    status: str = "stage1_unconditional"
    conditioning_key: ConditioningKey = None
    conditioner: ConditionerName = None
    data: DataConfig = field(default_factory=DataConfig)
    vqvae: VQVAEConfig = field(default_factory=VQVAEConfig)
    vqvae_loss: VQVAELossConfig = field(default_factory=VQVAELossConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    extra: dict[str, Any] = field(default_factory=dict)
