from dataclasses import dataclass
from typing import Any, Optional

from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()


@dataclass
class LightningTrainerConf:
    accelerator: Optional[str] = None
    accumulate_grad_batches: Any = 1
    amp_backend: str = 'native'
    amp_level: str = 'O2'
    auto_lr_find: Any = False
    auto_scale_batch_size: Any = False
    auto_select_gpus: bool = False
    benchmark: bool = False
    checkpoint_callback: bool = True
    check_val_every_n_epoch: int = 1
    default_root_dir: Optional[str] = None
    deterministic: bool = False
    fast_dev_run: bool = False
    flush_logs_every_n_steps: int = 100
    gpus: Optional[Any] = None
    gradient_clip_val: float = 0
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    limit_test_batches: float = 1.0
    log_gpu_memory: Optional[str] = None
    log_every_n_steps: int = 50
    automatic_optimization: bool = True
    prepare_data_per_node: bool = True
    process_position: int = 0
    progress_bar_refresh_rate: int = 1
    profiler: Any = None
    overfit_batches: float = 0.0
    precision: int = 32
    max_epochs: int = 1000
    min_epochs: int = 1
    max_steps: Optional[int] = None
    min_steps: Optional[int] = None
    num_nodes: int = 1
    num_sanity_val_steps: int = 2
    num_processes: int = 1
    reload_dataloaders_every_epoch: bool = False
    replace_sampler_ddp: bool = True
    resume_from_checkpoint: Optional[str] = None
    sync_batchnorm: bool = False
    terminate_on_nan: bool = False
    tpu_cores: Optional[Any] = None
    track_grad_norm: Any = -1
    truncated_bptt_steps: Optional[int] = None
    val_check_interval: float = 1.0
    weights_summary: Optional[str] = 'top'
    weights_save_path: Optional[str] = None


@dataclass
class LightningTrainerProjectConf(LightningTrainerConf):
    accelerator: Optional[str] = 'ddp'
    log_gpu_memory: Optional[str] = 'min_max'
    profiler: Any = 'simple'
    num_sanity_val_steps: int = 0
    max_epochs: int = 100000000
    max_steps: int = 100000000
    progress_bar_refresh_rate: int = 5

    # Since we have precomputed the FID stats
    limit_val_batches: float = 2
    limit_test_batches: float = 2


cs.store(group="trainer", name="default", node=LightningTrainerConf)
cs.store(group="trainer", name="project", node=LightningTrainerProjectConf)
