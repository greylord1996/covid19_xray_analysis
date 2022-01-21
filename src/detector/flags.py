# --- flags ---
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Flags:
    # General
    debug: bool = False

    fold: int = 0
    outdir: str = f"results/fold_{fold}"


    # Data config
    root_imgdir: str = "/home/hdd/storage/siim_covid_detection/resized_images_from_kaggle/"
    path_to_meta_df: str = "../cross_validation_scheme/detectron_prepared_data/meta_df.csv"
    path_to_train_df: str = "../cross_validation_scheme/detectron_prepared_data/detectron_prepared_df.csv"
    use_cache: bool = False
    img_size: str = "1024x1024"
    cv_scheme: str = "skf"
    use_negative: bool = True
    binary_task: bool = True
    iter: int = 10000
    roi_batch_size_per_image: int = 256
    eval_period: int = 1000
    lr_scheduler_name: str = "WarmupMultiStepLR"
    base_lr: float = 0.00025
    ims_per_batch: int = 4  # images per batch, this corresponds to "total batch size"
    num_workers: int = 4
    aug_kwargs: Dict = field(default_factory=lambda: {})

    def update(self, param_dict: Dict) -> "Flags":
        # Overwrite by `param_dict`
        for key, value in param_dict.items():
            if not hasattr(self, key):
                raise ValueError(f"[ERROR] Unexpected key for flag = {key}")
            setattr(self, key, value)
        return self