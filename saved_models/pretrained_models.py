import os
from typing import Dict
from pathlib import Path
dir_path = Path(os.path.dirname(os.path.realpath(__file__)))

# since I'll be using the pretrained models a lot in other files, I'll put them here to be imported
PRETRAINED_MODELS: Dict[str, str] = {
    "starry_night": dir_path / "starry_night_pretrained.pth",
    "rain_princess": dir_path / "rain_princess_pretrained.pth",
    "abstract": dir_path / "abstract_pretrained.pth",
    "mosaic": dir_path / "mosaic_pretrained.pth",
}

# TODO: retrain the abstract model with slightly different image and hyperparameters
