from typing import Dict
from pathlib import Path

# since I'll be using the pretrained models a lot in other files, I'll put them here to be imported
PRETRAINED_MODELS: Dict[str, str] = {
    "starry_night": Path(__file__) / "starry_night_pretrained.pth",
    "rain_princess": Path(__file__) / "rain_princess_pretrained.pth",
    "abstract": Path(__file__) / "abstract_pretrained.pth",
    "mosaic": Path(__file__) / "mosaic_pretrained.pth",
}

# TODO: retrain the abstract model with slightly different image and hyperparameters
