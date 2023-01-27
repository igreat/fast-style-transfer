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
    "spaceship_orion": dir_path / "spaceship_orion.pth",
    "cyber_fountain": dir_path / "cyber_fountain.pth",
    "escher1_grayscale": dir_path / "escher1_grayscale.pth",
    "cyberpunk_cranes": dir_path / "cyberpunk_cranes.pth",
    "shibuya_post_apoca": dir_path / "shibuya_post_apoca.pth",
    "168_british_library": dir_path / "168_british_library.pth",
    "117_british_library": dir_path / "117_british_library.pth",
    "193_british_library": dir_path / "193_british_library.pth",
    "063_135_140_177_composite_british_library.pth": dir_path / "063_135_140_177_composite_british_library.pth",
}

from pprint import pprint
print("models available: \n")
pprint(PRETRAINED_MODELS)

# TODO: retrain the abstract model with slightly different image and hyperparameters
