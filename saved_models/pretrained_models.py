# since I'll be using the pretrained models a lot in other files, I'll put them here to be imported
PRETRAINED_MODELS: dict[str, str] = {
    "starry_night": "saved_models/starry_night_pretrained.pth",
    "rain_princess": "saved_models/rain_princess_pretrained.pth",
    "abstract": "saved_models/abstract_pretrained.pth",
    "mosaic": "saved_models/mosaic_pretrained.pth",
}

# TODO: retrain the abstract model with slightly different image and hyperparameters
