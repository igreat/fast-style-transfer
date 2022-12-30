import torch
from models import loss_models, transformation_models
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms.functional import pil_to_tensor
from saved_models.pretrained_models import PRETRAINED_MODELS

device = {torch.has_cuda: "cuda", torch.has_mps: "mps"}.get(True, "cpu")


def stylize_image(path_to_image, path_to_model, path_to_save="images/styled_image.png"):
    img = (
        pil_to_tensor((Image.open(path_to_image)).convert("RGB"))
        .unsqueeze(0)
        .float()
        .to(device)
        .div(255)
    )

    transformation_model = transformation_models.TransformationModel().to(device)

    # code to load pretrained model
    checkpoint = torch.load(path_to_model)
    transformation_model.load_state_dict(checkpoint["model_state_dict"])

    mean = loss_models.VGG16Loss.MEAN.to(device)
    std = loss_models.VGG16Loss.STD.to(device)

    gen_image = transformation_model.eval()(img)
    gen_image = gen_image * std + mean
    gen_image = gen_image.clamp(0, 1)

    # saving image
    save_image(gen_image.squeeze(0), path_to_save)

    print(f"image saved successfully at {path_to_save}")


if __name__ == "__main__":

    image_config = {
        "path_to_image": "images/content_images/houses.jpg",
        "path_to_model": PRETRAINED_MODELS["rain_princess"],
    }

    stylize_image(
        image_config["path_to_image"],
        image_config["path_to_model"],
    )
