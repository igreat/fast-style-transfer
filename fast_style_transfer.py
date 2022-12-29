import torch
from models import loss_models, transformation_models
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms.functional import pil_to_tensor, resize
from utils import display_images_in_a_grid

device = {torch.has_cuda: "cuda", torch.has_mps: "mps"}.get(True, "cpu")


def apply_style(path_to_image, path_to_model):

    img = (
        pil_to_tensor((Image.open(path_to_image)).convert("RGB"))
        .unsqueeze(0)
        .float()
        .div(255)
    )

    transformation_model = transformation_models.TransformationModel()

    # code to load pretrained model
    checkpoint = torch.load(path_to_model)
    transformation_model.load_state_dict(checkpoint["model_state_dict"])

    mean, std = loss_models.VGG16Loss.MEAN, loss_models.VGG16Loss.STD
    gen_image = transformation_model.eval()(img)
    gen_image = gen_image * std + mean
    gen_image = gen_image.clamp(0, 1)

    # saving image
    save_image(gen_image.squeeze(0), "styled_image.png")


def main():
    apply_style(
        "images/lion.jpg",
        "saved-models/starry_night_pretrained.pth",
    )


if __name__ == "__main__":
    main()
