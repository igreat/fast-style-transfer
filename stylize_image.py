import torch
from models import loss_models, transformation_models
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms.functional import pil_to_tensor, resize
from utils import preprocess_image, deprocess_image
from argument_parsers import stylize_image_parser

device = {torch.has_cuda: "cuda", torch.has_mps: "mps"}.get(True, "cpu")


def stylize_image(image_path, image_size, model_path, save_path):
    mean = loss_models.VGG16Loss.MEAN.to(device)
    std = loss_models.VGG16Loss.STD.to(device)
    img = pil_to_tensor((Image.open(image_path)).convert("RGB")).to(device)
    if image_size:
        img = resize(img, size=image_size)
    img = preprocess_image(img, mean, std)

    transformation_model = transformation_models.TransformationModel().to(device).eval()

    # code to load pretrained model
    checkpoint = torch.load(model_path)
    transformation_model.load_state_dict(checkpoint["model_state_dict"])

    transformation_model.requires_grad_(False)

    gen_image = transformation_model(img)
    gen_image = deprocess_image(gen_image, mean, std)

    # saving image
    save_image(gen_image.squeeze(0), save_path)

    print(f"image saved successfully at {save_path}")


if __name__ == "__main__":
    args = stylize_image_parser()
    # stylize the image
    stylize_image(args.image_path, args.image_size, args.model_path, args.save_path)
