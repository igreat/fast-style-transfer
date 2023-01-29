import torch
from torchvision.transforms.functional import pil_to_tensor, resize
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
from models import loss_models, transformation_models

from argument_parsers import training_parser

class StyleModelTrainer:
    def __init__(self, model, loss_model, optimizer, training_config, device):
        self.transformation_model = model
        self.loss_model = loss_model
        self.optimizer = optimizer
        self.device = device
        self.training_config = training_config
        self.summary = SummaryWriter()

    def get_training_loader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(self.training_config["img_size"]),
                transforms.CenterCrop(self.training_config["img_size"]),
                transforms.ToTensor(),
                transforms.Normalize(self.loss_model.MEAN, self.loss_model.STD),
            ]
        )

        train_dataset = datasets.ImageFolder(
            root=self.training_config["path_to_dataset"], transform=transform
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.training_config["batch_size"], shuffle=True,
        )


        return train_loader

    def train(self):
        train_loader = self.get_training_loader()

        current_checkpoint = 1

        ## training ##
        # the size is rounded down to the nearest multiple of the batch size
        dataset_size = len(train_loader.dataset)
        dataset_size -= dataset_size % self.training_config["batch_size"]
        self.transformation_model.train()
        for epoch in range(self.training_config["epochs"]):
            for batch, (x, _) in enumerate(train_loader):

                # the batch size at the final batch is not always the same
                # as the batch size specified in the training config
                if len(x) != self.training_config["batch_size"]:
                    continue

                x = x.to(self.device)
                result = self.transformation_model(x)

                loss = self.loss_model(result, x)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # printing losses
                content_loss = self.loss_model.total_content_loss
                style_loss = self.loss_model.total_style_loss
                tv_loss = self.loss_model.tv_loss.loss
                current_iteration = batch * training_config["batch_size"]
                if current_iteration % 500 == 0:
                    current = batch * len(x)
                    print(f"style loss: {style_loss.item():>7f}", end="\t")
                    print(f"content loss: {content_loss.item():>7f}", end="\t")
                    print(f"tv loss: {tv_loss.item():>7f}", end="\t")
                    print(f"total loss: {loss.item():>7f}", end="\t")
                    print(f"[{current:>5d}/{dataset_size:>5d}]")

                # autosaving every 1000 training steps
                if current_iteration % 1000 == 0:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.transformation_model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "loss": loss,
                        },
                        "auto_save/auto_save.pth",
                    )

                # accumulative checkpointing
                if current_iteration % self.training_config["checkpoint_interval"] == 0:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.transformation_model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "loss": loss,
                        },
                        f"auto_save/checkpoint{current_checkpoint}.pth",
                    )
                    current_checkpoint += 1


                # adding losses to tensorboard
                self.summary.add_scalar(
                    "losses/content",
                    content_loss.item(),
                    current_iteration + epoch * dataset_size,
                )
                self.summary.add_scalar(
                    "losses/style",
                    style_loss.item(),
                    current_iteration + epoch * dataset_size,
                )
                self.summary.add_scalar(
                    "losses/tv",
                    tv_loss.item(),
                    current_iteration + epoch * dataset_size,
                )

                if current_iteration % 500 == 0:
                    # preparing and displaying the example image
                    example_image = x[0] * self.loss_model.STD + self.loss_model.STD
                    example_image = example_image.clamp(0, 1) * 255
                    example_image = (
                        example_image.detach().cpu().numpy().astype(np.uint8)
                    )
                    self.summary.add_image(
                        "images/example_image", 
                        example_image, 
                        current_iteration + epoch * dataset_size + 1,
                    )
                    # preparing and displaying the styled image
                    example_result = (
                        result[0] * self.loss_model.STD + self.loss_model.STD
                    )
                    example_result = example_result.clamp(0, 1) * 255
                    example_result = (
                        example_result.detach().cpu().numpy().astype(np.uint8)
                    )
                    self.summary.add_image(
                        "images/example_styled_image",
                        example_result,
                        current_iteration + epoch * dataset_size + 1,
                    )

        # NOTE: this save doesn't store optimizer information and is only used for inference
        #       if you want to continue training, use the checkpointing
        torch.save(
            self.transformation_model.state_dict(), "saved_models/trained_model.pth"
        )

if __name__ == "__main__":
    args = training_parser()

    # setting up the device
    print(f"Using {args.device} device")

    # setting up the training config
    training_config = {
        "path_to_dataset": args.train_dataset_path,
        "batch_size": args.batch_size,
        "img_size": args.image_size,
        "epochs": args.epochs,
        "checkpoint_interval": args.checkpoint_interval,
    }

    # setting up the model and optimizer
    transformation_model = transformation_models.TransformationModel().to(args.device)
    optimizer = torch.optim.Adam(transformation_model.parameters(), lr=args.learning_rate)

    # loading the model and optimizer
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path)
        # make sure the model has an optimizer state dict
        assert "optimizer_state_dict" in checkpoint, "checkpoint doesn't have optimizer state dict"
        transformation_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # setting up the loss model
    style_img = (
        pil_to_tensor((Image.open(args.style_image_path)).convert("RGB"))
        .unsqueeze(0)
        .float()
        .div(255)
    )
    if args.style_size:
        style_img = resize(style_img, args.style_size)
    mean, std = loss_models.VGG16Loss.MEAN, loss_models.VGG16Loss.STD
    style_img = (style_img - mean) / std

    # for some reason using process_image() here makes the style loss very large (big bug)

    loss_model = loss_models.VGG16Loss(
        style_img=style_img, 
        content_weight=args.content_weight, 
        style_weight=args.style_weight, 
        tv_weight=args.tv_weight, 
        batch_size=args.batch_size,
        device=args.device,
    )

    # training the model
    trainer = StyleModelTrainer(
        transformation_model, loss_model, optimizer, training_config, args.device
    )
    trainer.train()
    print("Training complete!")
    