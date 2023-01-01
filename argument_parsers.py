# this is where all argument parsing should be done
import argparse
from saved_models.pretrained_models import PRETRAINED_MODELS
import torch


def stylize_image_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stylize an image")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="path to the image to be stylized",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="size of the image to be stylized. if not specified, the image will not be resized",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        choices=PRETRAINED_MODELS.keys(),
        help="pretrained model to be used for stylizing the image",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="path to the model to be used for stylizing the image",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="images/generated_images/stylized_image.png",
        help="path to save the stylized image",
    )
    args = parser.parse_args()

    if not args.pretrained_model and not args.model_path:
        raise ValueError(
            "Either a pretrained model or a path to a model must be specified"
        )
    if not args.model_path:
        args.model_path = PRETRAINED_MODELS[args.pretrained_model]

    return args


def stylize_video_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stylize a video")
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="path to the video to be stylized",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        choices=PRETRAINED_MODELS.keys(),
        help="pretrained model to be used for stylizing the video",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="path to the model to be used for stylizing the video",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="videos/generated_videos/stylized_video.mp4",
        help="path to save the stylized video",
    )
    parser.add_argument(
        "--frames_per_step",
        type=int,
        default=1,
        help="number of frames to transform at a time. higher values will be faster but will result in signficantly more memory usage",
    )
    parser.add_argument(
        "--max_image_size",
        type=int,
        default=None,
        help="maximum size of dimensions of the video frames. if not specified, the frames will not be resized",
    )
    args = parser.parse_args()

    if not args.pretrained_model and not args.model_path:
        raise ValueError(
            "Either a pretrained model or a path to a model must be specified"
        )
    if not args.model_path:
        args.model_path = PRETRAINED_MODELS[args.pretrained_model]

    return args


def training_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train a model")
    parser.add_argument(
        "--style_image_path",
        type=str,
        required=True,
        help="path to the style image",
    )
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default="data/mscoco-new",
        help="path to the training dataset",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="saved_models/trained_model.pth",
        help="path to save the trained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="number of epochs to train the model for",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="batch size to train the model with",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="image size to train the model with",
    )
    parser.add_argument(
        "--style_size",
        type=int,
        default=None,
        help="style size to train the model with. if not specified, the orignal size will be used",
    )
    parser.add_argument(
        "--style_weight",
        type=float,
        default=5e7,
        help="weight of the style loss",
    )
    parser.add_argument(
        "--content_weight",
        type=float,
        default=1e2,
        help="weight of the content loss",
    )
    parser.add_argument(
        "--tv_weight",
        type=float,
        default=0,
        help="weight of the total variation loss",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="learning rate to train the model with",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="path to the checkpoint to resume training from. If not specified, training will start from scratch",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=2000,
        help="number of images to train on before saving a checkpoint. keep it a multiple of the batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        help="device to train the model on",
    )
    args = parser.parse_args()

    if not args.device:
        args.device = {torch.has_cuda: "cuda", torch.has_mps: "mps"}.get(True, "cpu")

    return args
