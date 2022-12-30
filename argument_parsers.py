# this is where all argument parsing should be done
import argparse
from saved_models.pretrained_models import PRETRAINED_MODELS


def stylize_image_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stylize an image")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="path to the image to be stylized",
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
    args = parser.parse_args()

    if not args.pretrained_model and not args.model_path:
        raise ValueError(
            "Either a pretrained model or a path to a model must be specified"
        )
    if not args.model_path:
        args.model_path = PRETRAINED_MODELS[args.pretrained_model]

    return args
