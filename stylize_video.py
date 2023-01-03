import numpy as np
import cv2
import torch
from models import loss_models, transformation_models
from utils import preprocess_batch, deprocess_batch
from torchvision.transforms.functional import resize
from argument_parsers import stylize_video_parser

device = {torch.has_cuda: "cuda", torch.has_mps: "mps"}.get(True, "cpu")


def stylize_video(video_path, model_path, save_path, frames_per_step, image_size):
    # load the video
    video = cv2.VideoCapture(video_path)

    # get the video's dimensions and frame count
    width_original = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_original = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    frames_to_capture = frame_count

    print(f"source video dimensions: {width_original}x{height_original}")
    print(f"source video frame count: {frame_count}")
    print(f"source video fps: {fps}\n")

    # get the video output dimensions
    width = width_original
    height = height_original
    if image_size:
        min_dim = min(width_original, height_original)
        width = int(width_original / min_dim * image_size)
        height = int(height_original / min_dim * image_size)

    print(f"output video dimensions: {width}x{height}")
    print(f"output video frame count: {frames_to_capture}")
    print(f"output video fps: {fps}\n")

    # setting up the model
    transformation_model = transformation_models.TransformationModel().to(device).eval()
    # loading weights of pretrained model
    checkpoint = torch.load(model_path)
    transformation_model.load_state_dict(checkpoint["model_state_dict"])
    transformation_model.requires_grad_(False)

    # partition the frames into batches of size 64
    frames_batch_size = 64

    # save the frames as a video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        save_path,
        fourcc,
        float(fps),
        (width, height),
    )

    # use the first iteration to get the frame sizes
    for i in range(0, frames_to_capture, frames_batch_size):
        print(
            f"stylizing frames <{i}/{frames_to_capture}> to <{i + frames_batch_size}/{frames_to_capture}>"
        )
        # make sure the last batch has the correct size
        if i + frames_batch_size > frames_to_capture:
            frames_batch = np.empty(
                (frames_to_capture - i, height_original, width_original, 3),
                dtype=np.uint8,
            )
        else:
            frames_batch = np.empty(
                (frames_batch_size, height_original, width_original, 3), dtype=np.uint8
            )

        # read the frames
        frame_index = 0
        ret = True
        while video.isOpened() and frame_index < frames_batch_size:
            ret, frame = video.read()
            if ret:
                frames_batch[frame_index] = frame
                frame_index += 1
            else:
                # end of frames batch
                break

        stylized_batch = stylize_frames_batch(
            frames_batch, transformation_model, frames_per_step, image_size
        )
        for styled_frame in stylized_batch:
            out.write(styled_frame)

    out.release()
    # to add the audio back to the video, run this command in the terminal:
    # ffmpeg -i {save_path} -i {video_path} -c copy -map 0:v:0 -map 1:a:0 {save_with_audio_path}


def stylize_frames_batch(
    frames, transformation_model, frames_per_step, image_size=None
):
    """
    Stylize a batch of frames
    """
    # change the frames into torch tensors
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)

    # preprocess the frames to what the model expects
    mean = loss_models.VGG16Loss.MEAN
    std = loss_models.VGG16Loss.STD
    frames = preprocess_batch(frames, mean, std)
    if image_size:
        frames = resize(frames, image_size)
    width, height = frames.shape[3], frames.shape[2]
    mean = mean.to(device)
    std = std.to(device)

    frames_to_capture = frames.shape[0]
    # stylize the frames in batches
    stylized_frames = torch.empty_like(frames)
    for i in range(0, frames_to_capture, frames_per_step):
        # get the batch
        section = frames[i : i + frames_per_step].to(device)
        # stylize the batch
        stylized_section = transformation_model(section)

        # depreprocess the batch
        stylized_section = deprocess_batch(stylized_section, mean, std)

        # for some reason the transformed image ends up having slightly different dimensions
        # so we resize it to the right dimensions
        stylized_section = resize(
            stylized_section, (section.shape[2], section.shape[3])
        )
        # save the batch
        stylized_frames[i : i + frames_per_step] = stylized_section

        # print progress every 24 frames
        if i % 24 == 0:
            print(f"from batch, stylized frame [{i}/{frames_to_capture}]")

    print("styled frames successfully\n")

    # convert the frames back to numpy arrays
    stylized_frames = (
        stylized_frames.detach()
        .cpu()
        .permute(0, 2, 3, 1)
        .mul(255)
        .numpy()
        .astype("uint8")
    )
    # colors channel is in BGR, so we convert it to RGB
    stylized_frames = stylized_frames[:, :, :, ::-1]

    return stylized_frames


if __name__ == "__main__":
    args = stylize_video_parser()
    # stylize the video
    stylize_video(
        args.video_path,
        args.model_path,
        args.save_path,
        args.frames_per_step,
        args.max_image_size,
    )
