import numpy as np
import cv2
import torch
from models import loss_models, transformation_models
from utils import preprocess_batch, deprocess_batch
from torchvision.transforms.functional import resize
from argument_parsers import stylize_video_parser

device = {torch.has_cuda: "cuda", torch.has_mps: "mps"}.get(True, "cpu")


def stylize_video(video_path, model_path, save_path, batch_size, image_size):
    # load the video
    video = cv2.VideoCapture(video_path)

    # get the video's dimensions and frame count
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    frames_to_capture = frame_count

    print(f"source video dimensions: {width}x{height}")
    print(f"source video frame count: {frame_count}")
    print(f"source video fps: {fps}\n")

    # create a numpy array to store the frames
    frames = np.empty((frames_to_capture, height, width, 3), np.dtype("uint8"))

    # read the frames
    frame_index = 0
    ret = True
    while video.isOpened() and frame_index < frames_to_capture:
        ret, frame = video.read()
        if ret:
            frames[frame_index] = frame
            frame_index += 1
        else:
            # end of video
            break
    video.release()

    # convert the frames to torch tensors
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

    print(f"output video dimensions: {width}x{height}")
    print(f"output video frame count: {frames_to_capture}")
    print(f"output video fps: {fps}\n")

    # setting up the model
    transformation_model = transformation_models.TransformationModel().to(device).eval()
    # loading weights of pretrained model
    checkpoint = torch.load(model_path)
    transformation_model.load_state_dict(checkpoint["model_state_dict"])

    transformation_model.requires_grad_(False)

    # stylize the frames in batches
    stylized_frames = torch.empty_like(frames)
    for i in range(0, frames_to_capture, batch_size):
        # get the batch
        batch = frames[i : i + batch_size].to(device)
        # stylize the batch
        stylized_batch = transformation_model(batch)

        # depreprocess the batch
        stylized_batch = deprocess_batch(stylized_batch, mean, std)

        # for some reason the transformed image ends up having slightly different dimensions
        # so we resize it to the right dimensions
        stylized_batch = resize(stylized_batch, (batch.shape[2], batch.shape[3]))
        # save the batch
        stylized_frames[i : i + batch_size] = stylized_batch

        # print progress every 24 frames
        if i % 24 == 0:
            print(f"stylized frame [{i}/{frames_to_capture}]")

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

    # save the frames as a video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        save_path,
        fourcc,
        float(fps),
        (stylized_frames.shape[2], stylized_frames.shape[1]),
    )
    print("saving video...\n")
    for styled_frame in stylized_frames:
        out.write(styled_frame)

    out.release()

    print(f"styled video saved successfully at {save_path}")

    # to add the audio back to the video, run this command in the terminal:
    # ffmpeg -i {save_path} -i {video_path} -c copy -map 0:v:0 -map 1:a:0 {save_with_audio_path}


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
