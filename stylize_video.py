import numpy as np
import cv2
import torch
from models import loss_models, transformation_models
from utils import preprocess_batch, deprocess_batch
from torchvision.transforms.functional import resize
from argument_parsers import stylize_video_parser

device = {torch.has_cuda: "cuda", torch.has_mps: "mps"}.get(True, "cpu")


def stylize_video(video_path, model_path, save_path):
    # load the video
    video = cv2.VideoCapture(video_path)

    # get the video's dimensions and frame count
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    frames_to_capture = frame_count

    print(f"video dimensions: {width}x{height}")
    print(f"video frame count: {frame_count}")
    print(f"video fps: {fps}")

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
    frames = preprocess_batch(frames, loss_models.VGG16Loss)

    # setting up the model
    transformation_model = transformation_models.TransformationModel().to(device)
    # loading weights of pretrained model
    checkpoint = torch.load(model_path)
    transformation_model.load_state_dict(checkpoint["model_state_dict"])

    transformation_model.requires_grad_(False)

    # stylize the frames in batches of 8
    stylized_frames = torch.empty_like(frames)
    batch_size = 8
    for i in range(0, frames_to_capture, batch_size):
        # get the batch
        batch = frames[i : i + batch_size].to(device)
        # stylize the batch
        stylized_batch = transformation_model(batch)

        # depreprocess the batch
        stylized_batch = deprocess_batch(stylized_batch, loss_models.VGG16Loss, device)

        # for some reason the transformed image ends up having slightly different dimensions
        # so we resize it to the right dimensions
        stylized_batch = resize(stylized_batch, (batch.shape[2], batch.shape[3]))
        # save the batch
        stylized_frames[i : i + batch_size] = stylized_batch

        # print progress every 24 frames
        if i % 24 == 0:
            print(f"stylized frame [{i}/{frames_to_capture}]")

    print("styled frames successfully")

    # convert the frames back to numpy arrays
    stylized_frames = (
        stylized_frames.detach().permute(0, 2, 3, 1).mul(255).numpy().astype("uint8")
    )

    # save the frames as a video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        save_path,
        fourcc,
        float(fps),
        (stylized_frames.shape[2], stylized_frames.shape[1]),
    )
    print(stylized_frames.shape)
    print("saving video...")
    for styled_frame in stylized_frames:
        out.write(styled_frame)

    out.release()

    print(f"styled video saved successfully at {save_path}")

    # to add the audio back to the video, run this command in the terminal:
    # ffmpeg -i {save_path} -i {video_path} -c copy -map 0:v:0 -map 1:a:0 {save_with_audio_path}


if __name__ == "__main__":
    args = stylize_video_parser()
    # stylize the video
    stylize_video(args.video_path, args.model_path, args.save_path)
