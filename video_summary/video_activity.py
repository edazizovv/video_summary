
import os
import torch
from transformers import XCLIPProcessor, XCLIPModel

from pytube import YouTube
import numpy as np
from decord import VideoReader, cpu, AVReader
import imageio

import subprocess

import os


def download_youtube_video(url: str):
    yt = YouTube(url)

    streams = yt.streams.filter(file_extension='mp4')
    file_path = streams[0].download()
    return file_path


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def my_sample_frame_indices(start_frame, end_frame, num_patches=32):
    indices = np.linspace(start_frame, end_frame, num=num_patches)
    indices = np.clip(indices, start_frame, end_frame - 1).astype(np.int64)
    return indices


def sample_frames_from_video_file(file_path: str, num_frames: int = 16):
    # videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))
    videoreader = AVReader(file_path, num_threads=1, ctx=cpu(0))

    # sample frames
    videoreader.seek(0)
    indices = sample_frame_indices(clip_len=num_frames, frame_sample_rate=4, seg_len=len(videoreader))
    # frames = videoreader.get_batch(indices).asnumpy()

    audio, frames = videoreader.get_batch(indices).asnumpy()
    return audio, frames


def convert_frames_to_gif(frames, second_start, second_end, fps=10):
    converted_frames = frames.astype(np.uint8)
    name = "cut_{0}_{1}.gif".format(second_start, second_end)
    imageio.mimsave(name, converted_frames, fps=fps)
    return name


def my_sample(file_path, start_second, end_second):
    videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))
    fps = np.round(videoreader.get_avg_fps())
    start_frame = start_second * fps
    end_frame = end_second * fps
    indices = my_sample_frame_indices(start_frame=start_frame, end_frame=end_frame)
    frames = videoreader.get_batch(indices).asnumpy()
    return frames


def my_sample_full(file_path, start_second, end_second):
    videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))
    fps = np.round(videoreader.get_avg_fps())
    frames = videoreader[start_second * fps:end_second * fps].asnumpy()
    return frames



def predict(video_path, labels, seconds_start, seconds_end):

    model_name = "microsoft/xclip-base-patch16-zero-shot"
    processor = XCLIPProcessor.from_pretrained(model_name)
    model = XCLIPModel.from_pretrained(model_name)

    resulted = {'second_start': [],
                'second_end': [],
                'gif_path': [],
                'label': [],
                'label_prob': []}
    for j in range(len(seconds_start)):
        second_start = seconds_start[j]
        second_end = seconds_end[j]

        resulted['second_start'].append(second_start)
        resulted['second_end'].append(second_end)

        frames = my_sample(video_path, start_second=second_start, end_second=second_end)
        # os.remove(video_path)
        gif_path = convert_frames_to_gif(frames, second_start=second_start, second_end=second_end)
        resulted['gif_path'].append(gif_path)

        inputs = processor(
            text=labels,
            videos=list(frames),
            return_tensors="pt",
            padding=True
        )
        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        probs = outputs.logits_per_video[0].softmax(dim=-1).cpu().numpy()
        lbs = []
        pbs = []
        for ind, label in enumerate(labels):
            lbs.append(label)
            pbs.append(float(probs[ind]))

        mx_label = lbs[np.argmax(pbs)]
        mx_prob = np.max(pbs)

        resulted['label'] = mx_label
        resulted['label_prob'] = mx_prob

    return resulted

def convert_video_to_wav(d):

    v = "{0}.wav".format(os.path.splitext(d)[0])
    os.remove(v)

    ffmpeg_command = [
        "ffmpeg",
        "-i",
        d,
        "-ab",
        "160k",
        "-ac",
        "2",
        "-ar",
        "44100",
        "-vn",
        v,
    ]

    try:
        subprocess.call(ffmpeg_command, shell=True)
    except FileNotFoundError as error:
        raise ValueError("ffmpeg was not found but is required to load audio files from filename") from error

    return v


from speech_activity import segmentation

y = 'https://www.youtube.com/watch?v=Y3LufB6DK4k'

d = download_youtube_video(url=y)
v = convert_video_to_wav(d=d)

result = segmentation(audio=v)


"""
labels = ['speaking', 'listening']
seconds_start = [0]
seconds_end = [10]
result = predict(video_path=d, labels=labels, seconds_start=seconds_start, seconds_end=seconds_end)
"""