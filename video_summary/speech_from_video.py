# https://huggingface.co/spaces/AIZeroToHero/Video-Automatic-Speech-Recognition/blob/main/app.py
# https://huggingface.co/models?pipeline_tag=automatic-speech-recognition

# !! # https://huggingface.co/pyannote/speaker-diarization

# https://huggingface.co/spaces/Cahlil/Speech-Recognition-with-Speaker-Segmentation
# X
# https://huggingface.co/spaces/fcakyon/zero-shot-video-classification

import subprocess

import numpy as np

from collections import deque
import streamlit as st
import torch
from transformers import AutoModelForCTC, Wav2Vec2Processor


model_path = "facebook/wav2vec2-large-robust-ft-swbd-300h"

processor = Wav2Vec2Processor.from_pretrained(model_path)
model = AutoModelForCTC.from_pretrained(model_path)

youtube_url = ...

chunk_duration_ms = 3000  # 2000 : 10000
pad_duration_ms = 1000    # 100  :  5000


def ffmpeg_stream(youtube_url, sampling_rate=16_000, chunk_duration_ms=5000, pad_duration_ms=200):
    """
    Helper function to read an audio file through ffmpeg.
    """
    chunk_len = int(sampling_rate * chunk_duration_ms / 1000)
    pad_len = int(sampling_rate * pad_duration_ms / 1000)
    read_chunk_len = chunk_len + pad_len * 2

    ar = f"{sampling_rate}"
    ac = "1"
    format_for_conversion = "f32le"
    dtype = np.float32
    size_of_sample = 4

    ffmpeg_command = [
        "ffmpeg",
        "-i",
        "pipe:",
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]

    ytdl_command = ["yt-dlp", "-f", "bestaudio", youtube_url, "--quiet", "-o", "-"]

    try:
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=-1)
        ytdl_process = subprocess.Popen(ytdl_command, stdout=ffmpeg_process.stdin)
    except FileNotFoundError:
        raise ValueError("ffmpeg was not found but is required to stream audio files from filename")

    acc = b""
    leftover = np.zeros((0,), dtype=np.float32)
    while ytdl_process.poll() is None:
        buflen = read_chunk_len * size_of_sample

        raw = ffmpeg_process.stdout.read(buflen)
        if raw == b"":
            break

        if len(acc) + len(raw) > buflen:
            acc = raw
        else:
            acc += raw

        audio = np.frombuffer(acc, dtype=dtype)
        audio = np.concatenate([leftover, audio])
        if len(audio) < pad_len * 2:
            # TODO: handle end of stream better than this
            break
        yield audio

        leftover = audio[-pad_len * 2:]
        read_chunk_len = chunk_len


def stream_text(url, chunk_duration_ms, pad_duration_ms):
    sampling_rate = processor.feature_extractor.sampling_rate

    # calculate the length of logits to cut from the sides of the output to account for input padding
    output_pad_len = model._get_feat_extract_output_lengths(int(sampling_rate * pad_duration_ms / 1000))

    # define the audio chunk generator
    stream = ffmpeg_stream(url, sampling_rate, chunk_duration_ms=chunk_duration_ms, pad_duration_ms=pad_duration_ms)

    leftover_text = ""
    for i, chunk in enumerate(stream):
        input_values = processor(chunk, sampling_rate=sampling_rate, return_tensors="pt").input_values

        with torch.no_grad():
            logits = model(input_values).logits[0]
            if i > 0:
                logits = logits[output_pad_len : len(logits) - output_pad_len]
            else:  # don't count padding at the start of the clip
                logits = logits[: len(logits) - output_pad_len]

            predicted_ids = torch.argmax(logits, dim=-1).cpu().tolist()
            if processor.decode(predicted_ids).strip():
                leftover_ids = processor.tokenizer.encode(leftover_text)
                # concat the last word (or its part) from the last frame with the current text
                text = processor.decode(leftover_ids + predicted_ids)
                # don't return the last word in case it's just partially recognized
                text, leftover_text = text.rsplit(" ", 1)
                yield text
            else:
                yield leftover_text
                leftover_text = ""
    yield leftover_text


texts = []
for text in stream_text(youtube_url, chunk_duration_ms, pad_duration_ms):
    texts.append(text)
