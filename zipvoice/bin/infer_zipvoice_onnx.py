#!/usr/bin/env python3
# Copyright         2025  Xiaomi Corp.        (authors: Han Zhu,
#                                                       Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script generates speech with our pre-trained ZipVoice or ZipVoice-Distill
    ONNX models. If no local model is specified,
    Required files will be automatically downloaded from HuggingFace.

Usage:

Note: If you having trouble connecting to HuggingFace,
    try switching endpoint to mirror site:
export HF_ENDPOINT=https://hf-mirror.com

(1) Inference of a single sentence:

python3 -m zipvoice.bin.infer_zipvoice_onnx \
    --onnx-int8 False \
    --model-name zipvoice \
    --prompt-wav prompt.wav \
    --prompt-text "I am a prompt." \
    --text "I am a sentence." \
    --res-wav-path result.wav

(2) Inference of a list of sentences:
python3 -m zipvoice.bin.infer_zipvoice_onnx \
    --onnx-int8 False \
    --model-name zipvoice \
    --test-list test.tsv \
    --res-dir results

`--model-name` can be `zipvoice` or `zipvoice_distill`,
    which are the models before and after distillation, respectively.

Each line of `test.tsv` is in the format of
    `{wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}`.

Set `--onnx-int8 True` to use int8 quantizated ONNX model.
Quantizated model has faster but lower quality.
"""

import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from lhotse.utils import fix_random_seed
from torch import Tensor, nn

from zipvoice.bin.infer_zipvoice import get_vocoder
from zipvoice.models.modules.solver import get_time_steps
from zipvoice.tokenizer.tokenizer import (
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    SimpleTokenizer,
)
from zipvoice.utils.common import AttributeDict, str2bool
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import (
    add_punctuation,
    chunk_tokens_punctuation,
    cross_fade_concat,
    load_prompt_wav,
    remove_silence,
    rms_norm,
)

HUGGINGFACE_REPO = "k2-fsa/ZipVoice"
MODEL_DIR = {
    "zipvoice": "zipvoice",
    "zipvoice_distill": "zipvoice_distill",
}


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--onnx-int8",
        type=str2bool,
        default=False,
        help="Whether to use the int8 model",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="zipvoice",
        choices=["zipvoice", "zipvoice_distill"],
        help="The model used for inference",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="The path to the local onnx model. "
        "Will download pre-trained checkpoint from huggingface if not specified.",
    )

    parser.add_argument(
        "--vocoder-path",
        type=str,
        default=None,
        help="The vocoder checkpoint. "
        "Will download pre-trained vocoder from huggingface if not specified.",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="emilia",
        choices=["emilia", "libritts", "espeak", "simple"],
        help="Tokenizer type.",
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="en-us",
        help="Language identifier, used when tokenizer type is espeak. see"
        "https://github.com/rhasspy/espeak-ng/blob/master/docs/languages.md",
    )

    parser.add_argument(
        "--test-list",
        type=str,
        default=None,
        help="The list of prompt speech, prompt_transcription, "
        "and text to synthesizein the format of "
        "'{wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}'.",
    )

    parser.add_argument(
        "--prompt-wav",
        type=str,
        default=None,
        help="The prompt wav to mimic",
    )

    parser.add_argument(
        "--prompt-text",
        type=str,
        default=None,
        help="The transcription of the prompt wav",
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="The text to synthesize",
    )

    parser.add_argument(
        "--res-dir",
        type=str,
        default="results",
        help="""
        Path name of the generated wavs dir,
        used when test-list is not None
        """,
    )

    parser.add_argument(
        "--res-wav-path",
        type=str,
        default="result.wav",
        help="""
        Path name of the generated wav path,
        used when test-list is None
        """,
    )

    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="The scale of classifier-free guidance during inference.",
    )

    parser.add_argument(
        "--num-step",
        type=int,
        default=None,
        help="The number of sampling steps.",
    )

    parser.add_argument(
        "--feat-scale",
        type=float,
        default=0.1,
        help="The scale factor of fbank feature",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Control speech speed, 1.0 means normal, >1.0 means speed up",
    )

    parser.add_argument(
        "--t-shift",
        type=float,
        default=0.5,
        help="Shift t to smaller ones if t_shift < 1.0",
    )

    parser.add_argument(
        "--target-rms",
        type=float,
        default=0.1,
        help="Target speech normalization rms value, set to 0 to disable normalization",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=666,
        help="Random seed",
    )

    parser.add_argument(
        "--num-thread",
        type=int,
        default=1,
        help="Number of threads to use for ONNX Runtime and PyTorch.",
    )

    parser.add_argument(
        "--raw-evaluation",
        type=str2bool,
        default=False,
        help="Whether to use the 'raw' evaluation mode where provided "
        "prompts and text are fed to the model without pre-processing",
    )

    parser.add_argument(
        "--remove-long-sil",
        type=str2bool,
        default=False,
        help="Whether to remove long silences in the middle of the generated "
        "speech (edge silences will be removed by default).",
    )
    return parser


class OnnxModel:
    def __init__(
        self,
        text_encoder_path: str,
        fm_decoder_path: str,
        num_thread: int = 1,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = num_thread
        session_opts.intra_op_num_threads = num_thread

        self.session_opts = session_opts

        self.init_text_encoder(text_encoder_path)
        self.init_fm_decoder(fm_decoder_path)

    def init_text_encoder(self, model_path: str):
        self.text_encoder = ort.InferenceSession(
            model_path,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

    def init_fm_decoder(self, model_path: str):
        self.fm_decoder = ort.InferenceSession(
            model_path,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )
        meta = self.fm_decoder.get_modelmeta().custom_metadata_map
        self.feat_dim = int(meta["feat_dim"])

    def run_text_encoder(
        self,
        tokens: Tensor,
        prompt_tokens: Tensor,
        prompt_features_len: Tensor,
        speed: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        out = self.text_encoder.run(
            [
                self.text_encoder.get_outputs()[0].name,
            ],
            {
                self.text_encoder.get_inputs()[0].name: tokens.numpy(),
                self.text_encoder.get_inputs()[1].name: prompt_tokens.numpy(),
                self.text_encoder.get_inputs()[2].name: prompt_features_len.numpy(),
                self.text_encoder.get_inputs()[3].name: speed.numpy(),
            },
        )
        return torch.from_numpy(out[0])

    def run_fm_decoder(
        self,
        t: Tensor,
        x: Tensor,
        text_condition: Tensor,
        speech_condition: torch.Tensor,
        guidance_scale: Tensor,
    ) -> Tensor:
        out = self.fm_decoder.run(
            [
                self.fm_decoder.get_outputs()[0].name,
            ],
            {
                self.fm_decoder.get_inputs()[0].name: t.numpy(),
                self.fm_decoder.get_inputs()[1].name: x.numpy(),
                self.fm_decoder.get_inputs()[2].name: text_condition.numpy(),
                self.fm_decoder.get_inputs()[3].name: speech_condition.numpy(),
                self.fm_decoder.get_inputs()[4].name: guidance_scale.numpy(),
            },
        )
        return torch.from_numpy(out[0])


def sample(
    model: OnnxModel,
    tokens: List[List[int]],
    prompt_tokens: List[List[int]],
    prompt_features: Tensor,
    speed: float = 1.0,
    t_shift: float = 0.5,
    guidance_scale: float = 1.0,
    num_step: int = 16,
) -> torch.Tensor:
    """
    Generate acoustic features, given text tokens, prompts feature and prompt
    transcription's text tokens.

    Args:
        tokens: a list of list of text tokens.
        prompt_tokens: a list of list of prompt tokens.
        prompt_features: the prompt feature with the shape
            (batch_size, seq_len, feat_dim).
        speed : speed control.
        t_shift: time shift.
        guidance_scale: the guidance scale for classifier-free guidance.
        num_step: the number of steps to use in the ODE solver.
    """
    # Run text encoder
    assert len(tokens) == len(prompt_tokens) == 1
    tokens = torch.tensor(tokens, dtype=torch.int64)
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.int64)
    prompt_features_len = torch.tensor(prompt_features.size(1), dtype=torch.int64)
    speed = torch.tensor(speed, dtype=torch.float32)

    text_condition = model.run_text_encoder(
        tokens, prompt_tokens, prompt_features_len, speed
    )

    batch_size, num_frames, _ = text_condition.shape
    assert batch_size == 1
    feat_dim = model.feat_dim

    # Run flow matching model
    timesteps = get_time_steps(
        t_start=0.0,
        t_end=1.0,
        num_step=num_step,
        t_shift=t_shift,
    )
    x = torch.randn(batch_size, num_frames, feat_dim)
    speech_condition = torch.nn.functional.pad(
        prompt_features, (0, 0, 0, num_frames - prompt_features.shape[1])
    )  # (B, T, F)
    guidance_scale = torch.tensor(guidance_scale, dtype=torch.float32)

    for step in range(num_step):
        v = model.run_fm_decoder(
            t=timesteps[step],
            x=x,
            text_condition=text_condition,
            speech_condition=speech_condition,
            guidance_scale=guidance_scale,
        )
        x = x + v * (timesteps[step + 1] - timesteps[step])

    x = x[:, prompt_features_len.item() :, :]
    return x


# Copied from zipvoice/bin/infer_zipvoice.py, but call an external sample function
def generate_sentence_raw_evaluation(
    save_path: str,
    prompt_text: str,
    prompt_wav: str,
    text: str,
    model: OnnxModel,
    vocoder: nn.Module,
    tokenizer: EmiliaTokenizer,
    feature_extractor: VocosFbank,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
):
    """
    Generate waveform of a text based on a given prompt waveform and its transcription,
        this function directly feed the prompt_text, prompt_wav and text to the model.
        It is not efficient and can have poor results for some inappropriate inputs.
        (e.g., prompt wav contains long silence, text to be generated is too long)
        This function can be used to evaluate the "raw" performance of the model.

    Args:
        save_path (str): Path to save the generated wav.
        prompt_text (str): Transcription of the prompt wav.
        prompt_wav (str): Path to the prompt wav file.
        text (str): Text to be synthesized into a waveform.
        model (torch.nn.Module): The model used for generation.
        vocoder (torch.nn.Module): The vocoder used to convert features to waveforms.
        tokenizer (EmiliaTokenizer): The tokenizer used to convert text to tokens.
        feature_extractor (VocosFbank): The feature extractor used to
            extract acoustic features.
        num_step (int, optional): Number of steps for decoding. Defaults to 16.
        guidance_scale (float, optional): Scale for classifier-free guidance.
            Defaults to 1.0.
        speed (float, optional): Speed control. Defaults to 1.0.
        t_shift (float, optional): Time shift. Defaults to 0.5.
        target_rms (float, optional): Target RMS for waveform normalization.
            Defaults to 0.1.
        feat_scale (float, optional): Scale for features.
            Defaults to 0.1.
        sampling_rate (int, optional): Sampling rate for the waveform.
            Defaults to 24000.
    Returns:
        metrics (dict): Dictionary containing time and real-time
            factor metrics for processing.
    """

    # Load and process prompt wav
    prompt_wav = load_prompt_wav(prompt_wav, sampling_rate=sampling_rate)
    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)

    # Extract features from prompt wav
    prompt_features = feature_extractor.extract(prompt_wav, sampling_rate=sampling_rate)

    prompt_features = prompt_features.unsqueeze(0) * feat_scale

    # Convert text to tokens
    tokens = tokenizer.texts_to_token_ids([text])
    prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])

    # Start timing
    start_t = dt.datetime.now()

    # Generate features
    pred_features = sample(
        model=model,
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        prompt_features=prompt_features,
        speed=speed,
        t_shift=t_shift,
        guidance_scale=guidance_scale,
        num_step=num_step,
    )

    # Postprocess predicted features
    pred_features = pred_features.permute(0, 2, 1) / feat_scale  # (B, C, T)

    # Start vocoder processing
    start_vocoder_t = dt.datetime.now()
    wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)

    # Calculate processing times and real-time factors
    t = (dt.datetime.now() - start_t).total_seconds()
    t_no_vocoder = (start_vocoder_t - start_t).total_seconds()
    t_vocoder = (dt.datetime.now() - start_vocoder_t).total_seconds()
    wav_seconds = wav.shape[-1] / sampling_rate
    rtf = t / wav_seconds
    rtf_no_vocoder = t_no_vocoder / wav_seconds
    rtf_vocoder = t_vocoder / wav_seconds
    metrics = {
        "t": t,
        "t_no_vocoder": t_no_vocoder,
        "t_vocoder": t_vocoder,
        "wav_seconds": wav_seconds,
        "rtf": rtf,
        "rtf_no_vocoder": rtf_no_vocoder,
        "rtf_vocoder": rtf_vocoder,
    }

    # Adjust wav volume if necessary
    if prompt_rms < target_rms:
        wav = wav * prompt_rms / target_rms
    torchaudio.save(save_path, wav.cpu(), sample_rate=sampling_rate)

    return metrics


def generate_sentence(
    save_path: str,
    prompt_text: str,
    prompt_wav: str,
    text: str,
    model: OnnxModel,
    vocoder: nn.Module,
    tokenizer: EmiliaTokenizer,
    feature_extractor: VocosFbank,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
    remove_long_sil: bool = False,
):
    """
    Generate waveform of a text based on a given prompt waveform and its transcription,
        this function will do the following to improve the generation quality:
        1. chunk the text according to punctuations.
        2. process chunked texts sequentially.
        3. remove long silences in the prompt audio.
        4. add punctuation to the end of prompt text and text if there is not.

    Args:
        save_path (str): Path to save the generated wav.
        prompt_text (str): Transcription of the prompt wav.
        prompt_wav (str): Path to the prompt wav file.
        text (str): Text to be synthesized into a waveform.
        model (torch.nn.Module): The model used for generation.
        vocoder (torch.nn.Module): The vocoder used to convert features to waveforms.
        tokenizer (EmiliaTokenizer): The tokenizer used to convert text to tokens.
        feature_extractor (VocosFbank): The feature extractor used to
            extract acoustic features.
        num_step (int, optional): Number of steps for decoding. Defaults to 16.
        guidance_scale (float, optional): Scale for classifier-free guidance.
            Defaults to 1.0.
        speed (float, optional): Speed control. Defaults to 1.0.
        t_shift (float, optional): Time shift. Defaults to 0.5.
        target_rms (float, optional): Target RMS for waveform normalization.
            Defaults to 0.1.
        feat_scale (float, optional): Scale for features.
            Defaults to 0.1.
        sampling_rate (int, optional): Sampling rate for the waveform.
            Defaults to 24000.
        remove_long_sil (bool, optional): Whether to remove long silences in the
            middle of the generated speech (edge silences will be removed by default).
    Returns:
        metrics (dict): Dictionary containing time and real-time
            factor metrics for processing.
    """

    # Track timing for each step
    step_times = {}
    total_start_t = dt.datetime.now()

    # Step 1: Load and process prompt wav
    step_start = dt.datetime.now()
    prompt_wav = load_prompt_wav(prompt_wav, sampling_rate=sampling_rate)
    step_times["load_prompt_wav"] = (dt.datetime.now() - step_start).total_seconds()

    # Step 2: Remove edge and long silences in the prompt wav.
    # Add 0.2s trailing silence to avoid leaking prompt to generated speech.
    step_start = dt.datetime.now()
    prompt_wav = remove_silence(
        prompt_wav, sampling_rate, only_edge=False, trail_sil=200
    )
    step_times["remove_silence"] = (dt.datetime.now() - step_start).total_seconds()

    # Step 3: RMS normalization
    step_start = dt.datetime.now()
    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)
    step_times["rms_norm"] = (dt.datetime.now() - step_start).total_seconds()

    prompt_duration = prompt_wav.shape[-1] / sampling_rate

    if prompt_duration > 20:
        logging.warning(
            f"Given prompt wav is too long ({prompt_duration}s). "
            f"Please provide a shorter one (1-3 seconds is recommended)."
        )
    elif prompt_duration > 10:
        logging.warning(
            f"Given prompt wav is long ({prompt_duration}s). "
            f"It will lead to slower inference speed and possibly worse speech quality."
        )

    # Step 4: Extract features from prompt wav
    step_start = dt.datetime.now()
    prompt_features = feature_extractor.extract(prompt_wav, sampling_rate=sampling_rate)

    prompt_features = prompt_features.unsqueeze(0) * feat_scale
    step_times["feature_extraction"] = (dt.datetime.now() - step_start).total_seconds()

    # Step 5: Add punctuation and tokenization
    step_start = dt.datetime.now()
    text = add_punctuation(text)
    prompt_text = add_punctuation(prompt_text)

    # Tokenize text (str tokens), punctuations will be preserved.
    tokens_str = tokenizer.texts_to_tokens([text])[0]
    prompt_tokens_str = tokenizer.texts_to_tokens([prompt_text])[0]
    step_times["tokenization"] = (dt.datetime.now() - step_start).total_seconds()

    # Step 6: Chunk text
    step_start = dt.datetime.now()
    token_duration = (prompt_wav.shape[-1] / sampling_rate) / (
        len(prompt_tokens_str) * speed
    )
    max_tokens = int((25 - prompt_duration) / token_duration)
    chunked_tokens_str = chunk_tokens_punctuation(tokens_str, max_tokens=max_tokens)
    print(f"Number of chunks: {len(chunked_tokens_str)}")
    print(f"Chunked tokens: {chunked_tokens_str}")

    # Tokenize text (int tokens)
    chunked_tokens = tokenizer.tokens_to_token_ids(chunked_tokens_str)
    prompt_tokens = tokenizer.tokens_to_token_ids([prompt_tokens_str])
    step_times["text_chunking"] = (dt.datetime.now() - step_start).total_seconds()

    # Step 7: Generate features for each chunk
    step_start = dt.datetime.now()
    chunked_features = []
    chunk_generation_times = []

    for chunk_idx, tokens in enumerate(chunked_tokens):
        chunk_start = dt.datetime.now()

        # Generate features
        pred_features = sample(
            model=model,
            tokens=[tokens],
            prompt_tokens=prompt_tokens,
            prompt_features=prompt_features,
            speed=speed,
            t_shift=t_shift,
            guidance_scale=guidance_scale,
            num_step=num_step,
        )

        # Postprocess predicted features
        pred_features = pred_features.permute(0, 2, 1) / feat_scale  # (B, C, T)
        chunked_features.append(pred_features)

        chunk_time = (dt.datetime.now() - chunk_start).total_seconds()
        chunk_generation_times.append(chunk_time)
        logging.info(
            f"Chunk {chunk_idx + 1}/{len(chunked_tokens)} generation time: {chunk_time:.4f}s"
        )

    step_times["feature_generation"] = (dt.datetime.now() - step_start).total_seconds()
    step_times["feature_generation_per_chunk"] = chunk_generation_times

    # Step 8: Vocoder processing
    step_start = dt.datetime.now()
    chunked_wavs = []
    chunk_vocoder_times = []

    for chunk_idx, pred_features in enumerate(chunked_features):
        chunk_start = dt.datetime.now()

        wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)
        # Adjust wav volume if necessary
        if prompt_rms < target_rms:
            wav = wav * prompt_rms / target_rms
        chunked_wavs.append(wav)

        chunk_time = (dt.datetime.now() - chunk_start).total_seconds()
        chunk_vocoder_times.append(chunk_time)
        logging.info(
            f"Chunk {chunk_idx + 1}/{len(chunked_features)} vocoder time: {chunk_time:.4f}s"
        )

    step_times["vocoder_processing"] = (dt.datetime.now() - step_start).total_seconds()
    step_times["vocoder_processing_per_chunk"] = chunk_vocoder_times

    # Step 9: Merge chunked wavs
    step_start = dt.datetime.now()
    final_wav = cross_fade_concat(
        chunked_wavs, fade_duration=0.1, sample_rate=sampling_rate
    )
    step_times["wav_merging"] = (dt.datetime.now() - step_start).total_seconds()

    # Step 10: Remove silence from final wav
    step_start = dt.datetime.now()
    final_wav = remove_silence(
        final_wav, sampling_rate, only_edge=(not remove_long_sil), trail_sil=0
    )
    step_times["final_silence_removal"] = (
        dt.datetime.now() - step_start
    ).total_seconds()

    # Calculate overall processing time metrics
    t_total = (dt.datetime.now() - total_start_t).total_seconds()
    t_no_vocoder = step_times["feature_generation"]
    t_vocoder = step_times["vocoder_processing"]
    wav_seconds = final_wav.shape[-1] / sampling_rate
    rtf = t_total / wav_seconds
    rtf_no_vocoder = t_no_vocoder / wav_seconds
    rtf_vocoder = t_vocoder / wav_seconds

    # Log detailed timing breakdown
    logging.info("=" * 60)
    logging.info("Timing Breakdown:")
    logging.info(f"  1. Load prompt wav:          {step_times['load_prompt_wav']:.4f}s")
    logging.info(f"  2. Remove silence:           {step_times['remove_silence']:.4f}s")
    logging.info(f"  3. RMS normalization:        {step_times['rms_norm']:.4f}s")
    logging.info(
        f"  4. Feature extraction:       {step_times['feature_extraction']:.4f}s"
    )
    logging.info(f"  5. Tokenization:             {step_times['tokenization']:.4f}s")
    logging.info(f"  6. Text chunking:            {step_times['text_chunking']:.4f}s")
    logging.info(
        f"  7. Feature generation:       {step_times['feature_generation']:.4f}s"
    )
    logging.info(
        f"  8. Vocoder processing:       {step_times['vocoder_processing']:.4f}s"
    )
    logging.info(f"  9. Wav merging:              {step_times['wav_merging']:.4f}s")
    logging.info(
        f" 10. Final silence removal:   {step_times['final_silence_removal']:.4f}s"
    )
    logging.info(
        f"Total time: {t_total:.4f}s | Generated audio: {wav_seconds:.2f}s | RTF: {rtf:.4f}"
    )
    logging.info("=" * 60)

    metrics = {
        "t": t_total,
        "t_no_vocoder": t_no_vocoder,
        "t_vocoder": t_vocoder,
        "wav_seconds": wav_seconds,
        "rtf": rtf,
        "rtf_no_vocoder": rtf_no_vocoder,
        "rtf_vocoder": rtf_vocoder,
        "step_times": step_times,
    }

    torchaudio.save(save_path, final_wav.cpu(), sample_rate=sampling_rate)
    return metrics


def generate_list(
    res_dir: str,
    test_list: str,
    model: OnnxModel,
    vocoder: nn.Module,
    tokenizer: EmiliaTokenizer,
    feature_extractor: VocosFbank,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
    raw_evaluation: bool = False,
    remove_long_sil: bool = False,
):
    total_t = []
    total_t_no_vocoder = []
    total_t_vocoder = []
    total_wav_seconds = []

    with open(test_list, "r") as fr:
        lines = fr.readlines()

    for i, line in enumerate(lines):
        wav_name, prompt_text, prompt_wav, text = line.strip().split("\t")
        save_path = f"{res_dir}/{wav_name}.wav"

        common_params = {
            "save_path": save_path,
            "prompt_text": prompt_text,
            "prompt_wav": prompt_wav,
            "text": text,
            "model": model,
            "vocoder": vocoder,
            "tokenizer": tokenizer,
            "feature_extractor": feature_extractor,
            "num_step": num_step,
            "guidance_scale": guidance_scale,
            "speed": speed,
            "t_shift": t_shift,
            "target_rms": target_rms,
            "feat_scale": feat_scale,
            "sampling_rate": sampling_rate,
        }

        if raw_evaluation:
            metrics = generate_sentence_raw_evaluation(**common_params)
        else:
            metrics = generate_sentence(
                **common_params,
                remove_long_sil=remove_long_sil,
            )
        logging.info(f"[Sentence: {i}] Saved to: {save_path}")
        logging.info(f"[Sentence: {i}] RTF: {metrics['rtf']:.4f}")
        total_t.append(metrics["t"])
        total_t_no_vocoder.append(metrics["t_no_vocoder"])
        total_t_vocoder.append(metrics["t_vocoder"])
        total_wav_seconds.append(metrics["wav_seconds"])

    logging.info(f"Average RTF: {np.sum(total_t) / np.sum(total_wav_seconds):.4f}")
    logging.info(
        f"Average RTF w/o vocoder: "
        f"{np.sum(total_t_no_vocoder) / np.sum(total_wav_seconds):.4f}"
    )
    logging.info(
        f"Average RTF vocoder: "
        f"{np.sum(total_t_vocoder) / np.sum(total_wav_seconds):.4f}"
    )


@torch.inference_mode()
def main():
    parser = get_parser()
    args = parser.parse_args()

    torch.set_num_threads(args.num_thread)
    torch.set_num_interop_threads(args.num_thread)

    params = AttributeDict()
    params.update(vars(args))
    fix_random_seed(params.seed)

    model_defaults = {
        "zipvoice": {
            "num_step": 16,
            "guidance_scale": 1.0,
        },
        "zipvoice_distill": {
            "num_step": 8,
            "guidance_scale": 3.0,
        },
    }

    model_specific_defaults = model_defaults.get(params.model_name, {})

    for param, value in model_specific_defaults.items():
        if getattr(params, param) is None:
            setattr(params, param, value)
            logging.info(f"Setting {param} to default value: {value}")

    assert (params.test_list is not None) ^ (
        (params.prompt_wav and params.prompt_text and params.text) is not None
    ), (
        "For inference, please provide prompts and text with either '--test-list'"
        " or '--prompt-wav, --prompt-text and --text'."
    )

    if params.onnx_int8:
        text_encoder_name = "text_encoder_int8.onnx"
        fm_decoder_name = "fm_decoder_int8.onnx"
    else:
        text_encoder_name = "text_encoder.onnx"
        fm_decoder_name = "fm_decoder.onnx"

    if params.model_dir is not None:
        params.model_dir = Path(params.model_dir)
        if not params.model_dir.is_dir():
            raise FileNotFoundError(f"{params.model_dir} does not exist")

        for filename in [
            text_encoder_name,
            fm_decoder_name,
            "model.json",
            "tokens.txt",
        ]:
            if not (params.model_dir / filename).is_file():
                raise FileNotFoundError(f"{params.model_dir / filename} does not exist")
        text_encoder_path = params.model_dir / text_encoder_name
        fm_decoder_path = params.model_dir / fm_decoder_name
        model_config = params.model_dir / "model.json"
        token_file = params.model_dir / "tokens.txt"
        logging.info(f"Using local model dir {params.model_dir}.")
    else:
        logging.info("Using pretrained model from the Huggingface")
        text_encoder_path = hf_hub_download(
            HUGGINGFACE_REPO,
            filename=f"{MODEL_DIR[params.model_name]}/{text_encoder_name}",
        )
        fm_decoder_path = hf_hub_download(
            HUGGINGFACE_REPO,
            filename=f"{MODEL_DIR[params.model_name]}/{fm_decoder_name}",
        )
        model_config = hf_hub_download(
            HUGGINGFACE_REPO, filename=f"{MODEL_DIR[params.model_name]}/model.json"
        )

        token_file = hf_hub_download(
            HUGGINGFACE_REPO, filename=f"{MODEL_DIR[params.model_name]}/tokens.txt"
        )

    if params.tokenizer == "emilia":
        tokenizer = EmiliaTokenizer(token_file=token_file)
    elif params.tokenizer == "libritts":
        tokenizer = LibriTTSTokenizer(token_file=token_file)
    elif params.tokenizer == "espeak":
        tokenizer = EspeakTokenizer(token_file=token_file, lang=params.lang)
    else:
        assert params.tokenizer == "simple"
        tokenizer = SimpleTokenizer(token_file=token_file)

    with open(model_config, "r") as f:
        model_config = json.load(f)

    model = OnnxModel(text_encoder_path, fm_decoder_path, num_thread=args.num_thread)

    vocoder = get_vocoder(params.vocoder_path)
    vocoder.eval()

    if model_config["feature"]["type"] == "vocos":
        feature_extractor = VocosFbank()
    else:
        raise NotImplementedError(
            f"Unsupported feature type: {model_config['feature']['type']}"
        )
    params.sampling_rate = model_config["feature"]["sampling_rate"]

    logging.info("Start generating...")
    if params.test_list:
        os.makedirs(params.res_dir, exist_ok=True)
        generate_list(
            res_dir=params.res_dir,
            test_list=params.test_list,
            model=model,
            vocoder=vocoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            num_step=params.num_step,
            guidance_scale=params.guidance_scale,
            speed=params.speed,
            t_shift=params.t_shift,
            target_rms=params.target_rms,
            feat_scale=params.feat_scale,
            sampling_rate=params.sampling_rate,
            raw_evaluation=params.raw_evaluation,
            remove_long_sil=params.remove_long_sil,
        )
    else:
        assert (
            not params.raw_evaluation
        ), "Raw evaluation is only valid with --test-list"
        generate_sentence(
            save_path=params.res_wav_path,
            prompt_text=params.prompt_text,
            prompt_wav=params.prompt_wav,
            text=params.text,
            model=model,
            vocoder=vocoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            num_step=params.num_step,
            guidance_scale=params.guidance_scale,
            speed=params.speed,
            t_shift=params.t_shift,
            target_rms=params.target_rms,
            feat_scale=params.feat_scale,
            sampling_rate=params.sampling_rate,
            remove_long_sil=params.remove_long_sil,
        )
        logging.info(f"Saved to: {params.res_wav_path}")
    logging.info("Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    main()
