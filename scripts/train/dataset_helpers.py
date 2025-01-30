# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import dataclasses
import json
import random
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import Compose, RandAugment, RandomResizedCrop, Resize, ToPILImage

from megatron.energon import Batch, CaptioningSample, DefaultTaskEncoder, OCRSample, VQASample
from megatron.energon.transforms import CustomTransform, MergeTransform
from megatron.training import get_args, get_tokenizer
from megatron.training.tokenizer.tokenizer import _HuggingFaceTokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


# Imagenet's mean and std.
pixel_mean = [123.675, 116.28, 103.53]
pixel_std = [58.395, 57.12, 57.375]

pixel_mean_siglip = [127.5, 127.5, 127.5]
pixel_std_siglip = [127.5, 127.5, 127.5]


def convert_to_rgb(image):
    return image.convert("RGB")

def _transform_train(img_h, img_w):
    return Compose([
        ToPILImage(),
        RandomResizedCrop((img_h, img_w), scale=(0.5, 1.0)),
        convert_to_rgb,
    ])

def _transform_train_aug(img_h, img_w):
    return Compose([
        ToPILImage(),
        RandomResizedCrop((img_h, img_w), scale=(0.5, 1.0)),
        convert_to_rgb,
        RandAugment(2, 5, isPIL=True, augs=['Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
    ])

def _transform_test(img_h, img_w):
    return Compose([
        ToPILImage(),
        Resize((img_h, img_w)),
        convert_to_rgb,
    ])

class RandomResize(CustomTransform):
    """Resizes the image by a random scale factor in the given interval, but at most max_size"""

    def __init__(self, min_scale: float, max_scale: float, max_size: int):
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._max_size = max_size

    def apply_transform(self, matrix: np.ndarray, dst_size: np.ndarray) -> Tuple[Any, Any, Any]:
        scale = random.uniform(self._min_scale, self._max_scale)
        new_size = tuple(int(x * scale) for x in dst_size)

        if max(new_size) > self._max_size:
            scale = self._max_size / max(new_size)
            new_size = tuple(int(x * scale) for x in dst_size)

        matrix = self.scale(scale, scale) @ matrix
        dst_size = np.array(new_size, dtype=dst_size.dtype)

        return matrix, dst_size, (self.__class__.__name__, scale)


class RandomResizeLongEdge(CustomTransform):
    """Resizes the image's longer edge to a random length between min_size and max_size pixels."""

    def __init__(self, min_size: int, max_size: int):
        self._min_size = min_size
        self._max_size = max_size

    def apply_transform(self, matrix: np.ndarray, dst_size: np.ndarray) -> Tuple[Any, Any, Any]:
        new_long = random.randint(self._min_size, self._max_size)
        if dst_size[0] > dst_size[1]:  # h > w
            new_w, new_h = int(new_long * dst_size[1] / dst_size[0]), new_long
        else:  # w > h
            new_w, new_h = new_long, int(new_long * dst_size[0] / dst_size[1])

        new_size = (new_h, new_w)
        matrix = self.scale(new_w / dst_size[1], new_h / dst_size[0]) @ matrix
        dst_size = np.array(new_size, dtype=dst_size.dtype)

        return matrix, dst_size, (self.__class__.__name__, new_size)


class RandomPad(CustomTransform):
    """Pads the image to the given size, randomly choosing the position of the image within the new larger image.
    If the image is already larger than the given size, it will not be padded in that direction(s)."""

    def __init__(self, size: Tuple[int, int]):
        self._new_size = size  # h, w

    def apply_transform(self, matrix: np.ndarray, dst_size: np.ndarray) -> Tuple[Any, Any, Any]:
        h_pad = max(self._new_size[0] - dst_size[0], 0)
        w_pad = max(self._new_size[1] - dst_size[1], 0)

        if h_pad == 0 and w_pad == 0:
            return matrix, dst_size, (self.__class__.__name__, None)
        else:
            # TODO: fix me
            # top = random.randint(0, h_pad)
            # left = random.randint(0, w_pad)
            top = 0
            left = 0

            matrix = self.translate(left, top) @ matrix
            dst_size = np.array(self._new_size, dtype=dst_size.dtype)
            return matrix, dst_size, (self.__class__.__name__, (top, left))


def _get_ocr_document_visual_transform(IMG_H=1024, IMG_W=1024):
    document_visual_transform = T.Compose(
        [
            MergeTransform(
                [
                    # T.RandomResizedCrop(size=FINAL_SIZE, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
                    RandomResizeLongEdge(960, 1008),  # Note: 1008 comes from list(range(960, 1024, 16))[-1]
                    T.RandomRotation(5, interpolation=T.InterpolationMode.BILINEAR),
                    T.RandomPerspective(distortion_scale=0.1, p=0.1),
                    RandomPad((IMG_H, IMG_W)),
                ]
            ),
            T.ColorJitter(brightness=(0.8, 1.2), contrast=(0.7, 1.0)),
            T.RandomGrayscale(p=0.5),
            T.RandomInvert(p=0.5),
            T.RandomAdjustSharpness(sharpness_factor=0.0, p=0.5),
            T.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
            # LogImage(),
            # T.ToTensor(),
            # T.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )
    return document_visual_transform

def _get_ocr_document_identity_transform(IMG_H=1024, IMG_W=1024):
    long_edge = max(IMG_H, IMG_W)
    document_identity_transform = T.Compose(
        [
            MergeTransform(
                [
                    RandomResizeLongEdge(long_edge, long_edge),
                    RandomPad((long_edge, long_edge)),
                ]
            )
        ]
    )
    return document_identity_transform

def _get_ocr_paragraph_visual_transform(IMG_H=1024, IMG_W=1024):
    paragraph_visual_transform = T.Compose(
        [
            MergeTransform(
                [
                    # T.RandomResizedCrop(size=FINAL_SIZE, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
                    RandomResize(0.5, 2.0, min(IMG_H, IMG_W)), #FINAL_SIZE),
                    T.RandomRotation(1, interpolation=T.InterpolationMode.BILINEAR),
                    T.RandomPerspective(distortion_scale=0.1, p=0.1),
                    RandomPad((IMG_H, IMG_W)),
                ]
            ),
            T.ColorJitter(brightness=(0.8, 1.2), contrast=(0.7, 1.0)),
            T.RandomGrayscale(p=0.5),
            T.RandomInvert(p=0.5),
            # T.RandomAdjustSharpness(sharpness_factor=0.0, p=0.5),
            # T.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
            # LogImage(),
            # T.ToTensor(),
            # T.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )
    return paragraph_visual_transform

# Type for intermediate batch, after batch()
@dataclass
class ImageTaskSample:
    __key__: str
    __subflavors__: Dict
    # (c, h, w)
    img: torch.Tensor
    text: np.ndarray
    prompt_len: np.int64
    img_clip: Optional[torch.Tensor] = None


# Typing for the resulting batch data after encode_batch()
@dataclass
class ImageTaskBatch(Batch):
    __keys__: List[str]
    __subflavors__: List[Dict]
    # (n, c, h, w)
    img: torch.Tensor
    # (n, seq_len)
    text: torch.Tensor
    # (n, 1)
    prompt_len: torch.Tensor
    # (n, c, h, w)
    img_clip: Optional[torch.Tensor] = None


class IdentitySplitter(object):
    def tokenize(self, *text):
        return text


class TaskEncoder(DefaultTaskEncoder[OCRSample, OCRSample, ImageTaskBatch, dict]):
    """A simple task encoder for captioning."""
    def __init__(
        self
    ):
        # Specify the batch_type for default batching (batching is performed here "manually" by
        # overwriting the `batch` method)
        super().__init__()

        self.args = get_args()

        self.training_stage = self.args.training_stage
        assert self.training_stage in [1, 2], f"Invalid training stage: {self.training_stage}"

        self.tokenizer: _HuggingFaceTokenizer = get_tokenizer()
        self.tokenizer.add_eos_token = False

        self.manual_prompts = json.load(open(self.args.prompt_path))
        self.seq_len = self.args.seq_length

        self.txt_to_token_dict = {}

        self.img_h, self.img_w = self.args.img_h, self.args.img_w

        print(f"vision_model_subtype: {self.args.vision_model_subtype}")
        if self.args.vision_model_subtype == "siglip":
            self.pixel_mean = torch.Tensor(pixel_mean_siglip).view(-1, 1, 1)
            self.pixel_std = torch.Tensor(pixel_std_siglip).view(-1, 1, 1)
            print("Using Siglip vision model's pixel mean and std")
        else:
            self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
            self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
            print("Using Imagenet's pixel mean and std")

        self.manual_prompts = json.load(open(self.args.prompt_path))

        # self.pre_prompt = "USER: "
        # self.post_prompt = "ASSISTANT: "
        self.pre_prompt = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n"
        self.post_prompt = "### 応答:\n"


    def get_visual_transform(self, img_sample, sample_augmentation=False):
        raw_h, raw_w = img_sample.shape[0], img_sample.shape[1]
        ratio = float(max(self.img_h, self.img_w)) / max(raw_h, raw_w)
        scaled_h, scaled_w = int(raw_h * ratio + 0.5), int(raw_w * ratio + 0.5)

        # if the sample needs augmentation or not
        if sample_augmentation:
            # further check if augmentation is a global flag in args
            if self.args.aug:
                visual_transform = _transform_train_aug(scaled_h, scaled_w)
            else:
                visual_transform = _transform_train(scaled_h, scaled_w)
        else:
            visual_transform = _transform_test(scaled_h, scaled_w)

        img = visual_transform(img_sample)

        # Normalize pixel values.
        img = (torch.Tensor(np.array(img)).permute(2, 0, 1) - self.pixel_mean) / self.pixel_std

        # Pad to target image size.
        delta_h, delta_w = self.img_h - scaled_h, self.img_w - scaled_w
        img = torch.nn.functional.pad(img, (0, delta_w, 0, delta_h))

        return img

    def encode_sample(self, sample: Union[
        CaptioningSample, OCRSample, VQASample]
        ):
        if isinstance(sample, VQASample):
            yield self.encode_vqa(sample)
        else:
            raise NotImplementedError('Sample format not supported')
            yield None

    def encode_vqa(self, sample: VQASample):

        no_image_flag = True if '-noimage' in sample.__key__ else False

        if 'pretrain' in sample.__key__:
            pass
        else:
            sample.__key__.split("/")[0]

        sample_augmentation = sample.__subflavors__["augmentation"] is True

        if no_image_flag:
            img = torch.from_numpy(np.array([0]).astype(np.float32))
        else:
            img = self.get_visual_transform(np.array(sample.image), sample_augmentation=sample_augmentation)

        if "<image>" in sample.context:
            sample.context = sample.context.replace("<image>","")

        # assign prompt
        if self.training_stage == 2:
            if sample.context == "":
                # case: caption
                # multiple "。" in the answer, assign detailed_caption_ja
                if sample.answers.count("。") > 1:
                    prompt_candidates = self.manual_prompts["detailed_caption_ja"]
                else:
                    prompt_candidates = self.manual_prompts["caption_ja"]
            else:
                # case: VQA
                # "。" is in the answer, assign detailed_question_ja
                if "。" in sample.answers:
                    prompt_candidates = self.manual_prompts["detailed_vqa_ja"]
                else:
                    prompt_candidates = self.manual_prompts["vqa_ja"]
            prompt = random.choice(prompt_candidates["raw"])
        else:
            prompt = ""

        sample.context = f"{self.pre_prompt}<image>\n{prompt}{sample.context}\n\n{self.post_prompt}"

        question_token = self.tokenizer.tokenize(sample.context)
        if isinstance(sample.answers, list):
            answer_list = sample.answers
            weight_list = np.array(sample.answer_weights).astype(np.float32)
            weight_list = weight_list / np.sum(weight_list)
            answer_idx = np.random.choice(weight_list.shape[0], 1, p=weight_list)[0]
            answer = answer_list[answer_idx]
        else:
            answer = sample.answers

        answer = answer.strip()
        answer_token = self.tokenizer.tokenize(answer, add_special_tokens=False)

        prompt_len = len(question_token) - 1  # do not count <image> tokens
        # print(f"prompt_len (dataset_helpers): {prompt_len}")
        # print(f"question_token (dataset_helpers): {question_token}")
        # print(f"answer_token (dataset_helpers): {answer_token}")

        seq_len = self.seq_len + 4

        text_sample = np.concatenate([question_token, answer_token])
        # padding
        text_len = len(text_sample)
        pad_len = seq_len - text_len
        if pad_len < 0:
            text_sample = text_sample[:seq_len]
            pad_len = 0
        text_sample = np.pad(text_sample, (0, pad_len), mode='constant', constant_values=self.tokenizer.pad)
        # print(f"text: {self.tokenizer._tokenizer.decode(text_sample, skip_special_tokens=True)}")

        return ImageTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            img=img,
            text=text_sample,
            prompt_len=prompt_len
        )

    def batch(self, samples: List[ImageTaskSample]) -> ImageTaskBatch:
        batch = ImageTaskBatch(
            __keys__=[s.__key__ for s in samples],
            __subflavors__=[s.__subflavors__ for s in samples],
            img=torch.stack([s.img for s in samples]),
            text=torch.from_numpy(np.stack([s.text for s in samples], axis=0).astype(np.int64)),
            prompt_len=torch.from_numpy(np.array([s.prompt_len for s in samples], dtype=np.int64))
        )

        return batch

    def encode_batch(self, batch: ImageTaskBatch) -> dict:
        raw = dataclasses.asdict(batch)
        del raw["__subflavors__"]
        return raw


def print_error_handler(exc: Exception, key: Optional[str]):
    print(
        f"The following exception occurred in the dataloader for sample {key} and is skipped",
        file=sys.stderr,
    )
    traceback.print_exc()
