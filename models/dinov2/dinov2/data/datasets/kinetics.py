# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import csv
import torch
from enum import Enum
import logging
import os
import random
from typing import Callable, List, Optional, Tuple, Union, Any
from PIL import Image

import numpy as np

from .extended import ExtendedVisionDataset
from .decoders import TargetDecoder, ImageDataDecoder


logger = logging.getLogger("dinov2")
_Target = int


class _Split(Enum):
    TRAIN = "train"
    # VAL = "val"
    # TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 239_789, # 241_258
            # _Split.VAL: 50_000,
            # _Split.TEST: 100_000,
        }
        return split_lengths[self]

    def get_dirname(self, class_id: Optional[str] = None) -> str:
        return self.value if class_id is None else os.path.join(self.value, class_id)

    def get_video_folder_relpath(self, video_id: str, class_name: str) -> str:
        dirname = self.get_dirname()
        return os.path.join(dirname, class_name, video_id)

    def parse_image_relpath(self, image_relpath: str) -> Tuple[str, int]:
        assert self == _Split.TRAIN
        dirname, filename = os.path.split(image_relpath)
        class_name = os.path.split(dirname)[-1]
        basename, _ = os.path.splitext(filename)
        video_id = basename
        return class_name, video_id


class Kinetics(ExtendedVisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "Kinetics.Split",
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        mixup: float = 0,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split

        self._entries = None
        self._class_ids = None
        self._class_names = None
        
        self._mixup = mixup


    @property
    def split(self) -> "Kinetics.Split":
        return self._split

    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_full_path = self._get_extra_full_path(extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        extra_full_path = self._get_extra_full_path(extra_path)
        os.makedirs(self._extra_root, exist_ok=True)
        np.save(extra_full_path, extra_array)

    @property
    def _entries_path(self) -> str:
        return f"entries-{self._split.value.upper()}.npy"

    @property
    def _class_ids_path(self) -> str:
        return f"class-ids-{self._split.value.upper()}.npy"

    @property
    def _class_names_path(self) -> str:
        return f"class-names-{self._split.value.upper()}.npy"

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    def _get_class_ids(self) -> np.ndarray:
        # if self._split == _Split.TEST:
        #     assert False, "Class IDs are not available in TEST split"
        if self._class_ids is None:
            self._class_ids = self._load_extra(self._class_ids_path)
        assert self._class_ids is not None
        return self._class_ids

    def _get_class_names(self) -> np.ndarray:
        # if self._split == _Split.TEST:
        #     assert False, "Class names are not available in TEST split"
        if self._class_names is None:
            self._class_names = self._load_extra(self._class_names_path)
        assert self._class_names is not None
        return self._class_names

    def find_class_id(self, class_index: int) -> str:
        class_ids = self._get_class_ids()
        return str(class_ids[class_index])

    def find_class_name(self, class_index: int) -> str:
        class_names = self._get_class_names()
        return str(class_names[class_index])

    def get_image_data(self, index: int) -> bytes:
        entries = self._get_entries()
        video_id = entries[index]["video_id"]
        class_name = self.get_class_name(index)
        video_folder_relpath = self.split.get_video_folder_relpath(video_id, class_name)
        video_folder_full_path = os.path.join(self.root, video_folder_relpath)

        frame_file = self.get_frames(video_folder_full_path)
        frame_path = os.path.join(video_folder_full_path, frame_file)
        
        with open(frame_path, mode="rb") as f:
            frame = f.read()
        image_data = frame
        
        return image_data
        
    def get_dataset(self):
        return "Kinetics"

    # def get_attr(self):
    #     return self._use_knn_feat, self._use_same_class

    def get_target(self, index: int) -> Optional[Target]:
        entries = self._get_entries()
        class_index = entries[index]["class_index"]
        # return None if self.split == _Split.TEST else int(class_index)
        return int(class_index)

    def get_targets(self) -> Optional[np.ndarray]:
        entries = self._get_entries()
        # return None if self.split == _Split.TEST else entries["class_index"]
        return entries["class_index"]

    def get_class_id(self, index: int) -> Optional[str]:
        entries = self._get_entries()
        class_id = entries[index]["class_id"]
        # return None if self.split == _Split.TEST else str(class_id)
        return str(class_id)

    def get_class_name(self, index: int) -> Optional[str]:
        entries = self._get_entries()
        class_name = entries[index]["class_name"]
        # return None if self.split == _Split.TEST else str(class_name)
        return str(class_name)
    
    def get_frames(self, full_path):
        frames = sorted(f for f in os.listdir(full_path))
        total_frames = len(frames)
        frame = frames[random.randint(0, total_frames - 1)]
        return frame

    def __len__(self) -> int:
        entries = self._get_entries()
        assert len(entries) == self.split.length
        return len(entries)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # if mixup is 0, use father class to get item
        if self._mixup <= 0 or np.random.rand() < 0.5:
            return super().__getitem__(index)

        def _load_image(index):
            try:
                image_data = self.get_image_data(index)
                image = ImageDataDecoder(image_data).decode()
            except Exception as e:
                raise RuntimeError(f"can not read image for sample {index}") from e
            return image

        index_random = np.random.randint(0, len(self))
        alpha = np.random.rand() * self._mixup
        image = _load_image(index)
        image_random = _load_image(index_random)
        image_random = image_random.resize(image.size, resample=Image.BILINEAR)
        image = Image.blend(image, image_random, alpha)

        target = self.get_target(index)
        target = TargetDecoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target

    def _load_labels(self, labels_path: str) -> List[Tuple[str, str]]:
        labels_full_path = os.path.join(self.root, labels_path)
        labels = []
        
        try:
            with open(labels_full_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    class_id, class_name = row
                    labels.append((class_id, class_name))
        except OSError as e:
            raise RuntimeError(f'can not read labels file "{labels_full_path}"') from e

        return labels
    
