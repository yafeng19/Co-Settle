import torchvision.transforms.v2 as T2
from torchvision.transforms import functional as F
import torch.utils.data as data
import random
import torch
from .video.image_utils import SingleRRC, DoubleRRC, DoubleTupleRRC
from .video.video_utils import sample_two_frames, sample_two_frames_tuple


class SingleRandomResizedCrop:

    def __init__(
            self,
            hflip_p=0.5,
            size=(224, 224),
            scale=(0.5, 1.0),
            ratio=(3./4., 4./3.),
            interpolation=F.InterpolationMode.BILINEAR,
            antialias=True,
            use_same_crop=True
        ):
        self.hflip_p = hflip_p
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.use_same_crop = use_same_crop
        self.resized_crop = SingleRRC(size, scale, ratio, interpolation, antialias)
        self.post_process = T2.Compose([
            T2.ToImage(),
            T2.ToDtype(torch.float32, scale=True),
            T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img1):
        
        # Crop
        cropped_img = self.resized_crop(img1)
        
        # Flip
        if random.random() < self.hflip_p:
            cropped_img = F.hflip(cropped_img)

        # Convert to float32 tensor + Normalize
        cropped_img = self.post_process(cropped_img)

        return cropped_img
    

class DoubleRandomResizedCrop:

    def __init__(
            self,
            hflip_p=0.5,
            size=(224, 224),
            scale=(0.5, 1.0),
            ratio=(3./4., 4./3.),
            interpolation=F.InterpolationMode.BILINEAR,
            antialias=True,
            use_same_crop=True
        ):
        self.hflip_p = hflip_p
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.use_same_crop = use_same_crop
        self.resized_crop = DoubleRRC(size, scale, ratio, interpolation, antialias)
        self.post_process = T2.Compose([
            T2.ToImage(),
            T2.ToDtype(torch.float32, scale=True),
            T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img1, img2):
        
        # Crop
        cropped_img_1, cropped_img_2 = self.resized_crop(img1, img2, same_crop=self.use_same_crop)
        
        # Flip
        if random.random() < self.hflip_p:
            cropped_img_1 = F.hflip(cropped_img_1)
            cropped_img_2 = F.hflip(cropped_img_2)

        # Convert to float32 tensor + Normalize
        cropped_img_1 = self.post_process(cropped_img_1)
        cropped_img_2 = self.post_process(cropped_img_2)
            
        return cropped_img_1, cropped_img_2


class DoubleTupleRandomResizedCrop:

    def __init__(
            self,
            hflip_p=0.5,
            size=(224, 224),
            scale=(0.5, 1.0),
            ratio=(3./4., 4./3.),
            interpolation=F.InterpolationMode.BILINEAR,
            antialias=True,
            use_same_crop=True
        ):
        self.hflip_p = hflip_p
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.use_same_crop = use_same_crop
        self.resized_crop = DoubleTupleRRC(size, scale, ratio, interpolation, antialias)
        self.post_process = T2.Compose([
            T2.ToImage(),
            T2.ToDtype(torch.float32, scale=True),
            T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img1, img2, img3, img4):
        
        # Crop
        cropped_img_1, cropped_img_2, cropped_img_3, cropped_img_4 \
        = self.resized_crop(img1, img2, img3, img4, same_crop=self.use_same_crop)
        
        # Flip
        if random.random() < self.hflip_p:
            cropped_img_1 = F.hflip(cropped_img_1)
            cropped_img_2 = F.hflip(cropped_img_2)
            cropped_img_3 = F.hflip(cropped_img_3)
            cropped_img_4 = F.hflip(cropped_img_4)

        # Convert to float32 tensor + Normalize
        cropped_img_1 = self.post_process(cropped_img_1)
        cropped_img_2 = self.post_process(cropped_img_2)
        cropped_img_3 = self.post_process(cropped_img_3)
        cropped_img_4 = self.post_process(cropped_img_4)
            
        return cropped_img_1, cropped_img_2, cropped_img_3, cropped_img_4


class VideoDataset(data.Dataset):
    def __init__(self, files, args):

        interpolation = getattr(F.InterpolationMode, 'BICUBIC')

        self.args = args
        if self.args.use_adapter:
            self.transforms_manager = DoubleTupleRandomResizedCrop(use_same_crop=True, interpolation=interpolation)
        else:
            self.transforms_manager = DoubleRandomResizedCrop(use_same_crop=True, interpolation=interpolation)
        self.files = files

    def __len__(self,):
        return len(self.files)

    def __getitem__(self, idx):
        video_path = self.files[idx]
        if self.args.use_adapter:
            f1, f2, f3, f4 = sample_two_frames_tuple(video_path, clip_gap=self.args.clip_gap)
            f1, f2, f3, f4 = self.transforms_manager(f1, f2, f3, f4)
            frames_1 = [f1.unsqueeze(0), f2.unsqueeze(0)]
            frames_2 = [f3.unsqueeze(0), f4.unsqueeze(0)]
            return torch.cat(frames_1, dim=0), torch.cat(frames_2, dim=0)
        else:
            f1, f2 = sample_two_frames(video_path, clip_gap=0.15)
            f1, f2= self.transforms_manager(f1, f2)
            frames_1 = f1.unsqueeze(0)
            frames_2 = f2.unsqueeze(0)
            return frames_1, frames_2
