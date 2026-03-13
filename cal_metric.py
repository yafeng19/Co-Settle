import os
import glob
import torch
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F

from models.mae import models_mae as mae_model
from models.ijepa.deit import deit_base as ijepa_model
from models.mocov3.vit_moco import vit_base as mocov3_model
from models.clip import clip as clip_model
from models.blip.vit import VisionTransformer as blip_model
from models.blip.blip import load_checkpoint as blip_load_checkpoint
from models.dino import vision_transformer as dino_model
from models.dino import utils as dino_utils
from models.ibot import models as ibot_model
from models.ibot import utils as ibot_utils
from models.dinov2.dinov2.eval.setup import setup_and_build_model as dinov2_model



EPS = 1e-6

class VideoFrameDataset(Dataset):
    def __init__(self, sample_csv_path, transform=None, target_frame_count=40):
        self.transform = transform
        self.target_frame_count = target_frame_count
        
        df = pd.read_csv(sample_csv_path, header=None, names=['video_path'])
        self.video_paths = df['video_path'].tolist()
        
        print(f"Loaded {len(self.video_paths)} video paths from {sample_csv_path}")
        
        valid_video_paths = []
        for video_path in tqdm(self.video_paths, desc="Validating video paths and frame counts"):
            if os.path.exists(video_path):
                frame_files = [
                    f for f in os.listdir(video_path) 
                    if f.endswith(('.jpg', '.png', '.jpeg'))
                ]
                if len(frame_files) == target_frame_count:
                    valid_video_paths.append(video_path)
                else:
                    print(f"Skipping {video_path}: has {len(frame_files)} frames, expected {target_frame_count}")
            else:
                print(f"Skipping {video_path}: path does not exist")
        
        self.video_paths = valid_video_paths
        print(f"Found {len(self.video_paths)} valid videos with exactly {target_frame_count} frames")
        
        if len(self.video_paths) == 0:
            raise ValueError(f"No valid videos found with exactly {target_frame_count} frames")
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frame_files = sorted([
            f for f in os.listdir(video_path) 
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])
        
        if len(frame_files) != self.target_frame_count:
            raise ValueError(f"Video {video_path} has {len(frame_files)} frames, expected {self.target_frame_count}")
        
        frames = []
        for frame_file in frame_files:
            img_path = os.path.join(video_path, frame_file)
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        return torch.stack(frames)  # (40, C, H, W)

def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def build_dataloader(sample_csv_path, batch_size=16, target_frame_count=40):
    transform = get_transform()
    dataset = VideoFrameDataset(
        sample_csv_path, 
        transform=transform, 
        target_frame_count=target_frame_count
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=8,  # Increase number of workers
        pin_memory=True,
        prefetch_factor=2,  # Prefetch factor
        persistent_workers=True  # Persistent workers
    )
    return dataloader



def affinity(x1, x2):
    ndim = x1.ndim
    if ndim < 4:  # add time dimension
        x1, x2 = x1.unsqueeze(1), x2.unsqueeze(1)
    elif ndim == 5:  # handle batch dimension [B, N, T, P, D]
        x1 = x1.permute(0, 1, 4, 2, 3)  # [B, N, D, T, P]
        x2 = x2.permute(0, 1, 4, 2, 3)  # [B, N, D, T, P]
        A = torch.einsum('bndti,bndtj->bntij', x1, x2)
        return A
    x1 = x1.permute(0, 3, 1, 2)
    x2 = x2.permute(0, 3, 1, 2)
    A = torch.einsum('bctn,bctm->btnm', x1, x2)
    return A


def stoch_mat(A, temp=0.03):
    return F.softmax(A/temp, dim=-1)


def compute_effective_rank(model, dataloader, device='cuda'):
    model.eval()
    all_representative_features = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting representative frame features"):
            batch = batch.to(device, non_blocking=True)  # [B, T, C, H, W]
            B, T, C, H, W = batch.shape
            
            # Select middle frame as representative frame
            mid_frame_idx = T // 2
            representative_frames = batch[:, mid_frame_idx, :, :, :]  # [B, C, H, W]
            
            if args.base_model == 'clip':
                feats = model.encode_image(representative_frames, use_adapter=args.use_adapter, is_training=False, eval_metric=True)
            else:
                feats = model(representative_frames, use_adapter=args.use_adapter, is_training=False, eval_metric=True)  # [B, P, D]
            
            all_representative_features.append(feats)  # Keep on GPU

    # Stack all representative frame features on GPU [N, P, D]
    representative_features = torch.cat(all_representative_features, dim=0)  # [N, P, D]
    representative_features = representative_features.view(-1, representative_features.shape[-1])  # [N*P, D]
    N, d = representative_features.shape
    # Effective rank
    singular_values = torch.linalg.svdvals(representative_features)
    p = singular_values / singular_values.sum()
    # Avoid log(0) by adding a small epsilon
    epsilon = 1e-12
    p = p + epsilon
    # Compute the Shannon entropy
    H = -torch.sum(p * torch.log(p))
    # Compute the effective rank
    rank = torch.exp(H)
    rank = rank / min(N, d)
    return rank.item()


def compute_intra_video_metric(model, dataloader, device='cuda'):
    """Compute intra-video distance: original distance and normalized distance using different percentile sample space diameters"""
    model.eval()
    all_intra_distances = []
    all_normalized_distances = {90: [], 95: [], 100: []}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing intra-video distances"):
            batch = batch.to(device, non_blocking=True)  # [B, T, C, H, W]
            B, T, C, H, W = batch.shape

            # Batch process all frames of all videos: [B*T, C, H, W]
            all_frames = batch.view(B * T, C, H, W)
            
            if args.base_model == 'clip':
                all_feats = model.encode_image(all_frames, use_adapter=args.use_adapter, is_training=False, eval_metric=True)
            else:
                all_feats = model(all_frames, use_adapter=args.use_adapter, is_training=False, eval_metric=True)  # [B*T, P, D]

            # Reshape to [B, T, P, D]
            P, D = all_feats.shape[1], all_feats.shape[2]
            all_feats = all_feats.view(B, T, P, D)
            
            # Exclude first and last 3 frames, get valid frame range [B, T-6, P, D]
            valid_T = T - 6
            if valid_T <= 1:
                continue
            valid_feats = all_feats[:, 3:T-3, :, :]  # [B, T-6, P, D]
            
            # === Vectorized processing: batch compute patch distances for all adjacent frame pairs ===
            # Current frame and next frame features, process up to second-to-last frame [B, T-7, P, D]
            curr_frames = valid_feats[:, :-1, :, :]  # [B, T-7, P, D]
            next_frames = valid_feats[:, 1:, :, :]   # [B, T-7, P, D]
            
            # Batch compute distance matrix: [B, T-7, P, P] using torch.cdist
            # Reshape to [B*(T-7), P, D] for cdist
            curr_reshaped = curr_frames.reshape(B * (valid_T - 1), P, D)  # [B*(T-7), P, D]
            next_reshaped = next_frames.reshape(B * (valid_T - 1), P, D)  # [B*(T-7), P, D]
            
            # Use cdist to compute Euclidean distance [B*(T-7), P, P]
            dist_matrices = torch.cdist(curr_reshaped, next_reshaped, p=2)  # [B*(T-7), P, P]
            
            # Reshape back to [B, T-7, P, P]
            dist_matrices = dist_matrices.view(B, valid_T - 1, P, P)
            
            # Find minimum distance from each patch in first frame to second frame [B, T-7, P]
            min_distances, _ = torch.min(dist_matrices, dim=-1)  # [B, T-7, P]
            
            # Select 50% of patches with smallest distances [B, T-7, P//2]
            top_k = int(P * 0.5)
            selected_distances, _ = torch.topk(min_distances, top_k, largest=False, dim=-1)  # [B, T-7, P//2]
            
            # Compute average distance for each adjacent frame pair [B, T-7]
            frame_pair_distances = selected_distances.mean(dim=-1)  # [B, T-7]
            
            # Compute original average distance for each video
            video_intra_distances = frame_pair_distances.mean(dim=1)  # [B]
            all_intra_distances.append(video_intra_distances)
            
            # === Compute normalized distances ===
            # Compute patch center and diameter for each video for normalization
            video_patches = all_feats.reshape(B, T * P, D)  # [B, T*P, D]
            video_centers = video_patches.mean(dim=1)  # [B, D]
            centers_expanded = video_centers.unsqueeze(1)  # [B, 1, D]
            distances_to_center = torch.norm(video_patches - centers_expanded, p=2, dim=-1)  # [B, T*P]
            
            # Compute diameters of spaces formed by different percentile samples
            for p in [90, 95, 100]:
                if p == 100:
                    batch_radii = distances_to_center.max(dim=1)[0]  # [B]
                else:
                    batch_radii = torch.quantile(distances_to_center, p/100.0, dim=1)  # [B]
                
                batch_diameters = batch_radii * 2  # [B]
                
                # Normalize original distances by diameter
                normalized_distances = video_intra_distances / (batch_diameters + 1e-8)  # [B]
                all_normalized_distances[p].append(normalized_distances)

    if not all_intra_distances:
        return {
            'original': 0.0,
            'normalized_90': 0.0,
            'normalized_95': 0.0,
            'normalized_100': 0.0
        }
    
    # Compute final averages
    original_distance = torch.cat(all_intra_distances, dim=0).mean().item()
    normalized_distances = {}
    for p in [90, 95, 100]:
        normalized_distances[f'normalized_{p}'] = torch.cat(all_normalized_distances[p], dim=0).mean().item()
    
    result = {'original': original_distance}
    result.update(normalized_distances)
    return result


def compute_inter_video_metric(model, dataloader, device='cuda'):
    """Compute inter-video distance: original distance and normalized distance using different percentile sample space diameters"""
    model.eval()
    all_representative_features = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting representative frame features"):
            batch = batch.to(device, non_blocking=True)  # [B, T, C, H, W]
            B, T, C, H, W = batch.shape
            
            # Select middle frame as representative frame
            mid_frame_idx = T // 2
            representative_frames = batch[:, mid_frame_idx, :, :, :]  # [B, C, H, W]
            
            if args.base_model == 'clip':
                feats = model.encode_image(representative_frames, use_adapter=args.use_adapter, is_training=False, eval_metric=True)
            else:
                feats = model(representative_frames, use_adapter=args.use_adapter, is_training=False, eval_metric=True)  # [B, P, D]
            
            all_representative_features.append(feats)  # Keep on GPU

    # Stack all representative frame features on GPU [N, P, D]
    representative_features = torch.cat(all_representative_features, dim=0)  # [N, P, D]
    N, P, D = representative_features.shape
    
    # Randomly sample 8 patches to reduce memory usage
    torch.manual_seed(42)  # Ensure reproducibility
    sample_size = min(8, P)
    if P > sample_size:
        # Randomly select patch indices
        patch_indices = torch.randperm(P, device=device)[:sample_size]
        representative_features = representative_features[:, patch_indices, :]  # [N, sample_size, D]
        print(f"Randomly sampled {sample_size} patches from {P} total patches")
    else:
        print(f"Using all {P} patches (less than or equal to 8)")
    
    # Update P to sampled patch count
    P = representative_features.shape[1]
    
    # Batch compute corresponding position patch distances for all video pairs
    # Expand dimensions for broadcasting [N, 1, P, D] and [1, N, P, D]
    features_expanded_1 = representative_features.unsqueeze(1)  # [N, 1, P, D]
    features_expanded_2 = representative_features.unsqueeze(0)  # [1, N, P, D]
    
    # Compute Euclidean distance [N, N, P]
    patch_distances = torch.norm(features_expanded_1 - features_expanded_2, p=2, dim=-1)  # [N, N, P]
    
    # For each video pair, compute average of all patch distances [N, N]
    video_distances = patch_distances.mean(dim=-1)  # [N, N]
    
    # Use CUDA-accelerated upper triangular matrix extraction
    triu_mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
    valid_distances = video_distances[triu_mask]  # [N*(N-1)/2]
    print(f"Number of video pairs: {len(valid_distances)}")
    
    # Compute original distance
    original_distance = valid_distances.mean().item()
    
    # === Compute normalized distances ===
    # Compute global patch center and diameters of spaces formed by different percentile samples
    all_patches = representative_features.view(-1, D)  # [N*P, D]
    global_center = all_patches.mean(dim=0)  # [D]
    distances_to_center = torch.norm(all_patches - global_center.unsqueeze(0), p=2, dim=-1)  # [N*P]
    
    normalized_distances = {}
    for p in [90, 95, 100]:
        if p == 100:
            # Use diameter of space formed by all samples
            radius = distances_to_center.max().item()
        else:
            # Use diameter of space formed by p% samples
            # Select p% samples closest to center
            num_samples = int(len(distances_to_center) * p / 100.0)
            sorted_distances, _ = torch.sort(distances_to_center)
            selected_distances = sorted_distances[:num_samples]  # Select closest p% samples
            radius = selected_distances.max().item()  # Farthest distance from center among these samples as radius
        
        diameter = radius * 2
        normalized_distances[f'normalized_{p}'] = original_distance / (diameter + 1e-8)
    
    result = {'original': original_distance}
    result.update(normalized_distances)
    return result


def compute_intra_video_diameter(model, dataloader, device='cuda', percentiles=[90, 95, 100]):
    """Compute intra-video diameter: center of all patches in each video, compute max distance from different percentile patches to center multiplied by 2"""
    model.eval()
    all_diameters = {p: [] for p in percentiles}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing intra-video diameters"):
            batch = batch.to(device, non_blocking=True)  # [B, T, C, H, W]
            B, T, C, H, W = batch.shape

            # Batch process all frames of all videos: [B*T, C, H, W]
            all_frames = batch.view(B * T, C, H, W)
            
            if args.base_model == 'clip':
                all_feats = model.encode_image(all_frames, use_adapter=args.use_adapter, is_training=False, eval_metric=True)
            elif args.base_model == 'cropmae' or args.base_model == 'siammae' or args.base_model == 'rsp' \
                or args.base_model == 'st_adapter' or args.base_model == 'AIM' or args.base_model == 'ZeroI2V':
                all_feats = model(all_frames)  # [B*T, P, D]
            else:
                all_feats = model(all_frames, use_adapter=args.use_adapter, is_training=False, eval_metric=True)  # [B*T, P, D]

            # Reshape to [B, T, P, D] then flatten to [B, T*P, D]
            P, D = all_feats.shape[1], all_feats.shape[2]
            all_feats = all_feats.view(B, T, P, D)
            video_patches = all_feats.view(B, T * P, D)  # [B, T*P, D]
            
            # Batch compute patch center for each video [B, D]
            video_centers = video_patches.mean(dim=1)  # [B, D]
            
            # Batch compute distance from each patch to its video center
            # Expand center dimension: [B, 1, D]
            centers_expanded = video_centers.unsqueeze(1)  # [B, 1, D]
            
            # Compute distances [B, T*P]
            distances_to_center = torch.norm(video_patches - centers_expanded, p=2, dim=-1)  # [B, T*P]
            
            # Batch compute radii for different percentiles, then multiply by 2 to get diameter
            for p in percentiles:
                if p == 100:
                    # Compute max distance (radius) for each video
                    batch_radii = distances_to_center.max(dim=1)[0]  # [B]
                else:
                    # Compute p% quantile radius for each video
                    batch_radii = torch.quantile(distances_to_center, p/100.0, dim=1)  # [B]
                
                # Multiply radius by 2 to get diameter
                batch_diameters = batch_radii * 2
                
                # Add results to list
                all_diameters[p].extend(batch_diameters.cpu().numpy().tolist())

    # Compute average diameters
    avg_diameters = {}
    for p in percentiles:
        if all_diameters[p]:
            avg_diameters[p] = float(np.mean(all_diameters[p]))
        else:
            avg_diameters[p] = 0.0
    
    return avg_diameters


def compute_inter_video_diameter(model, dataloader, device='cuda', percentiles=[90, 95, 100]):
    """Compute inter-video diameter: center of all video representative frame patches, compute max distance from different percentile patches to center multiplied by 2"""
    model.eval()
    all_representative_features = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting representative frame features for diameter"):
            batch = batch.to(device, non_blocking=True)  # [B, T, C, H, W]
            B, T, C, H, W = batch.shape
            
            # Select middle frame as representative frame
            mid_frame_idx = T // 2
            representative_frames = batch[:, mid_frame_idx, :, :, :]  # [B, C, H, W]
            
            if args.base_model == 'clip':
                feats = model.encode_image(representative_frames, use_adapter=args.use_adapter, is_training=False, eval_metric=True)
            else:
                feats = model(representative_frames, use_adapter=args.use_adapter, is_training=False, eval_metric=True)  # [B, P, D]
            
            all_representative_features.append(feats)

    # Stack all representative frame features on GPU and flatten [N*P, D]
    representative_features = torch.cat(all_representative_features, dim=0)  # [N, P, D]
    all_patches = representative_features.view(-1, representative_features.shape[-1])  # [N*P, D]
    
    # Compute center of all patches [D]
    global_center = all_patches.mean(dim=0)  # [D]
    
    # Compute distance from each patch to global center [N*P]
    distances_to_center = torch.norm(all_patches - global_center.unsqueeze(0), p=2, dim=-1)  # [N*P]
    
    # Compute radii for different percentiles, then multiply by 2 to get diameter
    diameters = {}
    for p in percentiles:
        if p == 100:
            radius = distances_to_center.max().item()
        else:
            radius = torch.quantile(distances_to_center, p/100.0).item()
        
        # Multiply radius by 2 to get diameter
        diameters[p] = radius * 2
    
    return diameters


def compute_cycle_accuracy(model, dataloader, device='cuda'):
    model.eval()
    acc_list = []

    time_pairs = torch.tensor([
        [0.25, 0.4],
        [0.35, 0.5],
        [0.45, 0.6],
        [0.55, 0.7],
        [0.65, 0.8],
    ], dtype=torch.float32)  # [N, 2]
    num_pairs = time_pairs.shape[0]
    num_frames_in_cycle = 3

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            batch = batch.to(device)  # [B, T, C, H, W]
            B, T, C, H, W = batch.shape

            # === Step 1: Vectorized extraction of 3 frames: f1, f2, f1 (cycle)
            t_idx = (time_pairs * T).long().clamp(max=T-1)  # [N, 2]
            f1_idx = t_idx[:, 0]
            f2_idx = t_idx[:, 1]
            cycle_idx = torch.stack([f1_idx, f2_idx, f1_idx], dim=1).view(-1)  # [3N]

            # Extract frames for all batches: [B, 3N, C, H, W]
            frames = batch[:, cycle_idx]  # [B, 3N, C, H, W]
            frames = frames.view(B * 3 * num_pairs, C, H, W)  # [B*3N, C, H, W]

            # === Step 2: Model encode all frames
            if args.base_model == 'clip':
                feats = model.encode_image(frames, use_adapter=args.use_adapter, is_training=False, eval_metric=True)  # (B*3N, P, D)
            else:
                feats = model(frames, use_adapter=args.use_adapter, is_training=False, eval_metric=True)  # (B*3N, P, D)

            # === Step 3: reshape to [B, N, 3, P, D]
            P, D = feats.shape[1], feats.shape[2]
            feats = feats.view(B, num_pairs, num_frames_in_cycle, P, D)  # [B, N, 3, P, D]

            # === Step 4: patch pooling, become 49 patches
            S = int(P**0.5)
            if P == S*S or P == S*S + 1:
                if P == S*S + 1:
                    feats = feats[:, :, :, 1:, :]  # remove CLS
                S = int((feats.shape[3])**0.5)
                feats = feats.view(B, num_pairs, num_frames_in_cycle, S, S, D).permute(0, 1, 2, 5, 3, 4)  # [B, N, 3, D, S, S]
                feats = F.avg_pool2d(feats.flatten(0, 2), kernel_size=2, stride=2)  # [B*N*3, D, 7, 7]
                feats = feats.flatten(2).transpose(1, 2).view(B, num_pairs, num_frames_in_cycle, -1, D)  # [B, N, 3, 49, D]
            else:
                raise NotImplementedError("Patch number not match")

            # === Step 5: Compute A12 and A21 (similarity)
            As = affinity(feats[:, :, :-1], feats[:, :, 1:])  # [B, N, 2, 49, 49]
            forward_sim = [stoch_mat(As[:, :, i]) for i in range(num_frames_in_cycle-1)]  # [B, N, 49, 49]
            
            # === Step 6: Construct random walk matrix
            forward_chain = forward_sim
            mat_mul_forward = forward_chain[0]
            for mat in forward_chain[1:]:
                mat_mul_forward = mat_mul_forward @ mat

            S_cycle = mat_mul_forward  # [B, N, 49, 49]

            # === Step 7: Compute cycle accuracy
            logit = torch.log(S_cycle + EPS).view(-1, S_cycle.shape[-1])  # [B*N*P, P]
            targets = torch.arange(S_cycle.shape[-1], device=logit.device).repeat(B * num_pairs)  # [B*N*P]
            acc = (logit.argmax(dim=-1) == targets).float().mean()
            acc_list.append(acc.item())

    return float(np.mean(acc_list))


def merge_all_csv_results(output_dir):
    
    # Find all matching CSV files
    csv_pattern = os.path.join(output_dir, "video_metrics_*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print("No CSV files found to merge.")
        return
    
    print(f"Found {len(csv_files)} CSV files to merge:")
    for csv_file in csv_files:
        print(f"  - {os.path.basename(csv_file)}")
    
    all_results = []
    
    for csv_file in csv_files:
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Extract model name from file name
            filename = os.path.basename(csv_file)
            if filename.startswith("video_metrics_") and filename.endswith(".csv"):
                model_name = filename[14:-4]  # Remove "video_metrics_" prefix and ".csv" suffix
                
                # Add model name column to DataFrame at the front
                df.insert(0, 'model_name', model_name)
                
                all_results.append(df)
                print(f"  Loaded {model_name}: {len(df)} rows")
            else:
                print(f"  Skipped {filename}: doesn't match expected pattern")
                
        except Exception as e:
            print(f"  Error loading {csv_file}: {e}")
    
    if not all_results:
        print("No valid CSV files were loaded.")
        return
    
    # Merge all DataFrames
    merged_df = pd.concat(all_results, ignore_index=True)
    
    # Sort by model name
    merged_df = merged_df.sort_values('model_name')
    
    # Save merged results
    merged_csv_path = os.path.join(output_dir, "merged_video_metrics.csv")
    merged_df.to_csv(merged_csv_path, index=False)
    
    print(f"\n{'='*50}")
    print(f"Merged Results Summary:")
    print(f"- Total models: {len(all_results)}")
    print(f"- Total rows: {len(merged_df)}")
    print(f"- Saved to: {merged_csv_path}")
    print(f"{'='*50}\n")
    
    # Display merged data preview
    print("\nMerged data preview:")
    print(merged_df.to_string(index=False))
    
    return merged_csv_path


def main_eval(args):
    # Set global random seed to ensure reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    import random
    random.seed(42)
    
    # Set CUDA deterministic behavior (optional, may affect performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Initialize model, only support loading ViT-B/16 checkpoints for now, can add more if needed
    if args.base_model == 'mae':
        model = mae_model.__dict__['mae_vit_base_patch16'](fusion_net=args.fusion_net)
        state_dict = torch.load(args.resume, map_location="cpu")
        msg = model.load_state_dict(state_dict, strict=False)
    
    elif args.base_model == 'ijepa':
        model = ijepa_model()
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        if args.use_adapter:
            pretrained_dict = checkpoint['model']
        else:
            pretrained_dict = checkpoint['encoder']
        pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
        msg = model.load_state_dict(pretrained_dict, strict=False)
        print(msg)
    
    elif args.base_model == 'mocov3':
        model = mocov3_model(fusion_net=args.fusion_net)
        state_dict = torch.load(args.resume, map_location="cpu")
        if args.use_adapter:
            state_dict = state_dict['model']
        else:
            state_dict = state_dict["state_dict"]
            state_dict = {k: v for k, v in state_dict.items() if k.startswith('module.momentum_encoder.')}
            state_dict = {k.replace("module.momentum_encoder.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)

    elif args.base_model == 'clip':
        model, preprocess = clip_model.load_eval(args.resume, device=args.device)
        model = model.to(args.device)

    elif args.base_model == 'blip':
        model = blip_model(patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                        use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0)  
        model = model.to(args.device)
        model, msg = blip_load_checkpoint(model, args.resume)
        print(msg)
   
    elif args.base_model == 'dino':
        model = dino_model.__dict__['vit_base'](patch_size=16, num_classes=0)
        model = model.to(args.device)
        dino_utils.load_pretrained_weights(model, args.resume, 'model', 'vit_base', 16)

    elif args.base_model == 'ibot':
        model = ibot_model.__dict__['vit_base'](
            patch_size=16, 
            return_all_tokens=True,
            num_classes=0,
            use_mean_pooling=False) # no need for mean pooling
        model = model.to(args.device)
        if args.use_adapter:
            ibot_utils.load_pretrained_weights(model, args.resume, 'model', 'vit_base', 16)
        else:
            ibot_utils.load_pretrained_weights(model, args.resume, 'state_dict', 'vit_base', 16)

    elif args.base_model == 'dinov2':
        model, autocast_dtype = dinov2_model(args, is_Training=False)
        model = model.to(args.device)
        
    else:
        return NotImplementedError("args.base_model {} is not supported.".format(args.base_model))
    

    model.eval()
    model = model.to(args.device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Build DataLoader using video paths from CSV file
    dataloader = build_dataloader(
        args.sample_csv_path, 
        batch_size=32,
        target_frame_count=40
    )
    
    print(f"Dataset size: {len(dataloader.dataset)} videos")
    print(f"Number of batches: {len(dataloader)}")
    print(f"Each video has exactly 40 frames")
    
    # Compute metrics - warm up GPU
    torch.cuda.empty_cache()  # Clear GPU cache
    
    # Use CUDA Events for precise timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    with torch.no_grad():
        inter_results = compute_inter_video_metric(model, dataloader, args.device)
        intra_results = compute_intra_video_metric(model, dataloader, args.device)
        effective_rank = compute_effective_rank(model, dataloader, args.device)
        cyc_acc = compute_cycle_accuracy(model, dataloader, args.device)
        
        # Compute diameters (for reference)
        intra_diameters = compute_intra_video_diameter(model, dataloader, args.device)
        inter_diameters = compute_inter_video_diameter(model, dataloader, args.device)
        
    end_event.record()
    
    torch.cuda.synchronize()  # Wait for all CUDA operations to complete
    elapsed = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
    
    # Print global results
    print(f"\n{'='*40}")
    print(f"Global Video Metric Results:")
    # print(f"- Cycle Accuracy: {cyc_acc:.4f}")

    print(f"- Effective Rank: {effective_rank:.4f}")
    
    print(f"\nInter-video Distances:")
    print(f"- Original: {inter_results['original']:.4f}")
    print(f"- Normalized by 90% diameter: {inter_results['normalized_90']:.4f}")
    print(f"- Normalized by 95% diameter: {inter_results['normalized_95']:.4f}")
    print(f"- Normalized by 100% diameter: {inter_results['normalized_100']:.4f}")
    
    print(f"\nIntra-video Distances:")
    print(f"- Original: {intra_results['original']:.4f}")
    print(f"- Normalized by 90% diameter: {intra_results['normalized_90']:.4f}")
    print(f"- Normalized by 95% diameter: {intra_results['normalized_95']:.4f}")
    print(f"- Normalized by 100% diameter: {intra_results['normalized_100']:.4f}")
    
    print(f"\nInter/Intra ratios:")
    print(f"- Original: {inter_results['original']/intra_results['original']:.4f}")
    print(f"- 90% normalized: {inter_results['normalized_90']/intra_results['normalized_90']:.4f}")
    print(f"- 95% normalized: {inter_results['normalized_95']/intra_results['normalized_95']:.4f}")
    print(f"- 100% normalized: {inter_results['normalized_100']/intra_results['normalized_100']:.4f}")
    
    print(f"\nTime elapsed: {elapsed:.2f}s")
    print(f"{'='*40}\n")
    
    # Save results to CSV
    os.makedirs(args.output_dir, exist_ok=True)
    if args.use_adapter:
        csv_path = os.path.join(args.output_dir, f"video_metrics_{args.base_model}+ours.csv")
    else:
        csv_path = os.path.join(args.output_dir, f"video_metrics_{args.base_model}.csv")
    
    # Global results
    result_dict = {
        'model': args.base_model,
        'cyc_acc': cyc_acc,
        'effective_rank': effective_rank,
        'ori_inter_distance': inter_results['original'],
        '100_inter_distance': inter_results['normalized_100'],
        '95_inter_distance': inter_results['normalized_95'],
        '90_inter_distance': inter_results['normalized_90'],
        'ori_intra_distance': intra_results['original'],
        '100_intra_distance': intra_results['normalized_100'],
        '95_intra_distance': intra_results['normalized_95'],
        '90_intra_distance': intra_results['normalized_90']
    }
    
    global_df = pd.DataFrame([result_dict])
    
    with open(csv_path, 'w') as f:
        global_df.to_csv(f, index=False)
    
    print(f"Results saved to: {csv_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_csv_path", type=str, required=True, help="Path to CSV file containing video paths")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save CSV results")
    parser.add_argument('--base_model', default='mae', type=str, choices=['mae', 'ijepa', 'clip', 'blip', 'mocov3', 'dino', 'ibot', 'dinov2'], help='basic model')
    parser.add_argument('--use_adapter', action='store_true')
    parser.add_argument('--fusion_net', default='linear', type=str, choices=['linear', 'mlp2', 'mlp3'], help='network for fusing representation and correlation')
    parser.add_argument('--generate_pseudo_label', action='store_true')
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--model_type", type=str, default="vitb16", help="")
    parser.add_argument('--resume', default='checkpoints/MAE.pth', type=str, help='Path to the model checkpoint')
    
    parser.add_argument("--config_file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("opts", 
        help="""
            Modify config options at the end of the command. For Yacs configs, use
            space-separated "PATH.KEY VALUE" pairs.
            For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=[],
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    main_eval(args)
    print(f"\nMerging all CSV files in {args.output_dir}...")
    merge_all_csv_results(args.output_dir)