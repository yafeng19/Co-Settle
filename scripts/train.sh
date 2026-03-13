export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=4,5,6,7


python -m torch.distributed.launch --nproc_per_node=4 --rdzv_endpoint=localhost:25925 main_pretrain.py \
    --batch_size 128 \
    --world_size 4 \
    --model_type vitb16 \
    --epochs 5 \
    --save_fre 4 \
    --warmup_epochs 1 \
    --blr 1e-4 \
    --weight_decay 0.05 \
    --dataset Kinetics-400 \
    --data_path ./data/k400_valid_files.txt \
    --output_dir ./experiments/mae_vitb16+ours \
    --log_dir ./experiments/mae_vitb16+ours \
    --resume ./ckpts/ori/MAE_INet_vitb16_800ep/checkpoint.pth \
    --base_model mae \
    --use_adapter \
    --fusion_net linear \
    --clip_gap 0.15 \
    --temp 0.03 \
    --lambda_reg 3 \
    --use_asy_pos_embedding \
    --asy_pos_ratio 10


python -m torch.distributed.launch --nproc_per_node=4 --rdzv_endpoint=localhost:25925 main_pretrain.py \
    --batch_size 128 \
    --world_size 4 \
    --model_type vitb16 \
    --epochs 5 \
    --save_fre 4 \
    --warmup_epochs 1 \
    --blr 1e-4 \
    --weight_decay 0.05 \
    --dataset Kinetics-400 \
    --data_path ./data/k400_valid_files.txt \
    --output_dir ./experiments/ijepa_vitb16+ours \
    --log_dir ./experiments/ijepa_vitb16+ours \
    --resume ./ckpts/ori/IJEPA_INet_vitb16_800ep/ijepa-ep800.pth.tar \
    --base_model ijepa \
    --use_adapter \
    --fusion_net linear \
    --clip_gap 0.15 \
    --temp 0.03 \
    --lambda_reg 3 \
    --use_asy_pos_embedding \
    --asy_pos_ratio 10


python -m torch.distributed.launch --nproc_per_node=4 --rdzv_endpoint=localhost:25925 main_pretrain.py \
    --batch_size 128 \
    --world_size 4 \
    --model_type vitb16 \
    --epochs 5 \
    --save_fre 4 \
    --warmup_epochs 1 \
    --blr 1e-4 \
    --weight_decay 0.05 \
    --dataset Kinetics-400 \
    --data_path ./data/k400_valid_files.txt \
    --output_dir ./experiments/mocov3_vitb16+ours \
    --log_dir ./experiments/mocov3_vitb16+ours \
    --resume ./ckpts/ori/mocov3_INet_vitb16_300ep/vit-b-300ep.pth.tar \
    --base_model mocov3 \
    --use_adapter \
    --fusion_net linear \
    --clip_gap 0.15 \
    --temp 0.03 \
    --lambda_reg 3 \
    --use_asy_pos_embedding \
    --asy_pos_ratio 10


python -m torch.distributed.launch --nproc_per_node=4 --rdzv_endpoint=localhost:25925 main_pretrain.py \
    --batch_size 128 \
    --world_size 4 \
    --model_type vitb16 \
    --epochs 5 \
    --save_fre 4 \
    --warmup_epochs 1 \
    --blr 1e-4 \
    --weight_decay 0.05 \
    --dataset Kinetics-400 \
    --data_path ./data/k400_valid_files.txt \
    --output_dir ./experiments/clip_vitb16+ours \
    --log_dir ./experiments/clip_vitb16+ours \
    --resume ./ckpts/ori/CLIP_vitb16/ViT-B-16.pt \
    --base_model clip \
    --use_adapter \
    --fusion_net linear \
    --clip_gap 0.15 \
    --temp 0.03 \
    --lambda_reg 3 \
    --use_asy_pos_embedding \
    --asy_pos_ratio 10


python -m torch.distributed.launch --nproc_per_node=4 --rdzv_endpoint=localhost:25925 main_pretrain.py \
    --batch_size 128 \
    --world_size 4 \
    --model_type vitb16 \
    --epochs 5 \
    --save_fre 4 \
    --warmup_epochs 1 \
    --blr 1e-4 \
    --weight_decay 0.05 \
    --dataset Kinetics-400 \
    --data_path ./data/k400_valid_files.txt \
    --output_dir ./experiments/blip_vitb16+ours \
    --log_dir ./experiments/blip_vitb16+ours \
    --resume ./ckpts/ori/BLIP_vitb16/model_base_retrieval_coco.pth \
    --base_model blip \
    --use_adapter \
    --fusion_net linear \
    --clip_gap 0.15 \
    --temp 0.03 \
    --lambda_reg 3 \
    --use_asy_pos_embedding \
    --asy_pos_ratio 10


python -m torch.distributed.launch --nproc_per_node=4 --rdzv_endpoint=localhost:25925 main_pretrain.py \
    --batch_size 128 \
    --world_size 4 \
    --model_type vitb16 \
    --epochs 5 \
    --save_fre 4 \
    --warmup_epochs 1 \
    --blr 1e-4 \
    --weight_decay 0.05 \
    --dataset Kinetics-400 \
    --data_path ./data/k400_valid_files.txt \
    --output_dir ./experiments/dino_vitb16+ours \
    --log_dir ./experiments/dino_vitb16+ours \
    --resume ./ckpts/ori/DINO_vitb16/dino_vitbase16_pretrain.pth \
    --base_model dino \
    --use_adapter \
    --fusion_net linear \
    --clip_gap 0.15 \
    --temp 0.03 \
    --lambda_reg 3 \
    --use_asy_pos_embedding \
    --asy_pos_ratio 10


python -m torch.distributed.launch --nproc_per_node=4 --rdzv_endpoint=localhost:25925 main_pretrain.py \
    --batch_size 128 \
    --world_size 4 \
    --model_type vitb16 \
    --epochs 5 \
    --save_fre 4 \
    --warmup_epochs 1 \
    --blr 1e-4 \
    --weight_decay 0.05 \
    --dataset Kinetics-400 \
    --data_path ./data/k400_valid_files.txt \
    --output_dir ./experiments/ibot_vitb16+ours \
    --log_dir ./experiments/ibot_vitb16+ours \
    --resume ./ckpts/ori/iBot_INet_vitb16_400ep/checkpoint_teacher_vitb-16.pth \
    --base_model ibot \
    --use_adapter \
    --fusion_net linear \
    --clip_gap 0.15 \
    --temp 0.03 \
    --lambda_reg 3 \
    --use_asy_pos_embedding \
    --asy_pos_ratio 10



python -m torch.distributed.launch --nproc_per_node=4 --rdzv_endpoint=localhost:25925 main_pretrain.py \
    --batch_size 128 \
    --world_size 4 \
    --model_type vitb16 \
    --epochs 5 \
    --save_fre 4 \
    --warmup_epochs 1 \
    --blr 1e-4 \
    --weight_decay 0.05 \
    --dataset Kinetics-400 \
    --data_path ./data/k400_valid_files.txt \
    --output_dir ./experiments/dinov2_vitb16+ours \
    --log_dir ./experiments/dinov2_vitb16+ours \
    --resume ./ckpts/ori/DINOv2_INet_vitb16_100ep/checkpoint.pth \
    --config_file=./models/dinov2/dinov2/configs/train/vitb16_short.yaml \
    --base_model dinov2 \
    --use_adapter \
    --fusion_net linear \
    --clip_gap 0.15 \
    --temp 0.03 \
    --lambda_reg 3 \
    --use_asy_pos_embedding \
    --asy_pos_ratio 10