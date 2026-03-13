
export CUDA_VISIBLE_DEVICES=0


python3 -m cal_metric \
    --sample_csv_path=./sampling/val_frames_1000_sample.csv \
    --output_dir=./eval_metric/ours \
    --model_type=vitb16 \
    --resume=./ckpts/ours/mae+ours/checkpoint.pth \
    --base_model=mae \
    --use_adapter


python3 -m cal_metric \
    --sample_csv_path=./sampling/val_frames_1000_sample.csv \
    --output_dir=./eval_metric/ours \
    --model_type=vitb16 \
    --resume=./ckpts/ours/ijepa+ours/checkpoint.pth \
    --base_model=ijepa \
    --use_adapter


python3 -m cal_metric \
    --sample_csv_path=./sampling/val_frames_1000_sample.csv \
    --output_dir=./eval_metric/ours \
    --model_type=vitb16 \
    --resume=./ckpts/ours/clip+ours/checkpoint.pth \
    --base_model=clip \
    --use_adapter


python3 -m cal_metric \
    --sample_csv_path=./sampling/val_frames_1000_sample.csv \
    --output_dir=./eval_metric/ours \
    --model_type=vitb16 \
    --resume=./ckpts/ours/blip+ours/checkpoint.pth \
    --base_model=blip \
    --use_adapter


python3 -m cal_metric \
    --sample_csv_path=./sampling/val_frames_1000_sample.csv \
    --output_dir=./eval_metric/ours \
    --model_type=vitb16 \
    --resume=./ckpts/ours/dino+ours/checkpoint.pth \
    --base_model=dino \
    --use_adapter


python3 -m cal_metric \
    --sample_csv_path=./sampling/val_frames_1000_sample.csv \
    --output_dir=./eval_metric/ours \
    --model_type=vitb16 \
    --resume=./ckpts/ours/mocov3+ours/checkpoint.pth \
    --base_model=mocov3 \
    --use_adapter


python3 -m cal_metric \
    --sample_csv_path=./sampling/val_frames_1000_sample.csv \
    --output_dir=./eval_metric/ours \
    --model_type=vitb16 \
    --resume=./ckpts/ours/ibot+ours/checkpoint.pth \
    --base_model=ibot \
    --use_adapter


python3 -m cal_metric \
    --sample_csv_path=./sampling/val_frames_1000_sample.csv \
    --output_dir=./eval_metric/ours \
    --model_type=vitb16 \
    --config_file=./models/dinov2/dinov2/configs/train/vitb16_short.yaml \
    --resume=./ckpts/ours/dinov2+ours/checkpoint.pth \
    --base_model=dinov2 \
    --use_adapter
