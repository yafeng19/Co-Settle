
export CUDA_VISIBLE_DEVICES=0


python3 -m cal_metric \
    --sample_csv_path=./sampling/val_frames_1000_sample.csv \
    --output_dir=./eval_metric/base \
    --model_type=vitb16 \
    --resume=./ckpts/base/MAE_INet_vitb16_800ep/checkpoint.pth \
    --base_model=mae


python3 -m cal_metric \
    --sample_csv_path=./sampling/val_frames_1000_sample.csv \
    --output_dir=./eval_metric/base \
    --model_type=vitb16 \
    --resume=./ckpts/base/IJEPA_INet_vitb16_800ep/ijepa-ep800.pth.tar \
    --base_model=ijepa


python3 -m cal_metric \
    --sample_csv_path=./sampling/val_frames_1000_sample.csv \
    --output_dir=./eval_metric/base \
    --model_type=vitb16 \
    --resume=./ckpts/base/CLIP_vitb16/ViT-B-16.pt \
    --base_model=clip


python3 -m cal_metric \
    --sample_csv_path=./sampling/val_frames_1000_sample.csv \
    --output_dir=./eval_metric/base \
    --model_type=vitb16 \
    --resume=./ckpts/base/BLIP_vitb16/model_base_retrieval_coco.pth \
    --base_model=blip


python3 -m cal_metric \
    --sample_csv_path=./sampling/val_frames_1000_sample.csv \
    --output_dir=./eval_metric/base \
    --model_type=vitb16 \
    --resume=./ckpts/base/DINO_vitb16/dino_vitbase16_pretrain.pth \
    --base_model=dino


python3 -m cal_metric \
    --sample_csv_path=./sampling/val_frames_1000_sample.csv \
    --output_dir=./eval_metric/base \
    --model_type=vitb16 \
    --resume=./ckpts/base/mocov3_INet_vitb16_300ep/vit-b-300ep.pth.tar \
    --base_model=mocov3


python3 -m cal_metric \
    --sample_csv_path=./sampling/val_frames_1000_sample.csv \
    --output_dir=./eval_metric/base \
    --model_type=vitb16 \
    --resume=./ckpts/base/iBot_INet_vitb16_400ep/checkpoint_teacher_vitb-16.pth \
    --base_model=ibot


python3 -m cal_metric \
    --sample_csv_path=./sampling/val_frames_1000_sample.csv \
    --output_dir=./eval_metric/base \
    --model_type=vitb16 \
    --config_file=./models/dinov2/dinov2/configs/train/vitb16_short.yaml \
    --resume=./ckpts/base/DINOv2_INet_vitb16_100ep/checkpoint.pth \
    --base_model=dinov2

