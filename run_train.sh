# DHF1k pretraining
python3 main.py \
        --n_epochs 250 \
        --root /mnt/sdb_path/JayChao/Project/NTU/Saliency_Detection/dataset/DHF1k \
        --dataset DHF1k \
        --backbone_pretrained ./checkpoints/S3D_kinetics400.pt \
        --batch_size 3 \
        --workers 4 \
        --lr 0.0001 \
        --image_width 384 \
        --image_height 224 \
        --temporal 32

# DIEM finetuning
python3 main.py \
         --n_epochs 250 \
         --root ../dataset/DIEM \
         --dataset DIEM \
         --load ./checkpoints/DHF1k.pth \
         --batch_size 3 \
         --workers 4 \
         --lr 0.0001 \
         --image_width 384 \
         --image_height 224 \
         --temporal 32

# DIEM testing
python3 diem_val.py \
        --root ../dataset/DIEM/ \
        --load ./checkpoints/DIEM.pth
