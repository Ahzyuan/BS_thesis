#!/bin/bash
# pwd: DOIC/

script_path=$(readlink -f "${BASH_SOURCE[0]}")
script_dir=$(dirname "$script_path")
cd $(dirname "$script_dir") # .../DOIC/

# install ultralytics
pip install ultralytics

# prepare dataset
cd Dataset
if ! [ -d "TPZ_VOCdevkit" ]; then
    bash auto_makeset.sh
fi
cd ..

# prepare pretrained weights
cd Weights
if ! [ -f "yolov8s.pt" ]; then
    wget https://bgithub.xyz/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt
fi
cd ..

# train
## d2city -> d435 (domain adaptation)
python Weights/train_da.py \
-w Weights/yolov8s.pt \
-t Dataset/TPZ_VOCdevkit/d435/TPZ.yaml \
-s Dataset/TPZ_VOCdevkit/d2city/TPZ.yaml \
--imgsz 480 \
--batch 16 \
--enable_da \
--enable_train \
--val \
--plots \
--profile

## d435 -> d2city (domain adaptation)
python Weights/train_da.py \ 
-w Weights/yolov8s.pt \
-t Dataset/TPZ_VOCdevkit/d2city/TPZ.yaml \
-s Dataset/TPZ_VOCdevkit/d435/TPZ.yaml \
--imgsz 480 \
--batch 16 \
--enable_da \
--enable_train \
--val \
--plots \
--profile

## d2city -> d2city & d435 (cross-domain testing)
python Weights/train_da.py \
-w "Weights/S(d2city)_T(d2city)_best.pt" \
-t Dataset/TPZ_VOCdevkit/d435/TPZ.yaml \
-s Dataset/TPZ_VOCdevkit/d2city/TPZ.yaml \
--imgsz 480 \
--plots \
--profile

## d435 -> d435 && d2city (cross-domain testing)
python Weights/train_da.py \
-w Weights/yolov8s.pt \
-t Dataset/TPZ_VOCdevkit/d435/TPZ.yaml \
-s Dataset/TPZ_VOCdevkit/d435/TPZ.yaml \
--imgsz 480 \
--enable_train \
--val \
--plots \
--profile

## d2city+d435 -> d2city+d435 
python Weights/train_da.py \
-w Weights/yolov8s.pt \
-t Dataset/TPZ_VOCdevkit/d2city_d435/TPZ.yaml \
-s Dataset/TPZ_VOCdevkit/d2city_d435/TPZ.yaml \
--imgsz 480 \
--enable_train \
--val \
--plots \
--profile

## d2city+d435 -> d2city & d435 (cross-domain testing)
python Weights/train_da.py \
-w "runs/detect/train_S(d2city_d435)-T(d2city_d435)/weights/best.pt" \
-t Dataset/TPZ_VOCdevkit/d2city/TPZ.yaml \
-s Dataset/TPZ_VOCdevkit/d435/TPZ.yaml \
--imgsz 480 \
--plots \
--profile