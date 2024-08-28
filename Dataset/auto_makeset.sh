#!/bin/bash
# pwd: DOIC/Dataset

script_path=$(readlink -f "${BASH_SOURCE[0]}")
script_dir=$(dirname "$script_path")
cd $script_dir

# construct d435 dataset
if [ -f "D435_vid.mp4" ]; then
    python framing.py -i D435_vid.mp4 -l d435_labels -s d435_imgs
    python yolo_dataset_generator.py -i d435_imgs -l d435_labels -s TPZ_VOCdevkit/d435
else
    echo "pick_video.mp4 not exists!"
fi

# construct d2city dataset
if [ -f "pick_video.mp4" ]; then
    python framing.py -i pick_video.mp4 -l d2city_labels -s d2city_imgs
    python yolo_dataset_generator.py -i d2city_imgs -l d2city_labels -s TPZ_VOCdevkit/d2city
else
    echo "pick_video.mp4 not exists!"
fi

# merge d2city and d435 dataset
if [ -f "D435_vid.mp4" ] && [ -f "pick_video.mp4" ]; then
    cd TPZ_VOCdevkit 
    mkdir d2city_d435

    cp -r d2city/* d2city_d435/
    cp -r d435/train/images/* d2city_d435/train/images/
    cp -r d435/train/labels/* d2city_d435/train/labels/
    cp -r d435/val/images/* d2city_d435/val/images/
    cp -r d435/val/labels/* d2city_d435/val/labels/
    
    sed '6c\path: $script_dir/TPZ_VOCdevkit/d2city_d435' d2city_d435/TPZ.yaml
    sed '7c\train: $script_dir/TPZ_VOCdevkit/d2city_d435/train/images' d2city_d435/TPZ.yaml
    sed '8c\val: $script_dir/TPZ_VOCdevkit/d2city_d435/val/images' d2city_d435/TPZ.yaml

    echo "dataset merge done!"
    cat d2city_d435/TPZ.yaml
fi

cd ..