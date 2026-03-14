"""
Build IN-9 dataset from imagenet-mini.
Output structure:
    in9/
        train/
            0/  1/  2/  ...  8/
        val/
            0/  1/  2/  ...  8/
"""

import os
import json
import shutil

IMAGENET_MINI = "imagenet-mini"
OUT_DIR = "in9"
MAPPING_FILE = "in_to_in9.json"

CLASS_NAMES = ['Dog', 'Bird', 'Vehicle', 'Reptile', 'Carnivore',
               'Insect', 'Instrument', 'Primate', 'Fish']

with open(MAPPING_FILE) as f:
    idx_to_in9 = json.load(f)  # {"0": 5, "1": 3, ...}

for split in ["train", "val"]:
    split_dir = os.path.join(IMAGENET_MINI, split)
    classes = sorted(os.listdir(split_dir))  # tri alphabétique = index ImageNet
    synset_to_idx = {s: i for i, s in enumerate(classes)}

    copied = 0
    for synset, idx in synset_to_idx.items():
        if str(idx) not in idx_to_in9:
            continue
        super_class = idx_to_in9[str(idx)]
        src = os.path.join(split_dir, synset)
        dst = os.path.join(OUT_DIR, split, str(super_class))
        os.makedirs(dst, exist_ok=True)
        for img in os.listdir(src):
            shutil.copy(os.path.join(src, img), os.path.join(dst, img))
            copied += 1

    print(f"{split}: {copied} images copiées")
    for i, name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(OUT_DIR, split, str(i))
        count = len(os.listdir(class_dir)) if os.path.exists(class_dir) else 0
        print(f"  {i} {name}: {count} images")
