# VisionFM - Vision Foundation Model

**Paper:** Qiu, Jianing, et al. “Development and Validation of a Multimodal Multitask Vision Foundation Model for Generalist Ophthalmic Artificial Intelligence.” NEJM AI vol. 1,12 (2024). doi:10.1056/AIoa2300221

**VisionFM original github:** https://github.com/ABILab-CUHK/VisionFM.git

## Environment setup 
Create the environment with conda commands:

```bash
conda create -n vfm python=3.8
conda activate vfm
```
Install the dependencies:
```bash
git clone https://github.com/ABILab-CUHK/VisionFM.git
cd VisionFM
pip install -r requirments.txt
```

## Data Layout

Use the processed images from the `images_Spectralis.ipynb` and `images_cirrus.ipynb` from the `RetiFluidNet` folder.

```text
RetouchData/
├── Spectralis/
│   ├── retouch_data/          # Create by images_Spectralis.ipynb, refer this file for more details
│   └── retouch_data_mask/     # Create by images_Spectralis.ipynb, refer this file for more details
└── Cirrus/
    ├── retouch_data/          # Create by images_cirrus.ipynb, refer this file for more details
    └── retouch_data_mask/     # Create by images_cirrus.ipynb, refer this file for more details
```

## Preprocessing

`dataset.py`: This file defines dataset loader, SegImgs, that control how images and masks are read, how many augmentation variants to generate, and how chosen transforms are applied during training/validation.

`transforms.py`: This file provides the orifinal general-purpose transform functions used by VisionFM (we mainly use flips, Resize, NormalizeMinMax here) that can be composed into custom augmentation pipelines for both images and image–mask pairs.

`transforms_retifluidnet.py`: This file contains the set of augmentations with the style from RetiFluidNet, converted from TensorFlow to PyTorch. It provides multiple variants per image, including flips, contrast, rotations, translations, for training, and simple resize/normalize for validation.

Remember to download the **pretrained weight** from https://drive.google.com/file/d/1o6E-ine2QLx2pxap-c77u-SU0FjxwypA/view.

## Train

```bash
python3 evaluation/train_seg_decoder_CV.py \
  --data_path # your path to dataset folder\
  --modality OCT \
  --pretrained_weights # your path to VFM_OCT_weights.pth \
  --num_labels 4 \
  --input_size 512\
  --num_workers 0 \
  --name # your name of this round of training \
  --output_dir # your path to output results folder \
  --split_file # your path to retouch_splitted.json \
  --epochs 30 \
  --batch_size_per_gpu 64
```

## Evaluations
