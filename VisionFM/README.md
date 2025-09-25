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
