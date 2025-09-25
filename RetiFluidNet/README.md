# RetiFluidNet — Self-Adaptive Multi-Attention U-Net for Retinal OCT Fluid Segmentation

**Paper:** Rasti, Reza et al. “RetiFluidNet: A Self-Adaptive and Multi-Attention Deep Convolutional Network for Retinal OCT Fluid Segmentation.” IEEE transactions on medical imaging vol. 42,5 (2023): 1413-1423. doi:10.1109/TMI.2022.3228285
 
**RetiFluidNet original github:** https://github.com/aidialab/RetiFluidNet

## Environment setup 
```bash
cd RetiFluidNet
conda env create -f environment.yml
conda activate retifluidnet  
```

## Data Layout

Add a RetouchData folder under this RetifluidNet folder with the following structure.
```text
RetouchData/
├── Spectralis/
│   ├── raw_bscans/            # The raw bscans (.raw & .mhd) from the RETOUCH dataset
│   └── raw_masks/             # The raw masks (.raw & .mhd) from the RETOUCH dataset
│   └── retouch_data/          # Create by images_Spectralis.ipynb, refer this file for more details
│   └── retouch_data_mask/     # Create by images_Spectralis.ipynb, refer this file for more details
└── Cirrus/
    ├── raw_bscans/            # The raw bscans (.raw & .mhd) from the RETOUCH dataset
    └── raw_masks/             # The raw masks (.raw & .mhd) from the RETOUCH dataset  
    └── retouch_data/          # Create by images_cirrus.ipynb, refer this file for more details
    └── retouch_data_mask/     # Create by images_cirrus.ipynb, refer this file for more details
```

## Preprocessing

Run `images_Spectralis.ipynb` and `images_cirrus.ipynb` to convert the raw images to .png files. 

The `split_data.py` is the code for creating a fair split for comparison, `retouch_splitted.json` is a saved 3-fold cross validation train-test split, which has 16 Spectralis and 16 Cirrus for each train set and 8 Spectralis and 8 Cirrus for each test set.

The DataReader.py contains functions for resize, normalization, and augementations.

## Train
Under the retifluidnet environment and folder, run:
```bash
python train.py
```

## Evaluations

The `viz_sof_icc.ipynb` notebook contains the pairwise visualization with raw OCT, ground truth, and predicted segmentations. It also computes Sum-of-Fractions (SOF) for IRF/SRF/PED at slice and patient level. Building on this, it reports ICC(A,1), the intraclass correlation coefficient of the absolute agreement between prediction and reference, stratified by devices and slices.

The `treatment_explore.ipynb` notebook explore the saved model on segmenting a patient from WashU unlabelled OCT dataset. Showing that the sum of fraction of fluids reduce after 16 months after anti-VEGF injection.
