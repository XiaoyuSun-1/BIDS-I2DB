import random
import math
import torch
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np

class RetifluidTransforms:
    """
    A set of static methods to perform RetiFluidNet-style transforms:
      - train(): returns a list of augmented (image, mask) pairs for training
      - val(): performs only resizing and normalization for validation
    """

    @staticmethod
    def _minmax_to_float(img: Image.Image):
        # Convert PIL [0,255] to float tensor [0,1], shape [C,H,W]
        return F.to_tensor(img)

    @staticmethod
    def _to_pil(tensor):
        # Convert single-channel Tensor back to PIL Image.   
        arr = tensor.numpy().astype(np.uint8)
        return Image.fromarray(arr, mode='L')

    @staticmethod
    def _fixed_rotate(img, mask, angle_rad):
        # rotate img and mask by radians
        angle_deg = angle_rad * 180.0 / math.pi
        return (
            F.rotate(img, angle_deg, interpolation=F.InterpolationMode.BILINEAR),
            F.rotate(mask, angle_deg, interpolation=F.InterpolationMode.NEAREST)
        )

    @staticmethod
    def _fixed_translate(img, mask, dx, dy):
        # Translate image and mask by (dx, dy) pixels.
        return (
            F.affine(img, angle=0, translate=(dx, dy), scale=1, shear=0,
                     interpolation=F.InterpolationMode.BILINEAR),
            F.affine(mask, angle=0, translate=(dx, dy), scale=1, shear=0,
                     interpolation=F.InterpolationMode.NEAREST)
        )

    @staticmethod
    def train(img: Image.Image, mask: Image.Image):
        # resize 
        img = F.resize(img, (512, 512), interpolation=F.InterpolationMode.BILINEAR)
        mask = F.resize(mask, (512, 512), interpolation=F.InterpolationMode.NEAREST)

        # normalize image and map mask to {0,1,2,3}
        img = RetifluidTransforms._minmax_to_float(img)       
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8))   
        mask = torch.div(mask, 85, rounding_mode='floor').long().unsqueeze(0) 

        out = []
        # original
        out.append((img, mask))

        #  horizontal flip
        out.append((F.hflip(img), F.hflip(mask)))

        # contrast adjustments
        out.append((F.adjust_contrast(img, 0.5), mask))

        # rotations
        for a in (0.01, -0.01, 0.02, -0.02, 0.05, -0.05):
            out.append(RetifluidTransforms._fixed_rotate(img, mask, a))

        # random translation
        tr = random.randint(0, 20)
        for dx_sign, dy_sign in ((1,1),(-1,1),(1,-1),(-1,-1)):
            dx, dy = dx_sign * tr, dy_sign * tr
            out.append(RetifluidTransforms._fixed_translate(img, mask, dx, dy))

        return out

    @staticmethod
    def val(img: Image.Image, mask: Image.Image):
        # resize and convert to tensor without aug
        img = F.resize(img, (512, 512), interpolation=F.InterpolationMode.BILINEAR)
        mask = F.resize(mask, (512, 512), interpolation=F.InterpolationMode.NEAREST)

        img = RetifluidTransforms._minmax_to_float(img)
        mask = torch.div(torch.from_numpy(np.array(mask, dtype=np.uint8)), 85,
                         rounding_mode='floor').long()
        return img, mask
