import os
from typing import Sequence, Optional
import warnings
import math

import cv2
import numpy as np
import torch

import albumentations as A
from mmengine.registry import TRANSFORMS
from mmpretrain.structures import DataSample
from mmcv.transforms import LoadImageFromFile, BaseTransform
from mmengine.structures import LabelData
from mmengine.utils import is_str

from .augs import CustomRandomSizedCropNoResize
from .utils.s2d2s import depth_to_space, space_to_depth

aux_view = {
    "AT": 0,
    "CC": 1,
    "LM": 2,
    "LMO": 3,
    "ML": 4,
    "MLO": 5,
}
aux_density = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
}


@TRANSFORMS.register_module(force=True)
class DepthToSpace(BaseTransform):
    def __init__(self, scale):
        self.scale = scale

    def transform(self, results: dict) -> Optional[dict]:
        results["inputs"] = depth_to_space(results["inputs"], self.scale)
        return results


@TRANSFORMS.register_module(force=True)
class SpaceToDepth(BaseTransform):
    def __init__(self, scale):
        self.scale = scale

    def transform(self, results: dict) -> Optional[dict]:
        results["inputs"] = space_to_depth(results["inputs"], self.scale)
        return results


@TRANSFORMS.register_module(force=True)
class TrainAugment(BaseTransform):

    def __init__(self):
        self.transform_fn = A.Compose(
            [
                # crop
                CustomRandomSizedCropNoResize(
                    scale=(0.5, 1.0), ratio=(0.5, 0.8), always_apply=False, p=0.4
                ),
                # downscale
                A.OneOf(
                    [
                        A.Downscale(
                            scale_min=0.75,
                            scale_max=0.95,
                            interpolation=dict(
                                upscale=cv2.INTER_LINEAR, downscale=cv2.INTER_AREA
                            ),
                            always_apply=False,
                            p=0.1,
                        ),
                        A.Downscale(
                            scale_min=0.75,
                            scale_max=0.95,
                            interpolation=dict(
                                upscale=cv2.INTER_LANCZOS4, downscale=cv2.INTER_AREA
                            ),
                            always_apply=False,
                            p=0.1,
                        ),
                        A.Downscale(
                            scale_min=0.75,
                            scale_max=0.95,
                            interpolation=dict(
                                upscale=cv2.INTER_LINEAR, downscale=cv2.INTER_LINEAR
                            ),
                            always_apply=False,
                            p=0.8,
                        ),
                    ],
                    p=0.125,
                ),
                # contrast
                # relative dark/bright between region, like HDR
                A.OneOf(
                    [
                        A.RandomToneCurve(scale=0.3, always_apply=False, p=0.5),
                        A.RandomBrightnessContrast(
                            brightness_limit=(-0.1, 0.2),
                            contrast_limit=(-0.4, 0.5),
                            brightness_by_max=True,
                            always_apply=False,
                            p=0.5,
                        ),
                    ],
                    p=0.5,
                ),
                # affine
                A.OneOf(
                    [
                        A.ShiftScaleRotate(
                            shift_limit=0,
                            scale_limit=[-0.15, 0.15],
                            rotate_limit=[-30, 30],
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=0,
                            shift_limit_x=[-0.1, 0.1],
                            shift_limit_y=[-0.2, 0.2],
                            rotate_method="largest_box",
                            always_apply=False,
                            p=0.6,
                        ),
                        # one of with other affine
                        A.ElasticTransform(
                            alpha=1,
                            sigma=20,
                            # alpha_affine=10,
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=None,
                            approximate=False,
                            same_dxdy=False,
                            always_apply=False,
                            p=0.2,
                        ),
                        # distort
                        A.GridDistortion(
                            num_steps=5,
                            distort_limit=0.3,
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=None,
                            normalized=True,
                            always_apply=False,
                            p=0.2,
                        ),
                    ],
                    p=0.5,
                ),
            ],
            p=0.9,
        )

        print("TRAIN AUG:\n", self.transform_fn)

    def transform(self, results: dict) -> Optional[dict]:
        results["img"] = self.transform_fn(image=results["img"])["image"]
        return results


@TRANSFORMS.register_module(force=True)
class ValTransform(BaseTransform):

    def __init__(self, size, interpolation=cv2.INTER_CUBIC):
        self.size = size
        self.interpolation = interpolation
        self.max_h, self.max_w = size

        def _fit_resize(image, **kwargs):
            img_h, img_w = image.shape[:2]
            r = min(self.max_h / img_h, self.max_w / img_w)
            new_h, new_w = int(img_h * r), int(img_w * r)
            new_image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
            # print(image.shape, new_image.shape)
            return new_image

        self.transform_fn = A.Compose(
            [
                A.Lambda(name="FitResize", image=_fit_resize, always_apply=True, p=1.0),
                A.PadIfNeeded(
                    min_height=self.max_h,
                    min_width=self.max_w,
                    pad_height_divisor=None,
                    pad_width_divisor=None,
                    position=A.augmentations.geometric.transforms.PadIfNeeded.PositionType.CENTER,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=None,
                    always_apply=True,
                    p=1.0,
                ),
            ]
        )

    def transform(self, results: dict) -> Optional[dict]:
        results["img"] = self.transform_fn(image=results["img"])["image"]
        return results


@TRANSFORMS.register_module(force=True)
class LoadImageRSNABreastAux(LoadImageFromFile):
    def __init__(
        self,
        to_float32: bool = False,
        color_type: str = "color",
        imdecode_backend: str = "cv2",
        file_client_args: dict = dict(backend="disk"),
        ignore_empty: bool = False,
        cropped=True,
        file_key="",
        label_key="",
        img_prefix="",
        extension="",
    ) -> None:
        super().__init__(
            to_float32, color_type, imdecode_backend, file_client_args, ignore_empty
        )
        self.file_key = file_key
        self.label_key = label_key
        self.img_prefix = img_prefix
        self.extension = extension
        self.cropped = cropped

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        filename = f"{results['patient_id']}@{results['image_id']}.png"
        filename = os.path.join(
            self.img_prefix, f"{results['dataset']}/cleaned_images/", filename
        )
        img = cv2.imread(filename)
        assert img is not None

        results["img"] = img
        results["img_shape"] = img.shape[:2]
        results["ori_shape"] = img.shape[:2]
        # label part
        results["gt_label"] = int(results["cancer"])

        # aux label part
        try:
            view = aux_view[results["view"]]
        except Exception:
            view = -1
        try:
            BIRADS = results["BIRADS"]
            if isinstance(BIRADS, str):
                BIRADS = float(BIRADS[0]) - 1
            elif math.isnan(BIRADS):
                BIRADS = -1
        except Exception:
            BIRADS = -1
        try:
            invasive = results["invasive"]
            if isinstance(invasive, str):
                invasive = float(invasive)
            elif math.isnan(invasive):
                invasive = -1
        except Exception:
            invasive = -1
        try:
            difficulty = results["difficult_negative_case"]
            if isinstance(difficulty, str):
                difficulty = float(difficulty)
            elif math.isnan(difficulty):
                difficulty = -1
        except Exception:
            difficulty = -1
        try:
            implant = results["implant"]
            if isinstance(implant, str):
                implant = float(implant)
            elif math.isnan(implant):
                implant = -1
        except Exception:
            implant = -1
        try:
            age = results["age"]
            if isinstance(age, str):
                age = float(age)
            elif math.isnan(age):
                age = -1
        except Exception:
            age = -1
        try:
            density = aux_density[results["density"]]
        except Exception:
            density = -1
        aux_label = np.array(
            [view, BIRADS, invasive, difficulty, implant, age, density],
            dtype=np.float32,
        )
        results["aux_label"] = aux_label
        return results


class MxDataSample(DataSample):
    """
    without correct collate_fn the aux label will stay in cpu
    """

    def set_aux_label(self, value) -> "DataSample":
        label_data = getattr(self, "_aux_label", LabelData())
        label_data.label = torch.tensor(value, dtype=torch.float32)
        self.aux_label = label_data
        return self

    @property
    def aux_label(self):
        return self._aux_label

    @aux_label.setter
    def aux_label(self, value: LabelData):
        self.set_field(value, "_aux_label", dtype=LabelData)

    @aux_label.deleter
    def aux_label(self):
        del self._aux_label


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.
    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f"Type {type(data)} cannot be converted to tensor."
            "Supported types are: `numpy.ndarray`, `torch.Tensor`, "
            "`Sequence`, `int` and `float`"
        )


@TRANSFORMS.register_module()
class PackMxInputs(BaseTransform):
    """Pack the inputs data for the classification.
    **Required Keys:**
    - img
    - gt_label (optional)
    - ``*meta_keys`` (optional)
    **Deleted Keys:**
    All keys in the dict.
    **Added Keys:**
    - inputs (:obj:`torch.Tensor`): The forward data of models.
    - data_samples (:obj:`~mmcls.structures.DataSample`): The annotation
      info of the sample.
    Args:
        meta_keys (Sequence[str]): The meta keys to be saved in the
            ``metainfo`` of the packed ``data_samples``.
            Defaults to a tuple includes keys:
            - ``sample_idx``: The id of the image sample.
            - ``img_path``: The path to the image file.
            - ``ori_shape``: The original shape of the image as a tuple (H, W).
            - ``img_shape``: The shape of the image after the pipeline as a
              tuple (H, W).
            - ``scale_factor``: The scale factor between the resized image and
              the original image.
            - ``flip``: A boolean indicating if image flip transform was used.
            - ``flip_direction``: The flipping direction.
    """

    def __init__(
        self,
        meta_keys=(
            "sample_idx",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "flip",
            "flip_direction",
        ),
    ):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data."""
        packed_results = dict()
        if "img" in results:
            img = results["img"]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            packed_results["inputs"] = to_tensor(img)
        else:
            warnings.warn(
                'Cannot get "img" in the input dict of `PackClsInputs`,'
                "please make sure `LoadImageFromFile` has been added "
                "in the data pipeline or images have been loaded in "
                "the dataset."
            )

        data_sample = MxDataSample()
        if "gt_label" in results:
            gt_label = results["gt_label"]
            data_sample.set_gt_label(gt_label)
        if "aux_label" in results:
            data_sample.set_aux_label(results["aux_label"])

        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)
        packed_results["data_samples"] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(meta_keys={self.meta_keys})"
        return repr_str
