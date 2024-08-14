from typing import Sequence, Union, Optional

import numpy as np
import pandas as pd

from mmpretrain.datasets import CustomDataset

from mmengine.registry import DATASETS


@DATASETS.register_module(force=True)
class CsvGeneralDataset(CustomDataset):
    def __init__(
        self,
        ann_file,
        metainfo: Optional[dict] = None,
        data_root: str = "",
        data_prefix: Union[str, dict] = "",
        extensions: Sequence[str] = (
            ".jpg",
            ".jpeg",
            ".png",
            ".ppm",
            ".bmp",
            ".pgm",
            ".tif",
        ),
        lazy_init: bool = False,
        split=0,
        train=True,
        label_key="",
        **kwargs,
    ):
        self.split = split
        self.train_fold = train
        assert label_key
        self.label_key = label_key
        super().__init__(
            data_root,
            data_prefix,
            ann_file,
            True,
            extensions,
            metainfo,
            lazy_init,
            **kwargs,
        )

    def load_data_list(self):

        df1 = pd.read_csv(self.ann_file)
        if self.split >= 0:
            if self.train_fold:
                df1 = df1[df1["split"] != self.split]
            else:
                df1 = df1[df1["split"] == self.split]
        data_list = df1.to_dict("records")
        return data_list

    def get_gt_labels(self):
        gt_labels = np.array(
            [self.get_data_info(i)[self.label_key] for i in range(len(self))],
            dtype=np.int64,
        )
        return gt_labels
