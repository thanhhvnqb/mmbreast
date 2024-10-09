from typing import Sequence, Union, Optional
import os

import numpy as np
import pandas as pd

from mmpretrain.datasets import CustomDataset

from mmengine.registry import DATASETS


@DATASETS.register_module(force=True)
class CsvGeneralDataset(CustomDataset):
    def __init__(
        self,
        dataset,
        ann_path: str,
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
        if not isinstance(dataset, list):
            dataset = [dataset]
        self.ann_path = ann_path
        self.datasets = dataset
        super().__init__(
            data_root,
            data_prefix,
            os.path.join(ann_path, dataset[0], "cleaned_label_split.csv"),
            True,
            extensions,
            metainfo,
            lazy_init,
            **kwargs,
        )

    def load_data_list(self):
        data_list = []
        for dataset in self.datasets:
            ann_file = os.path.join(self.ann_path, dataset, "cleaned_label_split.csv")
            df1 = pd.read_csv(ann_file)
            if self.split >= 0:
                if self.train_fold:
                    df1 = df1[df1["split"] != self.split]
                else:
                    df1 = df1[df1["split"] == self.split]
            df1["dataset"] = dataset
            print("Loaded", len(df1), "images from", dataset)
            data_list.extend(df1.to_dict("records"))
        print("Total", len(data_list), "images")
        return data_list

    def get_gt_labels(self):
        gt_labels = np.array(
            [self.get_data_info(i)[self.label_key] for i in range(len(self))],
            dtype=np.int64,
        )
        return gt_labels
