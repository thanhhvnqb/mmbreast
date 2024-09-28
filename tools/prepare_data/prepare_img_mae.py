import argparse
import os
import shutil
import sys

import pandas as pd
from tqdm import tqdm
from mmpretrain import ImageClassificationInferencer
from mmengine.dataset import Compose, default_collate

sys.path
sys.path.append(os.getcwd())
from mmbreast import *
from mmpretrain.registry import TRANSFORMS

PROCESSED_DATA_DIR = "../datasets/mmbreast"


class Inferencer(ImageClassificationInferencer):

    def __init__(
        self,
        model,
        pretrained=True,
        device=None,
        classes=None,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model, pretrained=pretrained, device=device, classes=classes, **kwargs
        )

    def _init_pipeline(self, cfg):
        test_pipeline_cfg = cfg.test_dataloader.dataset.pipeline
        test_pipeline = Compose([TRANSFORMS.build(t) for t in test_pipeline_cfg])
        return test_pipeline

    def preprocess(self, inputs, batch_size=1):
        chunked_data = self._get_chunk_data(map(self.pipeline, inputs), batch_size)
        yield from map(default_collate, chunked_data)


def parse_args():
    parser = argparse.ArgumentParser("Prepair mmbreast dataset.")
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument(
        "--data-path",
        type=str,
        default=PROCESSED_DATA_DIR,
        help="Path to root directory of processed dataset.",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default=0,
        help="Path to root directory of processed dataset.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    datasets = [
        "bmcd",
        "cmmd",
        "cddcesm",
        "miniddsm",
        # "rsna",
        # "vindr",
    ]  # "bmcd", "cmmd", "cddcesm", "miniddsm", "rsna", "vindr"
    args = parse_args()
    inferencer = Inferencer(args.config, args.checkpoint, device="cuda:0")
    total_saved_images_all = 0
    total_saved_images_0 = 0
    total_saved_images_1 = 0
    total_images_all = 0
    total_images_0 = 0
    total_images_1 = 0
    for dataset in datasets:
        SAVE_DIR = os.path.join(args.data_path, "mae", dataset)
        if os.path.isdir(SAVE_DIR):
            print(f"Directory {SAVE_DIR} exists, removing it.")
            shutil.rmtree(SAVE_DIR)
        os.makedirs(SAVE_DIR, exist_ok=True)
        for label in [0, 1]:
            os.makedirs(os.path.join(SAVE_DIR, str(label)), exist_ok=True)
        print("Processing dataset: ", dataset)
        ann_file = os.path.join(args.data_path, dataset, "cleaned_label_split.csv")
        df = pd.read_csv(ann_file)
        if dataset == "vindr":
            df = df[df["split"] == "training"].reset_index(drop=True)
        else:
            df = df[df["split"] != args.fold]
        print("Total images: ", len(df))
        df["dataset"] = dataset
        df_lst = [row for _, row in df.iterrows()]
        results = inferencer(df_lst, batch_size=2)
        num_saved_images_0 = 0
        num_saved_images_1 = 0
        num_images_0 = 0
        num_images_1 = 0
        for (index, row), result in zip(df.iterrows(), results):
            num_images_0 += 1 if row["cancer"] == 0 else 0
            num_images_1 += 1 if row["cancer"] == 1 else 0
            if result["pred_score"] > 0.5 and result["pred_label"] == row["cancer"]:
                filename = f"{row['patient_id']}@{row['image_id']}.png"
                full_filename = os.path.join(
                    args.data_path, f"{dataset}/cleaned_images/", filename
                )
                num_saved_images_0 += 1 if result["pred_label"] == 0 else 0
                num_saved_images_1 += 1 if result["pred_label"] == 1 else 0
                os.symlink(
                    full_filename,
                    os.path.join(
                        SAVE_DIR, str(result["pred_label"]), f"{dataset}_{filename}"
                    ),
                )
            # else:
            #     print(
            #         f"Skip {row['patient_id']}@{row['image_id']} with score {result['pred_score']} and label {result['pred_label']} in comparing to groundtruth: {row['cancer']}.")
        num_saved_images = num_saved_images_0 + num_saved_images_1
        print(f"Dataset: {dataset} | len = {len(df)} | 0: {num_images_0} | 1: {num_images_1}.")
        print(f"Saved {num_saved_images} | Saved 0: {num_saved_images_0} | Saved 1: {num_saved_images_1}")
        total_saved_images_all += num_saved_images
        total_saved_images_0 += num_saved_images_0
        total_saved_images_1 += num_saved_images_1
        total_images_0 += num_images_0
        total_images_1 += num_images_1
        total_images_all += len(df)
    print(f"Total images: {total_images_all}. | 0: {total_images_0} | 1: {total_images_1}.")
    print(
        f"Saved {total_saved_images_all} | 0: {total_saved_images_0} | 1: {total_saved_images_1}."
    )
