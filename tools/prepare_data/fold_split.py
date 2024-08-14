import argparse
import os
import sys

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold


# - There are no new site ID in test dataset.
# - Patient IDs in train and test sets do not overlap
# - Image IDs in train and test sets do not overlap
# - There are no new laterality values in test dataset. (R: 27439, L: 27267)
# - There are new machine IDs in test dataset. (This is already raised by @abebe9849 in here)
# - There are no new view values in test dataset. (MLO: 27903, CC: 26765, AT: 19, LM: 10, ML: 8, LMO: 1)
# - No. of images/patient are all >= 4 (train: True)
# - Site ID is always the same for each patient. (train: True)
# - Age is always the same for each patient. (train: True)
# - There are no overlap of machine IDs between two sites in test dataset. (train: True)
# - Some patients underwent mammography with multiple machines. (train: only 1 patient 22637 with more than 1 machine)
# - No. of images in site ID 1 > No. of images in site ID 2.
# - All patients have CC and MLO images for both sides
# - More than 40% of images are from machine ID 49 (43% for train dataset) (by @kaggleqrdl)
# - Mean age of patients is between 56-61 (58.6 for train set), and patients in site 1 are younger than those in site 2. (by @kaggleqrdl)
# - 1-2% of patients use implants (1.4% for train set).(by @kaggleqrdl)
# - Positive rate is between 0.0204-0.0256 (0.0206 for train dataset). (shown by @zzy990106 and others)
# - Age column contains nan, while others do not


# Each fold split should stratified the following assumtions:
# - Group by 'patient_id'
# - Positive rate is between 0.0204-0.0256 (0.0206 for train dataset)
# - There are new machine IDs in test dataset --> group by machine_id as well (generalization to un-seen machine or not?)
# - No. of images in site ID 1 > No. of images in site ID 2.
# - All patients have CC and MLO images for both sides
# - More than 40% of images are from machine ID 49 (43% for train dataset)
# - Mean age of patients is between 56-61 (58.6 for train set), and patients in site 1 are younger than those in site 2
# - 1-2% of patients use implants (1.4% for train set)

PROCESSED_DATA_DIR = "./datasets/"


# For BMCD/CMMD/CDD-CESM/MiniDDSM dataset
def fold_check_other(train_df, val_df):
    # ensure no overlap
    train_patients = set(train_df.patient_id.unique())
    val_patients = set(val_df.patient_id.unique())
    assert len(list(val_patients - train_patients)) == len(list(val_patients))

    ret = {}
    num_samples = len(val_df)
    num_patients = val_df.patient_id.nunique()
    # percent of positive (samples level)
    ret["val_pos_sample_num"] = val_df.cancer.sum()
    ret["val_pos_patient_num"] = val_df[val_df.cancer == 1].patient_id.nunique()
    ret["val_pos_sample_percent"] = ret["val_pos_sample_num"] / num_samples
    ret["val_pos_patient_percent"] = ret["val_pos_patient_num"] / num_patients
    ret["mean_age"] = val_df.age.mean()
    return ret


# For Vindr dataset
def fold_check_vindr(train_df, val_df):
    # ensure no overlap
    train_patients = set(train_df.patient_id.unique())
    val_patients = set(val_df.patient_id.unique())
    assert len(list(val_patients - train_patients)) == len(list(val_patients))

    ret = {}
    num_samples = len(val_df)
    num_patients = val_df.patient_id.nunique()
    # percent of positive (samples level)
    ret["val_pos_sample_num"] = val_df.cancer.sum()
    ret["val_pos_patient_num"] = val_df[val_df.cancer == 1].patient_id.nunique()
    ret["val_pos_sample_percent"] = ret["val_pos_sample_num"] / num_samples
    ret["val_pos_patient_percent"] = ret["val_pos_patient_num"] / num_patients

    def convert_age(age_str):
        # Remove the trailing 'Y' and leading zeros, then convert to integer
        try:
            return int(age_str.rstrip("Y"))
        except:
            return 0

    ret["mean_age"] = val_df.age.apply(convert_age).mean()
    return ret


# For RSNA dataset
def fold_check_rsna(train_df, val_df):
    # ensure no overlap
    train_patients = set(train_df.patient_id.unique())
    val_patients = set(val_df.patient_id.unique())
    assert len(list(val_patients - train_patients)) == len(list(val_patients))

    ret = {}
    num_samples = len(val_df)
    num_patients = val_df.patient_id.nunique()
    # percent of positive (samples level)
    ret["val_pos_sample_num"] = val_df.cancer.sum()
    ret["val_pos_patient_num"] = val_df[val_df.cancer == 1].patient_id.nunique()
    ret["val_pos_sample_percent"] = ret["val_pos_sample_num"] / num_samples
    ret["val_pos_patient_percent"] = ret["val_pos_patient_num"] / num_patients
    val_machine_ids = sorted(list(val_df.machine_id.unique()))
    train_machine_ids = sorted(list(train_df.machine_id.unique()))
    not_in_train_machine_ids = list(set(val_machine_ids) - set(train_machine_ids))
    ret["val_machine_ids"] = val_machine_ids
    ret["train_machine_ids"] = train_machine_ids
    ret["not_in_train_machine_ids"] = not_in_train_machine_ids
    ret["mean_site_id"] = val_df.site_id.mean()
    num_machine_id_49 = len(val_df[val_df.machine_id == 49])
    ret["num_machine_id_49"] = num_machine_id_49
    #     assert num_machine_id_49 / num_samples > 0.4
    ret["mean_age"] = val_df.age.mean()
    ret["mean_age_site1"] = val_df[val_df.site_id == 1].age.mean()
    ret["mean_age_site2"] = val_df[val_df.site_id == 2].age.mean()
    assert ret["mean_age_site1"] < ret["mean_age_site2"]
    ret["implant_pct"] = val_df[val_df.implant == 1].patient_id.nunique() / num_patients
    assert ret["implant_pct"] < 0.02 and ret["implant_pct"] > 0.01
    return ret


def parse_args():
    parser = argparse.ArgumentParser("Prepair classification dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["bmcd", "cmmd", "cddcesm", "miniddsm", "rsna", "vindr"],
        help="Path to root directory of processed dataset.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    fold_check_func = {
        "bmcd": fold_check_other,
        "cddcesm": fold_check_other,
        "cmmd": fold_check_other,
        "miniddsm": fold_check_other,
        "rsna": fold_check_rsna,
        "vindr": fold_check_vindr,
    }
    args = parse_args()
    dataset = args.dataset
    fold_check = fold_check_func[dataset]

    CSV_LABEL_PATH = os.path.join(
        PROCESSED_DATA_DIR,
        "classification",
        dataset,
        "cleaned_label.csv",
    )
    SPLIT_LABEL_PATH = os.path.join(
        PROCESSED_DATA_DIR,
        "classification",
        dataset,
        "cleaned_label_split.csv",
    )
    df = pd.read_csv(CSV_LABEL_PATH)
    SAVE_DIR = os.path.join(
        PROCESSED_DATA_DIR,
        "classification",
        dataset,
        "fold",
    )
    os.makedirs(SAVE_DIR, exist_ok=True)
    ret = []
    if dataset == "vindr":
        for i in range(4):
            fold_train_df = df[df["split"] == "training"].reset_index(drop=True)
            fold_val_df = df[df["split"] == "test"].reset_index(drop=True)
            save_fold_train_path = os.path.join(SAVE_DIR, f"train_fold_{i}.csv")
            save_fold_val_path = os.path.join(SAVE_DIR, f"val_fold_{i}.csv")
            fold_ret = fold_check(fold_train_df, fold_val_df)
            print(fold_ret)
            ret.append(fold_ret)
            # save
            fold_train_df.to_csv(save_fold_train_path, index=False)
            fold_val_df.to_csv(save_fold_val_path, index=False)
        df.loc[df["split"] == "training", "split"] = 1
        df.loc[df["split"] == "test", "split"] = 0
        fold_ret = fold_check(df.loc[df["split"] == 1], df.loc[df["split"] == 0])
        ret.append(fold_ret)
        df.to_csv(SPLIT_LABEL_PATH, index=False)
        print("\n--------------------\n\n\n")
    else:
        orig_df = df.copy()
        spliter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=67)
        for i, (train_idxs, val_idxs) in enumerate(
            spliter.split(df, df.cancer, groups=df.patient_id)
        ):
            print(f"Fold {i}:")
            fold_train_df = orig_df.loc[train_idxs].reset_index(drop=True)
            fold_val_df = orig_df.loc[val_idxs].reset_index(drop=True)
            print(len(fold_train_df), len(fold_val_df))
            save_fold_train_path = os.path.join(SAVE_DIR, f"train_fold_{i}.csv")
            save_fold_val_path = os.path.join(SAVE_DIR, f"val_fold_{i}.csv")
            fold_ret = fold_check(fold_train_df, fold_val_df)
            print(fold_ret)
            # save
            fold_train_df.to_csv(save_fold_train_path, index=False)
            fold_val_df.to_csv(save_fold_val_path, index=False)
            print("\n--------------------\n\n\n")
            df.loc[val_idxs, "split"] = i
        for i in range(4):
            fold_train_df = df.loc[df["split"] == i].reset_index(drop=True)
            fold_val_df = df.loc[df["split"] != i].reset_index(drop=True)
            fold_ret = fold_check(fold_train_df, fold_val_df)
            ret.append(fold_ret)
        df.to_csv(SPLIT_LABEL_PATH, index=False)

    for k in ret[0]:
        print(k)
        for fold_idx, fold_ret in enumerate(ret):
            print("\tfold ", fold_idx, ":", fold_ret[k])
