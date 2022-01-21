from glob import glob
import pandas as pd
from tqdm import tqdm
import cv2
import pickle
from detectron2.structures import BoxMode
from pathlib import Path
from typing import Optional
import numpy as np


def get_COVID19_data_dicts(
        root_imgdir: str,
        path_to_meta_df: str,
        path_to_train_df: str,
        use_cache: bool = True,
        debug: bool = False,
        img_size: str = '512x512',
        fold: int = 0,
        cv_scheme: str = 'skf',
        use_negative: bool = False,
        binary_task: bool = True,
        mode: str = 'train'

):
    train_df = pd.read_csv(path_to_train_df)
    cache_path = Path(".") / f"dataset_dicts_cache_{mode}_cv_scheme_{cv_scheme}_fold_{fold}_img_size_{img_size}_binary_task_{binary_task}_use_negative_{use_negative}.pkl"
    if not use_cache or not cache_path.exists():
        print("Creating data...")
        # df_meta = pd.read_csv("../input/siim-covid19-resized-1024px/meta.csv") !!!!!!!!!!!!!!!!!!!!!!
        df_meta = pd.read_csv(path_to_meta_df)

        if mode == 'train':
            if cv_scheme == 'skf':
                train_meta = df_meta[df_meta.skf_fold != fold]
            if cv_scheme == 'gkf':
                train_meta = df_meta[df_meta.gkf_fold != fold]
        if mode == 'val':
            if cv_scheme == 'skf':
                train_meta = df_meta[df_meta.skf_fold == fold]
            if cv_scheme == 'gkf':
                train_meta = df_meta[df_meta.gkf_fold == fold]

        if debug:
            train_meta = train_meta.iloc[:100]  # For debug....

        # Load 1 image to get image size.
        image_id = train_meta.iloc[0, 5]
        # image_path = str(imgdir / "train" / f"{image_id}.jpg")
        image_path = root_imgdir + img_size + f'/train/{image_id}.jpg'
        print(image_path)
        image = cv2.imread(image_path)
        resized_height, resized_width, ch = image.shape
        print(f"image shape: {image.shape}")

        dataset_dicts = []
        for index, train_meta_row in tqdm(train_meta.iterrows(), total=len(train_meta)):
            record = {}
            _, width, height, _, _, image_id = train_meta_row.values
            # filename = str(imgdir / "train" / f"{image_id}.jpg")
            # filename = str(f'../input/siim-covid19-resized-1024px/train/{image_id}.jpg') !!!!!!!!!!!!!!!
            image_path = root_imgdir + img_size + f'/train/{image_id}.jpg'
            filename = image_path
            record["file_name"] = filename
            record["image_id"] = image_id
            record["height"] = resized_height
            record["width"] = resized_width
            objs = []
            for index2, row in train_df.query("id == @image_id").iterrows():
                # print(row)
                # print(row["class_name"])
                # class_name = row["class_name"]
                class_id = None

                if binary_task:
                    class_id = row["binary_class"]
                if not binary_task:
                    class_id = row["class"]

                if class_id == 1:  # NO class
                    # It is "No finding"
                    if use_negative:
                        # Use this No finding class with the bbox covering all image area.
                        bbox_resized = [0, 0, resized_width, resized_height]
                        obj = {
                            "bbox": bbox_resized,
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "category_id": class_id,
                        }
                        objs.append(obj)

                elif class_id != 1:
                    h_ratio = resized_height / height
                    w_ratio = resized_width / width
                    bbox_resized = [
                        float(row["x_min"]) * w_ratio,
                        float(row["y_min"]) * h_ratio,
                        float(row["x_max"]) * w_ratio,
                        float(row["y_max"]) * h_ratio,
                    ]
                    obj = {
                        "bbox": bbox_resized,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": class_id,
                    }
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        with open(cache_path, mode="wb") as f:
            pickle.dump(dataset_dicts, f)

    print(f"Load from cache {cache_path}")
    with open(cache_path, mode="rb") as f:
        dataset_dicts = pickle.load(f)
    return dataset_dicts


def get_COVID19_data_dicts_test(
        root_imgdir: str,
        test_meta: pd.DataFrame,
        use_cache: bool = False,
        debug: bool = False,
):
    debug_str = f"_debug{int(debug)}"
    cache_path = Path(".") / f"dataset_dicts_cache_test.pkl"
    if not use_cache or not cache_path.exists():
        print("Creating data...")
        # test_meta = pd.read_csv(imgdir / "test_meta.csv")
        if debug:
            test_meta = test_meta.iloc[:100]  # For debug....
        # Load 1 image to get image size.
        image_id = test_meta.iloc[0, 0]
        # image_path = str(imgdir / "test" / f"{image_id}.jpg")
        image_path = root_imgdir + f'{image_id}.jpg'
        image = cv2.imread(image_path)
        resized_height, resized_width, ch = image.shape
        # print(f"image shape: {image.shape}")

        dataset_dicts = []
        for index, test_meta_row in tqdm(test_meta.iterrows(), total=len(test_meta)):
            record = {}

            image_id, height, width = test_meta_row.values
            # filename = str(imgdir / "test" / f"{image_id}.jpg")
            filename = root_imgdir + f'{image_id}.jpg'
            record["file_name"] = filename
            record["image_id"] = image_id
            # record["height"] = height
            # record["width"] = width
            record["height"] = resized_height
            record["width"] = resized_width
            # objs = []
            # record["annotations"] = objs
            dataset_dicts.append(record)
        with open(cache_path, mode="wb") as f:
            pickle.dump(dataset_dicts, f)

    # print(f"Load from cache {cache_path}")
    with open(cache_path, mode="rb") as f:
        dataset_dicts = pickle.load(f)
    return dataset_dicts