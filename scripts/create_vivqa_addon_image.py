import os
import shutil
import pandas as pd

train_addon_csv_path = "../data/vivqa/train_addon.csv"
test_csv_path = "../data/vivqa/test.csv"
MSCOCO_image_dir = "../data/MSCOCO/train2014"


def get_image_ids(data: pd.DataFrame):
    return data["img_id"].tolist()


def get_image_paths(image_ids: list):
    return [f"{MSCOCO_image_dir}/COCO_train2014_{image_id:012}.jpg" for image_id in image_ids]


def copy_to_vivqa_dir(image_paths: list, dest_dir: str):
    for image_path in image_paths:
        if not os.path.exists(image_path):
            image_path = image_path.replace("train2014", "val2014")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        shutil.copy(image_path, dest_dir)
        # then rename the image
        image_name = os.path.basename(image_path)
        new_name = image_name.split(".")[0].split("_")[-1]+".jpg"
        new_image_path = os.path.join(dest_dir, new_name)
        os.rename(os.path.join(dest_dir, image_name), new_image_path)
        

if __name__ == "__main__":
    train_addon_data = pd.read_csv(train_addon_csv_path)
    test_data = pd.read_csv(test_csv_path)

    train_addon_image_ids = get_image_ids(train_addon_data)
    test_image_ids = get_image_ids(test_data)

    train_addon_image_paths = get_image_paths(train_addon_image_ids)
    test_image_paths = get_image_paths(test_image_ids)

    image_dir = "../data/vivqa/addon_images"
    os.makedirs(image_dir, exist_ok=True)
    copy_to_vivqa_dir(train_addon_image_paths, image_dir)
    copy_to_vivqa_dir(test_image_paths, image_dir)
