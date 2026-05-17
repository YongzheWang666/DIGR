import os
from dataset_whu import DataLoaderTest, DataLoaderTrain


def get_training_data(rgb_dir,text_path):
    assert os.path.exists(rgb_dir)
    assert os.path.exists(text_path)
    print(rgb_dir)
    return DataLoaderTrain(rgb_dir,text_path)

def get_test_data(rgb_dir,text_path):
    assert os.path.exists(rgb_dir)
    assert os.path.exists(text_path)
    return DataLoaderTest(rgb_dir,text_path)
