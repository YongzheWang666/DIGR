import os
from dataset_potsdam import DataLoaderTest, DataLoaderTrain


def get_training_data(rgb_dir,text_path_train):
    assert os.path.exists(rgb_dir)
    assert os.path.exists(text_path_train)
    print(rgb_dir)
    return DataLoaderTrain(rgb_dir,text_path_train)

def get_test_data(rgb_dir,text_path_test):
    assert os.path.exists(rgb_dir)
    assert os.path.exists(text_path_test)
    return DataLoaderTest(rgb_dir,text_path_test)
