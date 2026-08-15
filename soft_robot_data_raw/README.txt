All raw images file should be in the raw_images folder. The images should be named in the following format:
1.jpg
2.jpg
...
960.jpg

Split dataset:
python split_train_test.py raw_images

Generate IC dataset:
python generate_ic.py cases.txt split_raw_datasets/train split_raw_datasets/test soft_robot_ic_dataset

IC dataset should be generated in the following format:
---soft_robot_ic_dataset
    |-- train
    |   |-- 0
    |   |-- 1
    |   |-- ...
    |   |-- 11
    |-- test
    |   |-- 0
    |   |-- 1
    |   |-- ...
    |   |-- 11

Generate IL dataset:
python generate_il.py cases.txt split_raw_datasets/train split_raw_datasets/test soft_robot_il_dataset

IL dataset should be generated in the following format:
---soft_robot_il_dataset
    |-- train
    |   |-- 0
    |   |-- 1
    |   |-- ...
    |   |-- 4
    |-- test
    |   |-- 0
    |   |-- 1
    |   |-- ...
    |   |-- 4