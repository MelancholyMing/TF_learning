# import sys,os
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#
# print(BASE_DIR)


# import os
# import random
#
# files_path = "data/img"
# assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)
#
# val_rate = 0.5
#
# files_name = sorted([file.split(".")[0] for file in os.listdir(files_path)])
# files_num = len(files_name)
# a = range(0,files_num)
# print(a)
# val_index = random.sample(a, k=int(files_num*val_rate))
# print(val_index)


class pe():

    def __init__(self):
        self.lis = [1, 2, 3]

    def __getitem__(self, itm):
        return self.lis[itm]


p = pe()

for i in p:
    print(i)