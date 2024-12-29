import os

cur_path = os.getcwd()
#path = r"\mnt\workspace\EMS-YOLO-main\bdd100k\images"
path = "./bdd100k/images"
path_train = path + "/train"
path_val = path + "/val"
path_test = path + "/test"
path_save = "./bdd100k"
list_train = os.listdir(path_train)
list_val = os.listdir(path_val)
list_test = os.listdir(path_test)

for i, strings in enumerate(list_train):
    strings_new = os.path.join(cur_path,"bdd100k/images/train", strings)
    list_train[i] = strings_new
path_save_train = path_save + "/train.txt"
with open(path_save_train, 'w') as f:
    for strings in list_train:
       f.write(strings + "\n")

for i, strings in enumerate(list_val):
    strings_new = os.path.join(cur_path,"bdd100k/images/val", strings)
    list_val[i] = strings_new
path_save_val = path_save + "/val.txt"
with open(path_save_val, 'w') as f:
    for strings in list_val:
       f.write(strings + "\n")

for i, strings in enumerate(list_test):
    strings_new = os.path.join(cur_path,"bdd100k/images/test", strings)
    list_test[i] = strings_new
path_save_test = path_save + r"/test.txt"
with open(path_save_test, 'w') as f:
    for strings in list_test:
       f.write(strings + "\n")