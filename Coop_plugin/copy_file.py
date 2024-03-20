import shutil
import os

root = '/groups/klhead/xic/cyverse_data/cooperative-vehicle-infrastructure/'
folders = ["infrastructure-trajectories/", "vehicle-trajectories/", "traffic-light/", "cooperative-trajectories/"]
train_raw_file_names = [x for x in os.listdir(root+"infrastructure-trajectories/" +"train") if "csv" in x]
val_raw_file_names = [x for x in os.listdir(root+"infrastructure-trajectories/"+"val") if "csv" in x]

copy_train_file_names = train_raw_file_names[:len(train_raw_file_names)//10]
copy_val_file_names = val_raw_file_names[:len(val_raw_file_names)//10]

dst_dir = '/groups/klhead/xic/v2x_data_mini/cooperative-vehicle-infrastructure/'

for folder in folders:
    print("copying into {}...".format(folder))
    for train_file in copy_train_file_names:
        shutil.copyfile(root+folder+"train/"+train_file, dst_dir+folder+"train/"+train_file)
    for val_file in copy_val_file_names:
        shutil.copyfile(root+folder+"val/"+val_file, dst_dir+folder+"val/"+val_file)

print("done!")

