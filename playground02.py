'''
# We're going to convert the class index on the .txt files. As we're working with only one class, it's supposed to be class 0.
# If the index is different from 0 then we're going to change it.
import glob
import os
import re

txt_file_paths = glob.glob(r"data/obj/*.txt")
for i, file_path in enumerate(txt_file_paths):
    # get image size
    with open(file_path, "r") as f_o:
        lines = f_o.readlines()

        text_converted = []
        for line in lines:
            print(line)
            numbers = re.findall("[0-9.]+", line)
            print(numbers)
            if numbers:
                # Define coordinates
                text = "{} {} {} {} {}".format(0, numbers[1], numbers[2], numbers[3], numbers[4])
                text_converted.append(text)
                print(i, file_path)
                print(text)
        # Write file
        with open(file_path, 'w') as fp:
            for item in text_converted:
                fp.writelines("%s\n" % item)
'''             
import glob
import re
import os

#Get all txt files and list them
images_list_txt = glob.glob("*.txt")
images_list_txt = [re.sub(r'\.txt$', '', file) for file in images_list_txt]
print(images_list_txt[:10])
print(len(images_list_txt))

#Get all jpeg files and list them
images_list = glob.glob("*.JPEG")
images_list = [re.sub(r'\.JPEG$', '', file) for file in images_list]
images_list = [re.sub(r'\.jpeg$', '', file) for file in images_list]
print(images_list[:10])
print(len(images_list))

#Get unlabeled files
images_list_unlabeled = [x for x in images_list if x not in images_list_txt]
images_list = [x for x in images_list if x not in images_list_unlabeled]
print(images_list_unlabeled[:10])
print(len(images_list_unlabeled))
print(len(images_list), len(images_list_txt), len(images_list_unlabeled), len(glob.glob("*")))

#Check if are the same
print(images_list_txt.sort()==images_list.sort())

#Rename to include file format
images_list_txt = [re.sub(r'\.*$', '.txt', file) for file in images_list_txt]
print(images_list_txt[:10])
print(len(images_list_txt))

images_list = [re.sub(r'\.*$', '.JPEG', file) for file in images_list]
print(images_list[:10])
print(len(images_list))

#get missed txt files
#miss = [x for x in images_list_txt if x not in images_list]
#images_list_txt = [x for x in images_list_txt if x not in miss]
#for file in images_list_unlabeled:
#    os.remove(file+".txt")

#Move them to other folder
#print(images_list_unlabeled[:10])
#for file in images_list_unlabeled:
#    os.rename(file+".JPEG", "test/"+file+".JPEG")


#test if all names listed can be opened indeed
import cv2
for file in images_list_txt:
    try:
        f = open(file)
        f.close()
        # Do something with the file
    except IOError:
        print(file)
        print("File not accessible")

for file in images_list:
    try:
        img = cv2.imread(file)
        # Do something with the file
    except IOError:
        print(file)
        print("File not accessible")


#Create training.txt file
file = open("data/train.txt", "w") 
file.write("\n".join(images_list)) 
file.close()