import cv2
import numpy as np
import os, os.path, shutil
def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
def remove_deleted_image():
    a = os.listdir("./Letter_extracted")
    n = 1
    i = 1
    while n <= len(a):
        path = "./Letter_extracted/%s.png" % n
        image = cv2.imread(path, 0)
        image = cv2.resize(image, (28, 28))
        intensity = np.sum(image)
        if intensity > 130000:
            file_name = "./Letter/%s.png" % i
            cv2.imwrite(file_name, image)
            i = i+1
        n+=1

