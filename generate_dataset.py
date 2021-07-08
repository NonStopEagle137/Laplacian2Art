import cv2
import glob
import numpy as np
file_names = []
#save_path = r'/content/drive/My Drive/full_set_processed/'
for name in glob.glob('/content/drive/My Drive/full_set_processed/*.jpg'):
    file_names.append(name)
for name in glob.glob('/content/drive/My Drive/new_images/*.jpg'):
    file_names.append(name)
for name in glob.glob('/content/drive/My Drive/train2/*.jpg'):
    file_names.append(name)
print(file_names)
class preprocessor:
    def __init__(self, file_names, save_path_processed, save_path_transform):
        self.file_names = file_names
        self.target = save_path_processed
        self.target_transform = save_path_transform
    def process(self):
        counter = 0
        for path in self.file_names:
            counter += 1
            temp = cv2.imread(path)
            resized = cv2.resize(temp.copy(), (1024, 1024), cv2.INTER_CUBIC)
            
            #resized_t1 = cv2.GaussianBlur(resized,(35,35),0)
            resized_t2 = cv2.Laplacian(resized, cv2.CV_8U, ksize = 5)
            #resized_t2[np.where(resized_t2 < 15)] = resized_t1[np.where(resized_t2 < 15)]
            cv2.imwrite(self.target + '{}.jpg'.format(counter), resized)
            cv2.imwrite(self.target_transform + '{}.jpg'.format(counter), resized_t2)
        print('[INFO] Command Executed Successfully.')
        return
save_path_processed = r'/content/drive/My Drive/full_set_processed_iter_2/'
save_path_transform = r'/content/drive/My Drive/full_set_bt/'
pre_obj = preprocessor(file_names, save_path_processed = save_path_processed ,
                      save_path_transform = save_path_transform)
pre_obj.process()