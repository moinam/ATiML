import cv2
from time import time
from project_code import image_dataset
import Dataset as dataset

sift = cv2.SIFT_create()

img = cv2.imread("C:\\Users\\pravi\\Desktop\\Untitled Folder\\ATiML\\VOCtrainval_06-Nov-2007\\VOCdevkit\\VOC2007\\JPEGImages\\009637.jpg", cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
kp, desc = sift.detectAndCompute(gray, None)

print(kp)
print("some space")
print(type(kp))
print(desc)
print("some space")
print(desc.shape)
# def my_sift(dataset):
#     keypoints = []
#     descriptors = []
#     for img in dataset:
#         keypoints_ind, descriptors_ind = sift.detectAndCompute(img.img,None)
#         keypoints.append(keypoints_ind)
#         descriptors.append(descriptors_ind)
#     return keypoints, descriptors

# def main():
#     tot_kp, tot_desc = my_sift(project_code.image_dataset)
#     print(tot_kp[1])

#     print("some space")

#     print(tot_desc[1])
    

# if __name__ == "__main__":
#     main()