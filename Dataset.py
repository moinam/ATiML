import os
from joblib import Parallel, delayed
import numpy as np
import glob
import cv2
import csv

# ------------ Data Initialization --------------
# Dataset class for easy access


class Dataset:
    def __init__(self, img, filename, filepath, descripList, feature):
        self.img = img
        self.filename = filename
        self.filepath = filepath
        self.descripList = descripList
        self.feature = feature

    def __repr__(self):
        return "Image: % s, Description List: % s, Feature: %s" % (self.filename, self.descripList, self.feature)

# Image Description class for easy access


class ImageDescrip:
    def __init__(self, descrip, imgList):
        self.descrip = descrip
        self.imgList = imgList

    def __repr__(self):
        return "Image Description: % s, ImageList: % s" % (self.descrip, self.imgList)

# Create Grayscale


def convert_greyscale(img):
    '''Convert image to Grayscale\n
       Parameters
           img: image object
    '''
    # we have to convert it to grayscale
    if (len(img.shape) > 2):
        img = img.astype(float)
        # RGB to grayscale convertion using Luminance
        # We are extracting R,G & B channels from patch and
        # using grayscale conversion algorithm that is used in OpenCV’s cvtColor().
        img = img[:, :, 0] * 0.3 + img[:, :, 1] * 0.59 + img[:, :, 2] * 0.11

    # converting to uint8 type for 256 graylevels
    img = img.astype(np.uint8)

    # normalize values can also help improving description
    i_min = np.min(img)
    i_max = np.max(img)
    if (i_max - i_min != 0):
        img = (img - i_min) / (i_max - i_min)

    return img

# Create DataSet Object


def create_dataset_object(img_file):
    '''Creates object of Class: DataSet and invokes opencv image read\n
       Parameters
           file: path for an image
    '''
    img = cv2.imread(img_file)
    return Dataset(img, img_file[-10:-4], img_file, [], None)

# Create Image Data Set


def create_dataset(img_folder_path):
    '''Import Images from file location and create DataSet class array: image_dataset
    '''
    # Image Data Set location setting
    imdir = os.getcwd() + img_folder_path + '*.jpg'

    # Reading images from location to a class array
    image_dataset = Parallel(n_jobs=-1)(delayed(create_dataset_object)(file)
                                        for file in glob.glob(imdir))

    return image_dataset, len(image_dataset)

# Extract ImageList


def extract_imageDescrip(img_class_set_names, img_class_path):
    '''Extracting Image List for different Image Descriptions.
       Import csv files from file location and create ImageDescrip class, 
       array: image_classSet
    '''
    image_classSet = []
    temp_img_classSet = []
    classSet = []
    for descrip in img_class_set_names:
        entry = []
        path_text = os.getcwd() + img_class_path + \
            descrip["name"] + '_trainval.csv'
        with open(path_text, 'r') as file:
            my_reader = csv.reader(file, delimiter=',')
            for row in my_reader:
                if(row[1] == '-1'):
                    continue
                entry.append(row[0])

        if descrip["type"] not in classSet:
            classSet.append(descrip["type"])
        temp_img_classSet.append(ImageDescrip(descrip["type"], entry))

    for classType in classSet:
        entry = set()
        for imgdescrip in temp_img_classSet:
            if classType == imgdescrip.descrip:
                entry.update(imgdescrip.imgList)
        image_classSet.append(ImageDescrip(classType, set((entry))))

    return image_classSet
