import glob
from math import dist
import random
import cv2
import csv
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from skimage import feature
from sklearn.pipeline import Pipeline

# --------Acepting Input----------
# Initialize parser
parser = argparse.ArgumentParser()
# Adding arguments
parser.add_argument("--k", required=True)
parser.add_argument("--query", required=True)

# ------------ Data Initialization --------------
# Dataset class for easy access


class Dataset:
    def __init__(self, img, filename, filepath, descripList, grayscale):
        self.img = img
        self.filename = filename
        self.filepath = filepath
        self.descripList = descripList
        self.grayscale = grayscale

    def __repr__(self):
        return "Image:% s Description List:% s" % (self.filename, self.descripList)

# Image Description class for easy access


class ImageDescrip:
    def __init__(self, descrip, imgList):
        self.descrip = descrip
        self.imgList = imgList

    def __repr__(self):
        return "Image Description:% s ImageList:% s " % (self.descrip, self.imgList)

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
    return Dataset(img, img_file[-10:-4], img_file, [], convert_greyscale(img))

# Create Image Data Set


def create_dataset():
    '''Import Images from file location and create DataSet class array: image_dataset
    '''
    # Image Data Set location setting
    imdir = os.getcwd() + img_folder_path + '*.jpg'

    # Reading images from location to a class array
    global image_dataset
    image_dataset = Parallel(n_jobs=-1)(delayed(create_dataset_object)(file)
                                        for file in glob.glob(imdir))

    # total of images
    global n_imgs
    n_imgs = len(image_dataset)

# Extract ImageList


def extract_imageDescrip():
    '''Extracting Image List for different Image Descriptions.
       Import csv files from file location and create ImageDescrip class, 
       array: image_classSet
    '''
    global image_classSet
    for descrip in img_class_set_names:
        entry = []
        path_text = os.getcwd() + img_class_path + \
            descrip + '_trainval.csv'
        with open(path_text, 'r') as file:
            my_reader = csv.reader(file, delimiter=',')
            for row in my_reader:
                if(row[1] == '-1'):
                    continue
                entry.append(row[0])

        image_classSet.append(ImageDescrip(descrip, entry))


# ------------------------- BOVW Functions ----------------------------

def get_patches(img, random_state, patch_size=(11, 11), n_patches=250):
    '''Extracts subimages from an image\n
       Parameters
           img_file: path for an image
           patch_size: size of each patch
           n_patches: number of patches to be extracted
    '''
    # Extract subimages
    patch = extract_patches_2d(img,
                               patch_size=patch_size,
                               max_patches=n_patches,
                               random_state=random_state)

    return patch.reshape((n_patches,
                          np.prod(patch_size) * len(img.shape)))


def lbp_features(img, radius=1, sampling_pixels=8):
    '''Creates Local Binary Patterns feature vector for each patch\n
       Parameters
           img: patch from patch array
           radius: Radius of the circle
           sampling_pixels: number of neigboring pixles
    '''
    # LBP operates in single channel images so if RGB images are provided
    # we have to convert it to grayscale
    # if (len(img.shape) > 2):
    #     img = img.astype(float)
    #     # RGB to grayscale convertion using Luminance
    #     # We are extracting R,G & B channels from patch and
    #     # using grayscale conversion algorithm that is used in OpenCV’s cvtColor().
    #     img = img[:, :, 0] * 0.3 + img[:, :, 1] * 0.59 + img[:, :, 2] * 0.11

    # # converting to uint8 type for 256 graylevels
    # img = img.astype(np.uint8)

    # # normalize values can also help improving description
    # i_min = np.min(img)
    # i_max = np.max(img)
    # if (i_max - i_min != 0):
    #     img = (img - i_min) / (i_max - i_min)

    img = convert_greyscale(img)

    # compute LBP
    lbp = feature.local_binary_pattern(
        img, sampling_pixels, radius, method="uniform")

    # LBP returns a matrix with the codes, so we compute the histogram
    # We only want the frequency of each "number" occuring which is
    # each combination of which pixels are smaller and which are greater than the center.
    # We only save the density values of the histogram
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(
        0, sampling_pixels + 3), range=(0, sampling_pixels + 2))

    # normalization
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    # return the histogram of Local Binary Patterns
    return hist


def bovw_fit_kmeans(n_dic, patch_lbp):
    '''Fitting K-means for BOVW\n
       Parameters
           n_dic: k value for k-Means
           patch_lbp: Linear Binary Pattern patches
    '''
    global bovw_kmeans_model
    # Define a KMeans clustering model
    bovw_kmeans_model = KMeans(n_clusters=n_dic,
                               verbose=False,
                               init='random',
                               random_state=random_state,
                               n_init=3)
    # fit the model
    bovw_kmeans_model.fit(patch_lbp)
    return bovw_kmeans_model


def execute_Bovw(dataset, tam_patch, n_patches, random_state):
    '''Creates Bag of Visual Words\n
       Parameters
           dataset: image dataset array
           tam_patch: size of each patch
           n_patches: number of patches to be extracted
    '''

    # Extract patches in parallel
    # returns a list of the same size of the number of images
    t0 = time()
    patch_arr = Parallel(n_jobs=-1)(delayed(get_patches)(img.img,
                                                         random_state,
                                                         tam_patch,
                                                         n_patches)
                                    for img in dataset)

    # Progress Report - BOVW [1]
    print('------------------ Feature Extraction - Bag of Visual Words ------------------')
    print('Patches extracted to create dictionary of features')
    print('Total of images = ', len(patch_arr))
    print('Size of each array of patches = ', patch_arr[0].shape)
    print("Patches Creation time: %0.3fs" % (time() - t0))

    # Debugging code [Use for clarity]
    # # shows some image patches
    # img_ind = 1
    # plt.figure(figsize=(8, 3))
    # for i in np.arange(1, 11):
    #     plt.subplot(2, 5, i)
    #     plt.imshow(patch_arr[img_ind][i].reshape(
    #         (tam_patch[0], tam_patch[1], 3)))
    #     plt.axis('off')
    # plt.show()

    patch_arr = np.array(patch_arr, copy=True)
    patch_arr = patch_arr.reshape((patch_arr.shape[0] * patch_arr.shape[1],
                                   tam_patch[0], tam_patch[0], 3))

    # obtaining features lbp for each patch
    patch_lbp = []
    t0 = time()
    patch_lbp = Parallel(n_jobs=-1)(delayed(lbp_features)
                                    (pat, 2, 8) for pat in patch_arr)

    # Progress Report - BOVW [2]
    print('Instances = ', len(patch_lbp), ' size = ', patch_lbp[0].shape[0])
    print('Created LBP feature spaces')
    print('\tpatches = ', len(patch_lbp), ' size = ', patch_lbp[0].shape[0])
    print("LBP Instance Creation time: %0.3fs" % (time() - t0))
    t0 = time()
    patch_lbp = np.array(patch_lbp, copy=False)
    kmeans_model = bovw_fit_kmeans(n_dic, patch_lbp)
    print("Kmeans fitting time: %0.3fs" % (time() - t0))
    # Debugging code [Use for clarity]
    # plt.scatter(patch_lbp[:, 0], patch_lbp[:, 1], c=kmeans_model.labels_)
    # plt.title('2 LBP components and its labels')
    # plt.axis('off')
    # plt.show()
    t0 = time()
    # compute features for each image
    img_feats = []
    for i in range(n_imgs):
        # predicting n_patches of an image
        y = kmeans_model.predict(
            patch_lbp[i*n_patches: (i*n_patches)+n_patches])

        # computes histogram and append in the array
        hist_bof, _ = np.histogram(y, bins=range(n_dic+1), density=True)
        img_feats.append(hist_bof)

    img_feats = np.array(img_feats, copy=False)
    # Progress Report - BOVW [3]
    print("Creation of image features: %0.3fs" % (time() - t0))
    print('Number of images and features = ', img_feats.shape)
    return img_feats

# ------------------------- SURF Functions ----------------------------


def execute_surf(dataset, threshold):
    '''Creates Speeded-Up Robust Features\n
       Parameters
           dataset: image dataset array
           threshold: Hessian Threshold
    '''

    #surf = cv2.features2d.SURF_create(threshold)


    # Extract patches in parallel
    # returns a list of the same size of the number of images
    t0 = time()
    # patch_arr = Parallel(n_jobs=-1)(delayed(get_patches)(img.img,
    #                                                      random_state,
    #                                                      tam_patch,
    #                                                      n_patches)
    #                                 for img in dataset)

    # Progress Report - SURF [1]
    # print('------------------ Feature Extraction - Bag of Visual Words ------------------')

    return 0

# ------------------------- Candidate Selection ----------------------------


def euclidean_dist(img_feats, query_feats):
    '''Calculates the Euclidean distance for the given features\n
       Parameters
           img_feats: candidate image features
           query_feats: query image feature
    '''
    dists = []
    for i in range(n_imgs):
        diq = np.sqrt(np.sum((img_feats[i]-query_feats)**2))
        dists.append(diq)
    return dists


def extract_feature_query(f_name, query_img):
    '''Extracts features from given query image\n
       Parameters
           f_name: Feature Name
           query_img: query image
    '''
    if(f_name == "BOVW"):
        query_patches = get_patches(
            query_img, random_state, tam_patch, n_patches)
        query_patches = np.array(query_patches, copy=False)
        query_patches = query_patches.reshape((query_patches.shape[0],
                                               tam_patch[0], tam_patch[0], 3))
        query_lbp = []
        for pat in query_patches:
            f = lbp_features(pat, 2, 8)
        query_lbp.append(f)
        query_lbp = np.array(query_lbp, copy=False)
        # get visual words for query
        y = bovw_kmeans_model.predict(query_lbp)
        # computes descriptor
        query_feats, _ = np.histogram(y, bins=range(n_dic+1), density=True)
    return query_feats


def select_candidates(k, feature_set, query_feature):
    '''Generates Candidates for clustering using kNN\n
       Parameters
           k: k value for kNN
           feature_set:  array of features of all the images
           query_feature: query image feature
    '''
    imgs = []
    dists = euclidean_dist(feature_set, query_feature)
    k_cbir = np.argsort(dists)[:k]
    for i in range(k):
        imgs.append(image_dataset[k_cbir[i]])
    return imgs

# ------------------------- Constraints Creation ----------------------------


def generate_img_descrip(imgList):
    descripList = []
    for cla in image_classSet:
        ImageList = []
        for cla_img in cla.imgList:
            for img in imgList:
                if(img.filename != cla_img):
                    continue
                img.descripList.append(cla.descrip)
                ImageList.append(img.filename)
        if(len(ImageList) != 0):
            descripList.append(cla.descrip)

    return descripList


def process_img_descrip(imgList, descripList):
    descripSet = []
    for img in imgList:
        if(len(img.descripList) > 1):
            descrip = random.choice(img.descripList)
            img.descripList = []
            img.descripList.append(descrip)

    for descrip in descripList:
        ImageList = []
        for img in imgList:
            if(img.descripList[0] != descrip):
                continue
            ImageList.append(img.filename)
        if(len(ImageList) != 0):
            descripSet.append(ImageDescrip(descrip, ImageList))

    return descripSet


# -------------- Main Function -------------------
image_dataset = []
img_folder_path = '\\VOCtrainval_06-Nov-2007\\VOCdevkit\\VOC2007\\JPEGImages\\'
img_class_path = '\\VOCtrainval_06-Nov-2007\\VOCdevkit\\VOC2007\\ImageSets\\Main\\processed_files\\'
img_class_set_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                       'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
image_classSet = []
n_imgs = 0
# BOF parameters
bovw_kmeans_model = None
tam_patch = (15, 15)
n_patches = 250
random_state = 1
n_dic = 50  # size of the dictionary


def main():

    t0 = time()
    # Read arguments from command line
    args = parser.parse_args()
    k = int(args.k)
    query = args.query
    path_query = os.getcwd() + img_folder_path + query + '.jpg'
    query_img = cv2.imread(path_query)
    # ----------- Creating Data Set ------------------
    extract_imageDescrip()
    create_dataset()
    print("Data Set Creation time: %0.3fs" % (time() - t0))
    # ------------------- SURF ----------------------
    #surf_features = execute_surf(image_dataset, 400)
    # ----------- Bag of Visual Words ---------------
    t0 = time()
    bovw_features = execute_Bovw(
        image_dataset, tam_patch, n_patches, random_state)
    print("BOVW features Creation time: %0.3fs" % (time() - t0))
    # ----------- Candidate Selection ---------------
    t0 = time()
    query_feature = extract_feature_query("BOVW", query_img)
    candidate_images_bovw = select_candidates(
        k, bovw_features, query_feature)
    print("Candidate Selection time: %0.3fs" % (time() - t0))
    # ----------- Constraint Creation ---------------
    t0 = time()
    cand_img_bovw_descripList = generate_img_descrip(candidate_images_bovw)
    cand_img_bovw_descripSet = process_img_descrip(
        candidate_images_bovw, cand_img_bovw_descripList)
    print("Constraint Creation time: %0.3fs" % (time() - t0))
    print(cand_img_bovw_descripSet)


if __name__ == "__main__":
    main()
