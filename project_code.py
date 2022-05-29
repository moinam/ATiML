import glob
from math import dist
import cv2
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
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
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
    def __init__(self, img, filename):
        self.img = img
        self.filename = filename

    def __repr__(self):
        return "Debug image:% s Filename:% s" % (self.img, self.filename)

# Create DataSet Object


def create_dataset_object(img_file):
    '''Creates object of Class: DataSet and invokes opencv image read\n
       Parameters
           file: path for an image
    '''
    return Dataset(cv2.imread(img_file), img_file)

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

# ------------------------- Candidate Selection ----------------------------


def euclidean_dist(img_feats, query_feats):
    dists = []
    for i in range(n_imgs):
        diq = np.sqrt(np.sum((img_feats[i]-query_feats)**2))
        dists.append(diq)
    return dists


def extract_feature_query(f_name, query_img):
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


def select_candidates(k, feature_set, query_feature, query_img):
    imgs = []
    dists = euclidean_dist(feature_set, query_feature)
    k_cbir = np.argsort(dists)[:k]
    for i in range(k):
        imgs.append(image_dataset[k_cbir[i]])
    return imgs


# -------------- Main Function -------------------
image_dataset = []
img_folder_path = '\\VOCtrainval_06-Nov-2007\\VOCdevkit\\VOC2007\\JPEGImages\\'
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
    create_dataset()
    print("Data Set Creation time: %0.3fs" %(time() - t0))
    # ----------- Bag of Visual Words ---------------
    t0 = time()
    bovw_features = execute_Bovw(
        image_dataset, tam_patch, n_patches, random_state)
    print("BOVW features Creation time: %0.3fs" %(time() - t0))
    # ----------- Candidate Selection ---------------
    query_feature = extract_feature_query("BOVW", query_img)
    candidate_images_bovw = select_candidates(
        k, bovw_features, query_feature, query_img)
    print(candidate_images_bovw)

if __name__ == "__main__":
    main()
