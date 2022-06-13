from math import dist
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from skimage import feature
import Dataset as dataset


class BOVW:
    def __init__(self, random_state, n_patches, tam_patch, n_dic):
        self.random_state = random_state
        self.n_patches = n_patches
        self.tam_patch = tam_patch
        self.n_dic = n_dic

    def __repr__(self):
        return "Hyper Params: random_state: %s, n_patches: %s, tam_patch: %s, n_dic: %s" % (self.random_state, self.n_patches, self.tam_patch, self.n_dic)

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
    img = dataset.convert_greyscale(img)

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


def bovw_fit_kmeans(n_dic, patch_lbp, random_state):
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


def execute_Bovw(dataset, bovw: BOVW, n_imgs):
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
                                                         bovw.random_state,
                                                         bovw.tam_patch,
                                                         bovw.n_patches)
                                    for img in dataset)

    # Progress Report - BOVW [1]
    print('------------------ Feature Extraction - Bag of Visual Words - START ----------')
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
                                   bovw.tam_patch[0], bovw.tam_patch[0], 3))

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
    kmeans_model = bovw_fit_kmeans(bovw.n_dic, patch_lbp, bovw.random_state)
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
            patch_lbp[i*bovw.n_patches: (i*bovw.n_patches)+bovw.n_patches])

        # computes histogram and append in the array
        hist_bof, _ = np.histogram(y, bins=range(bovw.n_dic+1), density=True)
        img_feats.append(hist_bof)

    img_feats = np.array(img_feats, copy=False)
    # Progress Report - BOVW [3]
    print("Creation of image features: %0.3fs" % (time() - t0))
    print('Number of images and features = ', img_feats.shape)
    print('------------------ Feature Extraction - Bag of Visual Words - END ------------')
    return img_feats


def get_bovw_features(query_img, bovw: BOVW):
    '''Extracts features from given query image\n
       Parameters
           f_name: Feature Name
           query_img: query image
    '''
    query_patches = get_patches(
        query_img, bovw.random_state, bovw.tam_patch, bovw.n_patches)
    query_patches = np.array(query_patches, copy=False)
    query_patches = query_patches.reshape((query_patches.shape[0],
                                           bovw.tam_patch[0], bovw.tam_patch[0], 3))
    query_lbp = []
    for pat in query_patches:
        f = lbp_features(pat, 2, 8)
        query_lbp.append(f)
        
    query_lbp = np.array(query_lbp, copy=False)
    # get visual words for query
    y = bovw_kmeans_model.predict(query_lbp)
    # computes descriptor
    query_feats, _ = np.histogram(y, bins=range(bovw.n_dic+1), density=True)

    return query_feats
