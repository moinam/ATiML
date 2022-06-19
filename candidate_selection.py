import numpy as np
import cv2
import sift as SIFT

# ------------------------- Candidate Selection ----------------------------


def dist(f_name, img_feats, query_feats, n_imgs):
    '''Calculates the Euclidean distance for the given features\n
       Parameters
           img_feats: candidate image features
           query_feats: query image feature
    '''
    dists = []
    if f_name == "BOVW":
        for i in range(n_imgs):
            diq = np.sqrt(np.sum((img_feats[i]-query_feats)**2))
            dists.append(diq)
    elif f_name == "MPEG7":
        for i in range(n_imgs):
            diq = np.sqrt(np.sum((img_feats[i][0] - query_feats[0]) ** 2)) + np.sqrt(
                np.sum((img_feats[i][1] - query_feats[1]) ** 2)) + np.sqrt(
                np.sum((img_feats[i][2] - query_feats[2]) ** 2))
            dists.append(diq)
    elif f_name == "SIFT":
        for i in range(n_imgs):
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(img_feats[i], query_feats)
            matches = sorted(matches, key=lambda x: x.distance)
            matches = len(matches)
            dists.append(matches)

    return dists


def format_imgResult(f_name, image_dataset, feature_set, dists, k):
    imgs = []
    min = 9999
    features = []
    if f_name == "SIFT":
        k_cbir = np.argsort(dists)[::-1][:k]
    else:
        k_cbir = np.argsort(dists)[:k]
    for i in range(k):
        if f_name != "SIFT":
            image_dataset[k_cbir[i]].feature = feature_set[k_cbir[i]]
        imgs.append(image_dataset[k_cbir[i]])
        features.append(feature_set[k_cbir[i]])
        if(min > len(feature_set[k_cbir[i]])):
            min = len(feature_set[k_cbir[i]])
    
    return imgs, features, min


def select_candidates(f_name, k, feature_set, query_feature, image_dataset, n_imgs):
    '''Generates Candidates for clustering using kNN\n
       Parameters
           k: k value for kNN
           feature_set:  array of features of all the images
           query_feature: query image feature
    '''
    imgs = []
    features = []
    dists = dist(f_name, feature_set, query_feature, n_imgs)
    imgs, features, min = format_imgResult(
        f_name, image_dataset, feature_set, dists, k)
    
    if f_name == "SIFT":
        features = []
        sift_keypoints, features = SIFT.execute_sift(imgs, min)
        for i in range(len(features)):
            features[i] = features[i][:min]
            imgs[i].feature = features[i]

    return imgs, features