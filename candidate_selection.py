import numpy as np

# ------------------------- Candidate Selection ----------------------------


def euclidean_dist(f_name, img_feats, query_feats, n_imgs):
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
    return dists


def select_candidates(f_name, k, feature_set, query_feature, image_dataset, n_imgs):
    '''Generates Candidates for clustering using kNN\n
       Parameters
           k: k value for kNN
           feature_set:  array of features of all the images
           query_feature: query image feature
    '''
    imgs = []
    dists = euclidean_dist(f_name, feature_set, query_feature, n_imgs)
    k_cbir = np.argsort(dists)[:k]
    for i in range(k):
        imgs.append(image_dataset[k_cbir[i]])
    return imgs