import cv2
import os
import argparse
import matplotlib.pyplot as plt
import os
from time import time
from sklearn import metrics
import Dataset as dataset
import bag_of_visual_words as bovw
import mpeg7_color_layout as mpeg7
import candidate_selection as cand_selec
import constraint_creation as gen_cons
import sift as SIFT
import PC_Kmeans as PCK
import COP_Kmeans as COPK
import numpy as np

# --------Acepting Input----------
# Initialize parser
parser = argparse.ArgumentParser()
# Adding arguments
parser.add_argument("--k", required=True)
parser.add_argument("--query", required=True)
# ---------------------------------

# -------------- Parameter Declaration -------------------
img_folder_path = '\\VOCtrainval_06-Nov-2007\\VOCdevkit\\VOC2007\\JPEGImages\\'
img_class_path = '\\VOCtrainval_06-Nov-2007\\VOCdevkit\\VOC2007\\ImageSets\\Main\\processed_files\\'
img_class_set_names = [{'name': 'aeroplane', 'type': 'vehicle'},
                       {'name': 'bicycle', 'type': 'vehicle'},
                       {'name': 'bird', 'type': 'animal'},
                       {'name': 'boat', 'type': 'vehicle'},
                       {'name': 'bottle', 'type': 'furniture'},
                       {'name': 'bus', 'type': 'vehicle'},
                       {'name': 'car', 'type': 'vehicle'},
                       {'name': 'cat', 'type': 'animal'},
                       {'name': 'chair', 'type': 'furniture'},
                       {'name': 'cow', 'type': 'animal'},
                       {'name': 'diningtable', 'type': 'furniture'},
                       {'name': 'dog', 'type': 'animal'},
                       {'name': 'horse', 'type': 'animal'},
                       {'name': 'motorbike', 'type': 'vehicle'},
                       {'name': 'person', 'type': 'person'},
                       {'name': 'pottedplant', 'type': 'furniture'},
                       {'name': 'sheep', 'type': 'animal'},
                       {'name': 'sofa', 'type': 'furniture'},
                       {'name': 'train', 'type': 'vehicle'},
                       {'name': 'tvmonitor', 'type': 'electronics'}]
# BOVW parameters
bovw_kmeans_model = None
tam_patch = (15, 15)
n_patches = 250
random_state = 1
n_dic = 50  # size of the dictionary


def gen_clus(c_name, cand_img, cons, f_name):
    '''Generate and Print Constrainted Cluster\n
       Parameters
           c_name: cluster name
           cand_img: candidate image list
           cons: Constraint class object
           f_name: feature name
    '''
    if c_name == "COP":
        '''---------------- COP-Kmeans ---------------'''
        c_kmeans = COPK.COP_KMeans(
            len(cons.descripList), cons.ml_g, cons.cl_g, f_name)
    else:
        '''---------------- PC-Kmeans ---------------'''
        c_kmeans = PCK.PC_Kmeans(len(cons.descripList), cons.ml_g, cons.cl_g,
                                 cons.neighborhoods, cons.y)

    c_kmeans.fit(cons.x)
    labels = c_kmeans.predict(cons.x)

    print(f'{c_name} Kmeans Clusters:')
    for i in range(len(c_kmeans.clusters)):
        print(f'Cluster {i + 1} :')
        count = 0
        for index in c_kmeans.clusters[i]:
            count += 1
            if count != len(c_kmeans.clusters[i]):
                print(f'{cand_img[index].filename}, ', end='')
            else:
                print(f'{cand_img[index].filename}.')

    return c_kmeans, labels


def execute(f_name, query_img, k, img_classSet, img_dataset, n_imgs):
    '''Generate and Print Constrainted Cluster\n
       Parameters
           f_name: cluster name
           query_img: the query image
           k: k neighbours for kNN
           img_classSet: label class array
           img_dataset: image dataset
           n_imgs: Length of img List
    '''
    if f_name == "BOVW":
        print('----------- Bag of Visual Words -----------')
        t0 = time()
        bovw_param = bovw.BOVW(random_state=random_state,
                               n_patches=n_patches, tam_patch=tam_patch, n_dic=n_dic)
        features = bovw.execute_Bovw(
            img_dataset, bovw_param, n_imgs)
        query_feature = bovw.get_bovw_features(query_img, bovw_param)
        print("BOVW features Creation time: %0.3fs" % (time() - t0))
    elif f_name == "SIFT":
        print('----------- SIFT -----------')
        t0 = time()
        sift_keypoints, features = SIFT.execute_sift(img_dataset)
        query_feature = SIFT.get_sift_features(query_img)
        print("SIFT features Creation time: %0.3fs" % (time() - t0))
    elif f_name == "MPEG7":
        print('----------- MPEG7 -----------')
        t0 = time()
        features = mpeg7.execute_mpeg7(img_dataset)
        query_feature = mpeg7.get_mpeg7_features(query_img)
        print("MPEG7 features Creation time: %0.3fs" % (time() - t0))

    '''----------- Candidate Selection ---------------'''
    t0 = time()
    cand_img, cand_features = cand_selec.select_candidates(
        f_name, k, features, query_feature, img_dataset, n_imgs)
    print("Candidate Selection time: %0.3fs" % (time() - t0))

    '''----------- Constraint Creation ---------------'''
    t0 = time()
    cand_img, cons = gen_cons.generate_constraints(
        cand_img, cand_features, img_classSet)
    print("Constraint Creation time: %0.3fs" % (time() - t0))

    return cand_img, cons


def pairwise_distance(f_name, img_feats, query_feats):
    if f_name == "BOVW":
        diq = np.sqrt(np.sum((img_feats - query_feats) ** 2))
    elif f_name == "MPEG7":
        diq = np.sqrt(np.sum((img_feats[0] - query_feats[0]) ** 2)) + np.sqrt(
            np.sum((img_feats[1] - query_feats[1]) ** 2)) + np.sqrt(
            np.sum((img_feats[2] - query_feats[2]) ** 2))
    elif f_name == "SIFT":
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(img_feats, query_feats)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = len(matches)
        diq = matches

    return diq


def silhouette_score(f_name, data_points, labels, k):
    num = len(labels)
    dist_matrix = []
    silh_points = []

    for i in range(num):
        temp = []
        for j in range(num):
            temp.append(pairwise_distance(f_name, data_points[i], data_points[j]))
        dist_matrix.append(temp)

    for i in range(num):
        clu = labels[i]
        same = 0.0
        diff = 0.0
        for j in range(num):
            if labels[j] == clu:
                same += dist_matrix[i][j]
            else:
                diff += dist_matrix[i][j]

        same_count = labels.count(labels[i])
        a = same / same_count
        b = diff / (num - same_count)
        silh_points.append((b - a) / max(b, a))

    final_score = 0.0
    for i in range(k):
        temp = 0.0
        for j in range(num):
            if i == labels[j]:
                temp += silh_points[j]

        temp /= labels.count(i)
        final_score += temp

    return final_score/k


def main():
    t0 = time()
    # Read arguments from command line
    args = parser.parse_args()
    k = int(args.k)
    query = args.query
    path_query = os.getcwd() + img_folder_path + query + '.jpg'
    query_img = cv2.imread(path_query)
    f_name = "MPEG7"

    '''----------- Creating Data Set ------------------'''
    image_classSet = dataset.extract_imageDescrip(
        img_class_set_names, img_class_path)
    image_dataset, n_imgs = dataset.create_dataset(img_folder_path)
    print("Data Set Creation time: %0.3fs" % (time() - t0))

    '''Extract image features, candidates using knn & create constraints for given feature name'''
    cand_img, cons = execute(f_name, query_img, k, image_classSet, image_dataset, n_imgs)

    '''Generate Clusters'''
    copk_clus, copk_labels = gen_clus("COP", cand_img, cons, f_name)
    pck_clus, pck_labels = gen_clus("PC", cand_img, cons, f_name)

    '''Evaluate Clustering'''
    print(f'COPKMeans Silhouette Score(n={k}): {silhouette_score(f_name, cons.x, copk_labels, len(cons.descripList))}')
    print(f'PCKMeans Silhouette Score(n={k}): {silhouette_score(f_name, cons.x, pck_labels, len(cons.descripList))}')


if __name__ == "__main__":
    main()
