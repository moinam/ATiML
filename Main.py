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
import cluster_evaluation as clus_eval
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
                                 cons.neighborhoods, cons.y, len(cand_img[0].feature), f_name)

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


def gen_feats(f_name, query_img, img_dataset, n_imgs):
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
        sift_keypoints, features = SIFT.execute_sift(img_dataset, 1000)
        query_feature = SIFT.get_sift_features(query_img)
        print("SIFT features Creation time: %0.3fs" % (time() - t0))
    elif f_name == "MPEG7":
        print('----------- MPEG7 -----------')
        t0 = time()
        features = mpeg7.execute_mpeg7(img_dataset)
        query_feature = mpeg7.get_mpeg7_features(query_img)
        print("MPEG7 features Creation time: %0.3fs" % (time() - t0))

    return features, query_feature


def main():
    args = parser.parse_args()
    k = int(args.k)
    query = args.query
    path_query = os.getcwd() + img_folder_path + query + '.jpg'
    query_img = cv2.imread(path_query)

    '''----------- Creating Data Set ------------------'''
    t0 = time()
    image_classSet = dataset.extract_imageDescrip(
    img_class_set_names, img_class_path)
    image_dataset, n_imgs = dataset.create_dataset(img_folder_path)
    print("Data Set Creation time: %0.3fs" % (time() - t0))

    '''Extract image features, candidates using knn & create constraints for given feature name'''
    f_name_arr = ["MPEG7", "BOVW", "SIFT"]
    for f_name in f_name_arr:
        features, query_feature = gen_feats(
            f_name, query_img, image_dataset, n_imgs)

        '''----------- Candidate Selection ---------------'''
        t0 = time()
        cand_img, cand_features = cand_selec.select_candidates(
            f_name, k, features, query_feature, image_dataset, n_imgs)
        print("Candidate Selection time: %0.3fs" % (time() - t0))

        '''----------- Constraint Creation ---------------'''
        t0 = time()
        cand_img, cons, dist_matrix = gen_cons.generate_constraints(
            cand_img, cand_features, image_classSet, f_name)
        print("Constraint Creation time: %0.3fs" % (time() - t0))

        '''Generate Clusters'''
        copk_clus, copk_labels = gen_clus("COP", cand_img, cons, f_name)
        pck_clus, pck_labels = gen_clus("PC", cand_img, cons, f_name)

        '''Evaluate Clustering'''

        print(
            f'COPKMeans Silhouette Score(n={k}): {clus_eval.silhouette_score(f_name, cons.x, copk_labels, len(cons.descripList), dist_matrix)}')
        print(
            f'PCKMeans Silhouette Score(n={k}): {clus_eval.silhouette_score(f_name, cons.x, pck_labels, len(cons.descripList), dist_matrix)}')

        print(f'COPKMeans V-Measure Score(n={k}): {clus_eval.my_v_measure_score(cons.y, copk_labels)}')
        print(f'PCKMeans V-Measure Score(n={k}): {clus_eval.my_v_measure_score(cons.y, pck_labels)}')

        print(f'COPKMeans Calinski Harabasz Score(n={k}): {clus_eval.my_calinski_harabasz_score(f_name, cons.x, copk_labels)}')
        print(f'PCKMeans SCalinski Harabasz Score(n={k}): {clus_eval.my_calinski_harabasz_score(f_name, cons.x, pck_labels)}')

if __name__ == "__main__":
    main()
