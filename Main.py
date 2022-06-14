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
# --------Acepting Input----------
# Initialize parser
parser = argparse.ArgumentParser()
# Adding arguments
parser.add_argument("--k", required=True)
parser.add_argument("--query", required=True)
# ---------------------------------

# -------------- Parameter Declaration -------------------
image_dataset = []
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
image_classSet = []
n_imgs = 0
# BOVW parameters
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

    '''----------- Creating Data Set ------------------'''
    image_classSet = dataset.extract_imageDescrip(
        img_class_set_names, img_class_path)
    image_dataset, n_imgs = dataset.create_dataset(img_folder_path)
    print("Data Set Creation time: %0.3fs" % (time() - t0))

    ''' ---------- Bag of Visual Words Feature Extraction ---------------'''
    t0 = time()
    bovw_param = bovw.BOVW(random_state=random_state,
                           n_patches=n_patches, tam_patch=tam_patch, n_dic=n_dic)
    bovw_features = bovw.execute_Bovw(
        image_dataset, bovw_param, n_imgs)
    print("BOVW features Creation time: %0.3fs" % (time() - t0))

    '''----------- SIFT Feature extraction ----------'''
    t0 = time()
    sift_keypoints, sift_desc = SIFT.execute_sift(image_dataset)
    print("SIFT features Creation time: %0.3fs" % (time() - t0))

    '''----------- MPEG7 Feature extraction ----------'''
    t0 = time()
    mpeg7_features = mpeg7.execute_mpeg7(image_dataset)
    print("MPEG7 features Creation time: %0.3fs" % (time() - t0))

    '''----------- Candidate Selection ---------------'''
    t0 = time()
    query_feature = bovw.get_bovw_features(query_img, bovw_param)
    candidate_images_bovw = cand_selec.select_candidates(
        "BOVW", k, bovw_features, query_feature, image_dataset, n_imgs)
    print("BOVW Candidate Selection time: %0.3fs" % (time() - t0))
    t0 = time()
    query_desc = SIFT.get_sift_features(query_img)
    candidate_images_sift = cand_selec.select_candidates(
        "SIFT", k, sift_desc, query_desc, image_dataset, n_imgs)
    print("SIFT Candidate Selection time: %0.3fs" % (time() - t0))
    t0 = time()
    query_feature = mpeg7.get_mpeg7_features(query_img)
    candidate_images_mpeg7, cand_mpeg7_features = cand_selec.select_candidates(
        "MPEG7", k, mpeg7_features, query_feature, image_dataset, n_imgs)
    print("MPEG7 Candidate Selection time: %0.3fs" % (time() - t0))

    '''----------- Constraint Creation ---------------'''
    t0 = time()
    candidate_images_bovw, bovw_cons = gen_cons.generate_constraints(
        candidate_images_bovw, image_classSet)
    candidate_images_sift, sift_cons = gen_cons.generate_constraints(
        candidate_images_sift, image_classSet)
    candidate_images_mpeg7, mpeg7_cons = gen_cons.generate_constraints(
        candidate_images_mpeg7, cand_mpeg7_features, image_classSet)
    print("Constraint Creation time: %0.3fs" % (time() - t0))

    '''---------------- COP-Kmeans ---------------'''

    # COP_Kmeans test repeat n times
    cop_kmeans = COPK.COP_KMeans(
        len(mpeg7_cons.descripList), mpeg7_cons.ml_g, mpeg7_cons.cl_g)
    cop_kmeans.fit(mpeg7_cons.x)
    predict_labels_copkmeans = cop_kmeans.is_clustered
    print("COP - Kmeans Clusters:")
    for i in range(len(cop_kmeans.clusters)):
        print(f'Cluster {i+1} :')
        count = 0
        for index in cop_kmeans.clusters[i]:
            count += 1
            if count != len(cop_kmeans.clusters[i]):
                print(f'{candidate_images_mpeg7[index].filename}, ', end='')
            else:
                print(f'{candidate_images_mpeg7[index].filename}.')

    '''---------------- PC-Kmeans ---------------'''

    # PC_Kmeans  test repeat n times
    pc_kmeans = PCK.PC_Kmeans(len(mpeg7_cons.descripList), mpeg7_cons.ml_g, mpeg7_cons.cl_g,
                              mpeg7_cons.neighborhoods, mpeg7_cons.y)
    pc_kmeans.fit(mpeg7_cons.x)
    predict_labels_pckmeans = pc_kmeans.is_clustered
    print("PC - Kmeans Clusters:")
    for i in range(len(pc_kmeans.clusters)):
        print(f'Cluster {i+1} :')
        count = 0
        for index in pc_kmeans.clusters[i]:
            count += 1
            if count != len(pc_kmeans.clusters[i]):
                print(f'{candidate_images_mpeg7[index].filename}, ', end='')
            else:
                print(f'{candidate_images_mpeg7[index].filename}.')


if __name__ == "__main__":
    main()
