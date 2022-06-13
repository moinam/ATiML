import cv2
import os
import argparse
import matplotlib.pyplot as plt
import os
from time import time
from joblib import Parallel, delayed
import Dataset as dataset
import bag_of_visual_words as bovw
import mpeg7_color_layout as mpeg7
import candidate_selection as cand_selec
import constraint_creation as gen_cons
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

    # ----------- Creating Data Set ------------------
    image_classSet = dataset.extract_imageDescrip(
        img_class_set_names, img_class_path)
    image_dataset, n_imgs = dataset.create_dataset(img_folder_path)
    print("Data Set Creation time: %0.3fs" % (time() - t0))

    # ----------- Bag of Visual Words ---------------
    t0 = time()
    bovw_param = bovw.BOVW(random_state=random_state,
                           n_patches=n_patches, tam_patch=tam_patch, n_dic=n_dic)
    bovw_features = bovw.execute_Bovw(
        image_dataset, bovw_param, n_imgs)
    print("BOVW features Creation time: %0.3fs" % (time() - t0))

    # ----------- MPEG7 Feature extraction ----------
    t0 = time()
    mpeg7_features = mpeg7.execute_mpeg7(image_dataset)
    print("MPEG7 features Creation time: %0.3fs" % (time() - t0))

    # ----------- Candidate Selection ---------------
    t0 = time()
    query_feature = bovw.get_bovw_features(query_img, bovw_param)
    candidate_images_bovw = cand_selec.select_candidates(
        "BOVW", k, bovw_features, query_feature, image_dataset, n_imgs)
    print("BOVW Candidate Selection time: %0.3fs" % (time() - t0))
    t0 = time()
    query_feature = mpeg7.get_mpeg7_features(query_img)
    candidate_images_mpeg7 = cand_selec.select_candidates(
        "MPEG7", k, mpeg7_features, query_feature, image_dataset, n_imgs)
    print("MPEG7 Candidate Selection time: %0.3fs" % (time() - t0))

    # ----------- Constraint Creation ---------------
    t0 = time()
    bovw_cons = gen_cons.Constraints([], [], [], [], [])
    mpeg7_cons = gen_cons.Constraints([], [], [], [], [])
    candidate_images_bovw, bovw_cons.neighborhoods, bovw_cons.ml, bovw_cons.cl, bovw_cons.ml_g, bovw_cons.cl_g = gen_cons.generate_constraints(
        candidate_images_bovw, image_classSet)
    candidate_images_mpeg7, mpeg7_cons.neighborhoods, mpeg7_cons.ml, mpeg7_cons.cl, mpeg7_cons.ml_g, mpeg7_cons.cl_g = gen_cons.generate_constraints(
        candidate_images_mpeg7, image_classSet)
    print("Constraint Creation time: %0.3fs" % (time() - t0))
    print("checkpoint - mew")


if __name__ == "__main__":
    main()
