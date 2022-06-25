import random
import cv2
from matplotlib.ft2font import GLYPH_NAMES
from Dataset import ImageDescrip
import candidate_selection as cand_selec
import numpy as np


# ------------------------- Constraints Creation ----------------------------


class Constraints:
    def __init__(self, ml, cl, ml_g, cl_g, neighborhoods, descripList, x, y):
        self.ml = ml
        self.cl = cl
        self.ml_g = ml_g
        self.cl_g = cl_g
        self.neighborhoods = neighborhoods
        self.descripList = descripList
        self.x = x
        self.y = y

    def __repr__(self):
        return "Must Link Graph: % s, Cannot Link Graph: % s, Neighborhoods: %s" % (
        self.ml_g, self.cl_g, self.neighborhoods)


def generate_img_descrip(imgList, image_classSet):
    '''Serarches for available description labels in candidate images\n
       Parameters
           imgList: image dataset array
           image_classSet: label class array
    '''
    descripList = []
    for cla in image_classSet:
        ImageList = []
        for cla_img in cla.imgList:
            for img in imgList:
                if (img.filename != cla_img):
                    continue
                img.descripList.append(cla.descrip)
                ImageList.append(img.filename)
        if (len(ImageList) != 0):
            descripList.append(cla.descrip)

    return descripList


def process_img_descrip(imgList, descripList):
    '''Attaches description labels with images\n
       Parameters
           imgList: image dataset array
           descripList: available label array
    '''
    descripSet = []
    neighborhoods = []
    y_labels = []
    for img in imgList:
        if (len(img.descripList) > 1):
            descrip = random.choice(img.descripList)
            img.descripList = []
            img.descripList.append(descrip)

        y_labels.append(descripList.index(img.descripList[0]))

    for descrip in descripList:
        ImageList = []
        for i in range(len(imgList)):
            if (imgList[i].descripList[0] != descrip):
                continue
            ImageList.append(i)
        if (len(ImageList) != 0):
            descripSet.append(ImageDescrip(descrip, ImageList))
            neighborhoods.append(ImageList)

    return descripSet, neighborhoods, y_labels


def gen_new_cons(imgList, f_name, feature_set, n_imgs, descripList):
    neighborhoods = []
    dist_matrix = []
    f_dist_matrix = []
    isTraversed = []
    k = len(descripList)
    i = 0
    # neighborhoods.append([x])
    # isTraversed.append(x)
    for img in imgList:
        dists = []
        f_dists = []
        if f_name == "SIFT":
            for _ in range(i+1):
                dists.append(0.0)
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            for j in range(i+1, len(feature_set)):
                matches = bf.match(feature_set[j], img.feature)
                matches = len(sorted(matches, key=lambda x: x.distance))
                dist = 1 - (matches / len(feature_set[0]))
                dists.append(dist)
                f_dists.append(dist)
        else:
            dists = cand_selec.dist(f_name, feature_set, img.feature, n_imgs)
            f_dists = cand_selec.dist(f_name, feature_set, img.feature, n_imgs)
        dist_matrix.append(dists)
        f_dist_matrix.append(f_dists)
        i+=1
    if f_name == "SIFT":
        for j in range(1, len(dist_matrix)):
            for d in range(j):
                dist_matrix[j][d] = dist_matrix[d][j]
    index_arr = range(n_imgs)
    while len(neighborhoods) < k:
        k_cbir = []
        traverse = True
        neighb = []
        if len(isTraversed) == 0:
            x = random.choice(index_arr)
        else:
            while (traverse):
                flag = True
                x = random.choice(index_arr)
                for i in isTraversed:
                    if i == x:
                        flag = False
                        break
                if flag:
                    traverse = False

        k_cbir = np.argsort(dist_matrix[x])[:3]
        for i in k_cbir:
            isTraversed.append(i)
            neighb.append(i)
        for dist in dist_matrix:
            for i in isTraversed:
                dist[i] = 9999
        neighborhoods.append(neighb)

    return neighborhoods, f_dist_matrix


def generate_constraints(imgList, x, image_classSet, f_name):
    '''Generates constraints and prepares the Constraints class\n
       Parameters
           imgList: image dataset array
           x: feature array
           image_classSet: label class array
    '''
    must_link = []
    cannot_link = []
    descripList = generate_img_descrip(imgList, image_classSet)
    neighborhoods, dist_matrix = gen_new_cons(
        imgList, f_name, x, len(imgList), descripList)
    candidate_descripSet, old_neighborhoods, y_labels = process_img_descrip(
        imgList, descripList)

    for set in neighborhoods:
        for i in set:
            for j in set:
                if i != j:
                    must_link.append((i, j))

    for set1 in neighborhoods:
        for set2 in neighborhoods:
            if set1 != set2:
                for i in set1:
                    for j in set2:
                        cannot_link.append((i, j))

    # for set in candidate_descripSet:
    #     for i in set.imgList:
    #         for j in set.imgList:
    #             if i != j:
    #                 must_link.append((i, j))

    # for set1 in candidate_descripSet:
    #     for set2 in candidate_descripSet:
    #         if set1.imgList != set2.imgList:
    #             for i in set1.imgList:
    #                 for j in set2.imgList:
    #                     cannot_link.append((i, j))

    ml_graph, cl_graph = transitive_entailment_graph(
        must_link, cannot_link, len(imgList))

    return imgList, Constraints(must_link, cannot_link,
                                ml_graph, cl_graph, neighborhoods, descripList, x, y_labels),  dist_matrix


def transitive_entailment_graph(ml, cl, dslen):
    '''Creates a graph using trasnsitivity\n
       Parameters
           ml: must links
           cl: cannot links
           dslen: length of images
    '''
    ml_graph = {}
    cl_graph = {}
    for i in range(dslen):
        ml_graph[i] = set()
        cl_graph[i] = set()

    for (i, j) in ml:
        ml_graph[i].add(j)
        ml_graph[j].add(i)

    def dfs(v, graph, visited, component):
        visited[v] = True
        for j in graph[v]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(v)

    for (i, j) in cl:
        cl_graph[i].add(j)
        cl_graph[j].add(i)
        for y in ml_graph[j]:
            cl_graph[i].add(y)
            cl_graph[y].add(i)
        for x in ml_graph[i]:
            cl_graph[x].add(j)
            cl_graph[j].add(x)
            for y in ml_graph[j]:
                cl_graph[x].add(y)
                cl_graph[y].add(x)

    return ml_graph, cl_graph
