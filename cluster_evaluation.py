import numpy as np
import cv2
from sklearn import metrics

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

# def silhouette_score(f_name, data_points, labels, k):
#     res = metrics.silhouette_score(data_points, labels, metric='euclidean')
#     return res


def my_rand_score(true_labels, pred_labels):
    return metrics.adjusted_rand_score(true_labels, pred_labels)

def my_nmi(true_labels, pred_labels):
    return metrics.normalized_mutual_info_score(true_labels, pred_labels)

def my_homogeneity_score(labels_true, labels_pred):
    return metrics.homogeneity_score(labels_true, labels_pred)

def my_completeness_score(labels_true, labels_pred):
    return metrics.completeness_score(labels_true, labels_pred)

def my_v_measure_score(labels_true, labels_pred):
    return metrics.v_measure_score(labels_true, labels_pred)

def my_davies_bouldin_score(fname, dataset, labels):
    if (fname == "MPEG7"):
        dataset = np.array(dataset)
        dataset = dataset.reshape(dataset.shape[0], (dataset.shape[1]*dataset.shape[2]))
        return metrics.davies_bouldin_score(dataset, labels)
    else:
        return metrics.davies_bouldin_score(dataset, labels)

def silhouette_score(f_name, data_points, labels, k):
    num = len(labels)
    dist_matrix = []
    silh_points = []

    for i in range(num):
        temp = []
        for j in range(num):
            temp.append(pairwise_distance(
                f_name, data_points[i], data_points[j]))
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

