import cv2

# --------------------SIFT---------------
def execute_sift(image_dataset, num_keypoints):
    '''Creates Speeded-Up Robust Features\n
       Parameters
           dataset: image dataset array
           num_keypoints: number of keypoints to detect.
    '''
    sift = cv2.SIFT_create(num_keypoints)
    keypoints = []
    descriptors = []
    for img in image_dataset:
        keypoints_ind, descriptors_ind = sift.detectAndCompute(
            cv2.cvtColor(img.img, cv2.COLOR_BGR2GRAY), None)
        keypoints.append(keypoints_ind)
        descriptors.append(descriptors_ind)
    return keypoints, descriptors


def get_sift_features(query_img):
    sift = cv2.SIFT_create()
    query_kp, query_desc = sift.detectAndCompute(
        cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY), None)
    return query_desc
