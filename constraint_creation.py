import random
from Dataset import ImageDescrip
# ------------------------- Constraints Creation ----------------------------


def generate_img_descrip(imgList, image_classSet):
    descripList = []
    for cla in image_classSet:
        ImageList = []
        for cla_img in cla.imgList:
            for img in imgList:
                if(img.filename != cla_img):
                    continue
                img.descripList.append(cla.descrip)
                ImageList.append(img.filename)
        if(len(ImageList) != 0):
            descripList.append(cla.descrip)

    return descripList


def process_img_descrip(imgList, descripList):
    descripSet = []
    for img in imgList:
        if(len(img.descripList) > 1):
            descrip = random.choice(img.descripList)
            img.descripList = []
            img.descripList.append(descrip)

    for descrip in descripList:
        ImageList = []
        for img in imgList:
            if(img.descripList[0] != descrip):
                continue
            ImageList.append(img.filename)
        if(len(ImageList) != 0):
            descripSet.append(ImageDescrip(descrip, ImageList))

    return descripSet
