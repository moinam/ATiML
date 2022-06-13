import random
from Dataset import ImageDescrip
# ------------------------- Constraints Creation ----------------------------
class Constraints:
    def __init__(self, ml, cl, ml_g, cl_g, neighborhoods):
        self.ml = ml
        self.cl = cl
        self.ml_g = ml_g
        self.cl_g = cl_g
        self.neighborhoods = neighborhoods

    def __repr__(self):
        return "Must Link Graph: % s, Cannot Link Graph: % s, Neighborhoods: %s" % (self.ml_g, self.cl_g, self.neighborhoods)

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
    neighborhoods = []
    for img in imgList:
        if(len(img.descripList) > 1):
            descrip = random.choice(img.descripList)
            img.descripList = []
            img.descripList.append(descrip)

    for descrip in descripList:
        ImageList = []
        for i in range(len(imgList)):
            if(imgList[i].descripList[0] != descrip):
                continue
            ImageList.append(i)
        if(len(ImageList) != 0):
            descripSet.append(ImageDescrip(descrip, ImageList))
            neighborhoods.append(ImageList)

    return descripSet, neighborhoods


def generate_constraints(imgList, image_classSet):
    must_link = []
    cannot_link = []
    candidate_descripSet, neighborhoods = process_img_descrip(
        imgList, generate_img_descrip(imgList, image_classSet))
    
    for set in candidate_descripSet:
        for i in set.imgList:
            for j in set.imgList:
                if i != j:
                    must_link.append((i, j))

    for set1 in candidate_descripSet:
        for set2 in candidate_descripSet:
            if set1.imgList != set2.imgList:
                for i in set1.imgList:
                    for j in set2.imgList:
                        cannot_link.append((i, j))
    
    ml_graph, cl_graph =transitive_entailment_graph(must_link, cannot_link, len(imgList))

    return imgList, neighborhoods, must_link, cannot_link, ml_graph, cl_graph


def transitive_entailment_graph(ml, cl, dslen):
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
