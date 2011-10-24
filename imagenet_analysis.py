from scipy.io import loadmat
import numpy as np
import os
from glob import glob

import xml.etree.ElementTree as ET
#import elementtree.ElementTree as ET

from IPython.core.debugger import Tracer
tracer = Tracer()

class ImageNetData(object):
    def __init__(self, meta_path, image_path, annotation_path):
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.meta_data = loadmat(meta_path)
        self.synsets = np.squeeze(self.meta_data['synsets'])
        self.ids = np.squeeze(np.array([x[0] for x in self.synsets]))
        self.wnids = np.squeeze(np.array([x[1] for x in self.synsets]))
        self.word = np.squeeze(np.array([x[2] for x in self.synsets]))
        self.num_children = np.squeeze(np.array([x[4] for x in self.synsets]))
        self.children = [np.squeeze(x[5]).astype(np.int) for x in self.synsets]

    def get_class(self, search_string):
        indices = np.where([search_string in x[2][0] for x in self.synsets])[0]
        for i in indices:
            print(self.synsets[i])
        return indices

    def get_children(self, aclass):
        # minus one converts ids into indices in our arrays
        children = self.synsets[aclass][5][0] - 1

        print(self.synsets[aclass])
        rchildren = children.tolist()

        print "-----------------"
        for child in children:
            print self.synsets[child]
            # recurse
            if self.synsets[child][4] != 0:
                rchildren.extend(self.get_children(child))
        return rchildren

    def get_bndbox(self, classid, imageid):
        wnid = self.wnids[classid]
        annotation_file = os.path.join(self.annotation_path, str(wnid), str(wnid) + "_" + str(imageid) + ".xml")
        xmltree = ET.parse(annotation_file)
        bndbox = xmltree.find("object").find("bndbox")
        #[xmin, ymin, xmax, ymax] = [it.text for it in bndbox]
        return [it.text for it in bndbox]

    def get_image_files(self, theclass):
        wnid = self.wnids[theclass]
        files = glob(os.path.join(self.image_path,wnid,wnid+"*"))
        print(files)


    def bounding_box_images(self, theclass):
        if not os.path.exists("bounding_box"):
            os.mkdir("bounding_box")
        class_string = self.word[theclass]
        wnid = self.wnids[theclass]
        if not os.path.exists(wnid):
            os.mkdir(wnid)

        self.get_image_files(theclass)



def main():
    imnet = ImageNetData("ILSVRC2011_devkit-2.0/data/meta.mat", "unpacked", "annotation")
    classes = imnet.get_class("hammerhead")
    aclass = classes[0]
    rchildren = imnet.get_children(aclass)
    imnet.bounding_box_images(aclass)
    for theclass in rchildren:
        imnet.bounding_box_images(theclass)
    #imnet.get_bndbox(979, 110)
    tracer()

if __name__ == "__main__":
    main()
