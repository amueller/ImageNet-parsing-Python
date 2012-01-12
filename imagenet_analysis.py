import sys
import os

sys.path.insert(0, os.path.join(os.getenv("HOME"),"python_packages/lib/python2.6/site-packages/"))

import scipy
from scipy.io import loadmat
import numpy as np
import os
from glob import glob

from img_funcs import draw_bounding_boxes, grab_bounding_boxes
import xml.etree.ElementTree as ET

from joblib import Memory

memory = Memory("cache")
#import elementtree.ElementTree as ET


@memory.cache
def cached_bow(files):
    features = []
    file_names = []
    wnids = []
    counts = []

    for bow_file in files:
        print("loading %s"%bow_file)
        bow_structs = loadmat(bow_file)['image_sbow']
        if int(scipy.version.version.split(".")[1]) < 10:
            file_names.extend([str(x[0]._fieldnames) for x in bow_structs])
            bags_of_words = [np.bincount(struct[0][1][0][0][0].ravel(), minlength=1000) for struct in bow_structs]
        else:
            file_names.extend([str(x[0][0]) for x in bow_structs])
            bags_of_words = [np.bincount(struct[0][1][0][0][0].ravel(), minlength=1000) for struct in bow_structs]
        features.extend(bags_of_words)
        # if we where interested in the actual words:
        #words = [struct[0][1][0][0][0] for struct in bow_structs]
        # there is other stuff in the struct but I don't care at the moment:
        #x = [struct[0][1][0][0][1] for struct in bow_structs]
        #y = [struct[0][1][0][0][2] for struct in bow_structs]
        #scale = [struct[0][1][0][0][3] for struct in bow_structs]
        #norm = [struct[0][1][0][0][4] for struct in bow_structs]
        wnid = os.path.basename(bow_file).split(".")[0]
        wnids.append(wnid)
        counts.append(len(bags_of_words))
    features = np.array(features)
    return features, wnids, counts


class ImageNetData(object):
    """ ImageNetData needs path to meta.mat, path to images and path to annotations.
     The images are assumed to be in folders according to their synsets names

     Synsets are always handled using their index in the 'synsets' dict. This
     is their id-1 and is referred to as classidx.
     Images are handles using their id, which is the number in the file name.
     These are non-concecutive and therefore called id/imgid.
     """
    def __init__(self, meta_path, image_path=None, annotation_path=None, bow_path=None):
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.meta_path = meta_path
        self.meta_data = loadmat(os.path.join(meta_path, "meta.mat"))
        self.bow_path = bow_path

        self.synsets = np.squeeze(self.meta_data['synsets'])

        if int(scipy.version.version.split(".")[1]) < 10:
            #['ILSVRC2010_ID', 'WNID', 'words', 'gloss', 'num_children', 'children', 'wordnet_height', 'num_train_images']
            self.ids = np.squeeze(np.array([x.ILSVRC2010_ID for x in self.synsets]))
            self.wnids = np.squeeze(np.array([x.WNID for x in self.synsets]))
            self.word = np.squeeze(np.array([x.words for x in self.synsets]))
            self.num_children = np.squeeze(np.array([x.num_children for x in self.synsets]))
            self.children = [np.squeeze(x.children).astype(np.int) for x in self.synsets]

        else:
            self.ids = np.squeeze(np.array([x[0] for x in self.synsets]))
            self.wnids = np.squeeze(np.array([x[1] for x in self.synsets]))
            self.word = np.squeeze(np.array([x[2] for x in self.synsets]))
            self.num_children = np.squeeze(np.array([x[4] for x in self.synsets]))
            self.children = [np.squeeze(x[5]).astype(np.int) for x in self.synsets]

    def img_path_from_id(self, classidx, imgidx):
        wnid = self.wnids[classidx]
        return os.path.join(self.image_path, wnid, wnid+'_'+imgidx+".JPEG")

    def class_idx_from_string(self, search_string):
        """Get class index from string in class name."""
        indices = np.where([search_string in x[2][0] for x in self.synsets])[0]
        for i in indices:
            print(self.synsets[i])
        return indices

    def get_children(self, aclass):
        """Traverse tree to the leafes. Takes classidx, returns
        list of all recursive chilren of this class."""

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

    def get_bndbox(self, classidx, imageid):
        """Get bouning box coordinates for image with id ``imageid``
        in synset given by ``classidx``."""

        wnid = self.wnids[classidx]
        annotation_file = os.path.join(self.annotation_path, str(wnid), str(wnid) + "_" + str(imageid) + ".xml")
        xmltree = ET.parse(annotation_file)
        objects = xmltree.findall("object")
        result = []
        for object_iter in objects:
            bndbox = object_iter.find("bndbox")
            result.append([int(it.text) for it in bndbox])
        #[xmin, ymin, xmax, ymax] = [it.text for it in bndbox]
        return result

    def get_image_ids(self, theclass):
        wnid = self.wnids[theclass]
        files = glob(os.path.join(self.image_path,wnid,wnid+"*"))
        filenames = [os.path.basename(f)[:-5] for f in files]
        numbers = map(lambda f: f.split("_")[1], filenames)
        return numbers

    def bounding_box_images(self, classidx):
        """Get list of cut out bounding boxes
        for a given classidx."""

        if not os.path.exists("output/bounding_box"):
            os.mkdir("output/bounding_box")
        #class_string = self.word[classidx]
        wnid = self.wnids[classidx]
        if not os.path.exists(os.path.join("output/bounding_box", wnid)):
            os.mkdir(os.path.join("output/bounding_box", wnid))

        image_ids = self.get_image_ids(classidx)
        bbfiles = []
        for imgid in image_ids:
            try:
                bounding_boxes = self.get_bndbox(classidx, imgid)
            except IOError:
                #no bounding box
                #print("no xml found")
                continue
            bbfiles.append(imgid)
            img_path = self.img_path_from_id(classidx, imgid)
            out_path = str(os.path.join("output/bounding_box", wnid, wnid+'_'+imgid+".png"))
            draw_bounding_boxes(img_path, bounding_boxes, out_path)
            #if len(bbfiles)>2:
                #break
        print("annotated files: %d"%len(bbfiles))

    def class_idx_from_wnid(self, wnid):
        """Get class index in ``self.synset`` from synset id"""
        result = np.where(self.wnids==wnid)
        if len(result[0]) == 0:
            raise ValueError("Invalid wnid.")
        return result[0][0]

    def all_bounding_boxes(self, classidx):
        image_ids = self.get_image_ids(classidx)
        all_bbs = []
        for imgid in image_ids:
            try:
                img_bbs = self.get_bndbox(classidx, imgid)
            except IOError:
                #no bounding box
                #print("no xml found")
                continue
            f = self.img_path_from_id(classidx, imgid)
            all_bbs.extend(grab_bounding_boxes(f, img_bbs))
        return all_bbs;

    def load_val_labels(self):
        return np.loadtxt(os.path.join(self.meta_path, "ILSVRC2010_validation_ground_truth.txt"))

    def load_bow(self, dataset="train"):
        """Get bow representation of dataset ``dataset``.
        Legal values are ``train``, ``val`` and ``test``.

        Returns
        -------
        features : numpy array, shape [n_samples, n_features],
            containing bow representation of all images in given dataset

        labels : numpy array, shape [n_samples],
            containing classidx for image labels. (Not available for ``test``)
        """
        if not self.bow_path:
            raise ValueError("You have to specify the path to" 
                "the bow features in ``bow_path`` to be able"
                "to load them")

        files = glob(os.path.join(self.bow_path, dataset, "*.sbow.mat"))

        if len(files) == 0:
            raise ValueError("Could not find any bow files.")

        features, wnids, counts = cached_bow(files)

        if dataset == "train":
            labels_nested = [[self.class_idx_from_wnid(wnid)] * count for wnid, count in zip(wnids, counts)]
            labels = np.array([x for l in labels_nested for x in l])
        elif dataset == "val":
            labels = self.load_val_labels()
        elif dataset == "test":
            labels = None
        else:
            raise ValueError("Unknow dataset %s"%dataset)

        return features, labels


def main():
    # ImageNetData needs path to meta.mat, path to images and path to annotations.
    # The images are assumed to be in folders according to their synsets names
    #imnet = ImageNetData("ILSVRC2011_devkit-2.0/data", "unpacked", "annotation")
    imnet = ImageNetData("/nfs3group/chlgrp/datasets/ILSVRC2010/devkit-1.0/data",
            bow_path="/nfs3group/chlgrp/datasets/ILSVRC2010")

    features, labels = imnet.load_bow()
    features_val, labels_val = imnet.load_bow('val')

    from IPython.core.debugger import Tracer
    tracer = Tracer(colors="LightBG")
    tracer()
        
        

if __name__ == "__main__":
    main()
