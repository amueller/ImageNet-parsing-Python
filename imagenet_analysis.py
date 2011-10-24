from scipy.io import loadmat
import numpy as np
from IPython.core.debugger import Tracer
tracer = Tracer()

class ImageNetData(object):
    def __init__(self, meta_path, image_path):
        self.image_path = image_path
        self.meta_data = loadmat(meta_path)
        self.synsets = np.squeeze(self.meta_data['synsets'])
        self.ids = np.squeeze(np.array([x[0] for x in self.synsets]))
        self.wids = np.squeeze(np.array([x[1] for x in self.synsets]))
        self.word = np.squeeze(np.array([x[2] for x in self.synsets]))
        self.num_children = np.squeeze(np.array([x[4] for x in self.synsets]))
        self.children = [np.squeeze(x[5]).astype(np.int) for x in self.synsets]

    def get_class(self, search_string):
        indices = np.where([search_string+', ' in x[2][0] for x in self.synsets])[0]
        #for i in indices:
            #print(self.synsets[i])
        return indices

    def get_children(self, aclass):
        # minus one converts ids into indices in our arrays
        children = self.synsets[aclass][5][0] - 1

        print(self.synsets[aclass])
        rchildren = []

        print "-----------------"
        for child in children:
            print self.synsets[child]
            # recurse
            if self.synsets[child][4] != 0:
                rchildren.extend(self.get_children(child))
        return children, rchildren


def main():
    imnet = ImageNetData("ILSVRC2011_devkit-2.0/data/meta.mat","unpacked")
    classes = imnet.get_class("person")
    aclass = classes[0]
    children, rchildren = imnet.get_children(aclass)
    tracer()

if __name__ == "__main__":
    main()
