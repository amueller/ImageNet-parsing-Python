import Image
import ImageDraw
import numpy as np

from IPython.core.debugger import Tracer
tracer = Tracer()

def draw_bounding_boxes(imgfile, bounding_boxes, outfile):
    """ Takes an image path, draws the bounding boxes in "box" (list of 4-tuples) and
    writes output to filename."""
    im = Image.open(imgfile)
    for xmin, ymin, xmax, ymax in bounding_boxes:
        draw = ImageDraw.Draw(im)
        draw.rectangle([xmin, ymin, xmax, ymax], outline='red')
    im.save(outfile)

def grab_bounding_boxes(imgfile, bounding_boxes):
    im = Image.open(imgfile)
    bbs = []
    for box in bounding_boxes:
        bbs.append(im.crop(box))
    return bbs

def collection_mean(images):
    """ input: list of pil images, output mean of images
    converts all images to 512x512 by scaling and padding
    """

    resized_images = []
    for image in images:
        # resize so larger side is 512
        scale_factor = 512. / max(image.size)
        image = image.resize(np.array(image.size)*scale_factor)
        image = np.array(image)/255.

        # make square by padding with zeros
        padding = np.abs(image.shape[0] - image.shape[1])
        up = np.ceil(padding/2.)
        down = np.floor(padding/2.)
        if image.shape[0] > image.shape[1]:
            padded = np.hstack([np.zeros([image.shape[0], up, 3]),image,np.zeros([image.shape[0], down, 3])])
        else:
            padded = np.vstack([np.zeros([up, image.shape[1], 3]),image,np.zeros([down, image.shape[1], 3])])

        resized_images.append(padded)

    mean = np.mean(resized_images, axis=0)
    import matplotlib.pyplot as plt
    plt.imshow(mean)
    plt.show()
    tracer()




