import Image
import ImageDraw

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

