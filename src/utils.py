import matplotlib.pyplot as plt 
import numpy as np 
import cv2 as cv 
import json 
import os

COLORS = [[43, 43, 200], [43, 75, 200], [43, 106, 200], [43, 137, 200], [43, 169, 200], [43, 200, 195], [43, 200, 163], [43, 200, 132], [43, 200, 101], [43, 200, 69], [54, 200, 43], [85, 200, 43], [116, 200, 43], [148, 200, 43], [179, 200, 43], [
        200, 184, 43], [200, 153, 43], [200, 122, 43], [200, 90, 43], [200, 59, 43], [200, 43, 64], [200, 43, 95], [200, 43, 127], [200, 43, 158], [200, 43, 190], [174, 43, 200], [142, 43, 200], [111, 43, 200], [80, 43, 200], [43, 43, 200]]

def rename_all(directory, extension):
    for i, filename in enumerate(os.listdir(directory)):
        current = directory + filename
        new_name = directory + str(i) + extension 
        os.rename(current, new_name)


def generate_all_masks():
    image_list = sorted(os.listdir('../dataset'), key=lambda x: int(x.split('.')[0]))
    marked_list = sorted(os.listdir('../marked'), key=lambda x: int(x.split('.')[0]))
    i = 0
    for img, mrk in zip(image_list, marked_list):
        im = cv.imread(os.path.join('../dataset', img), 0)
        mrk_path = os.path.join('../marked', mrk)
        shape_dicts = get_poly(mrk_path)
        binary_img = create_binary_masks(im, shape_dicts)
        mask_path = os.path.join('../masks', str(i)+'.jpg')
        cv.imwrite(mask_path, binary_img)
        i += 1
        # plot_pair([im, binary_img], gray=True)
        # plt.show()
        # break
        

def get_poly(marked_path):
    with open(marked_path) as handle:
        data = json.load(handle)
    shape_dicts = data['shapes']
    return shape_dicts


def create_binary_masks(img, shape_dicts):
    blank = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.int32)
    for shape in shape_dicts:
        points = np.array(shape['points'], dtype=np.int32)
        cv.fillPoly(blank, [points], COLORS[np.random.choice(len(COLORS))])  
    return blank


def plot_pair(images, gray=False):
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(10,8))
    i = 0
    for y in range(2):
        if gray:
            axes[y].imshow(images[i], cmap='gray')
        else:
            axes[y].imshow(images[i])
        axes[y].get_xaxis().set_visible(False)
        axes[y].get_yaxis().set_visible(False)
        i += 1
    plt.show()


