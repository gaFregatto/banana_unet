from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import matplotlib.pyplot as plt 
from pdb import set_trace
import numpy as np 
import cv2 as cv 
import json 
import os

MODEL = 'F_test'

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 1

BATCH_TRAIN = 8
BATCH_VAL = 2

EP = 30
STEP = 300
V_STEP = 60
E_STEP = 150

FILE = f"{MODEL}_{IMG_WIDTH}x{IMG_HEIGHT}x{IMG_CHANNELS}_ep{EP}_step{STEP}_vstep{V_STEP}_bt{BATCH_TRAIN}_bv{BATCH_VAL}.h5"
FILE_CSV = f"{MODEL}_{IMG_WIDTH}x{IMG_HEIGHT}x{IMG_CHANNELS}_ep{EP}_step{STEP}_vstep{V_STEP}_bt{BATCH_TRAIN}_bv{BATCH_VAL}.csv"
FILE_PNG = f"{MODEL}_{IMG_WIDTH}x{IMG_HEIGHT}x{IMG_CHANNELS}_ep{EP}_step{STEP}_vstep{V_STEP}_bt{BATCH_TRAIN}_bv{BATCH_VAL}.png"

COLORS = [[43, 43, 200], [43, 75, 200], [43, 106, 200], [43, 137, 200], [43, 169, 200], [43, 200, 195], [43, 200, 163], 
            [43, 200, 132], [43, 200, 101], [43, 200, 69], [54, 200, 43], [85, 200, 43], [116, 200, 43], [148, 200, 43], 
            [179, 200, 43], [200, 184, 43], [200, 153, 43], [200, 122, 43], [200, 90, 43], [200, 59, 43], [200, 43, 64], 
            [200, 43, 95], [200, 43, 127], [200, 43, 158], [200, 43, 190], [174, 43, 200], [142, 43, 200], [111, 43, 200], 
            [80, 43, 200], [43, 43, 200]]

def rename_all(directory, extension):
    for i, filename in enumerate(os.listdir(directory)):
        current = directory + filename
        new_name = directory + str(i) + extension 
        os.rename(current, new_name)


def resize_images(directory, width, height):
    for i, img_name in enumerate(os.listdir(directory)):
        if img_name.split('.')[1] == 'jpg':
            img = cv.imread(os.path.join(directory, img_name))
            # img = cv.imread(os.path.join(directory, img_name), 0)
            new_img = np.array(cv.resize(img, (width, height)))
            cv.imwrite(os.path.join(directory, img_name), new_img)


def generate_masks(path_data='../dataset', path_marked='../marked', path_masks='../masks'):
    image_list = sorted(os.listdir(path_data), key=lambda x: int(x.split('.')[0]))
    marked_list = sorted(os.listdir(path_marked), key=lambda x: int(x.split('.')[0]))
    i = 0
    for img, mrk in zip(image_list, marked_list):
        im = cv.imread(os.path.join(path_data, img), 0)
        mrk_path = os.path.join(path_marked, mrk)
        shape_dicts = get_poly(mrk_path)
        binary_img = create_binary_masks(im, shape_dicts)
        # binary_img = create_color_masks(im, shape_dicts)
        mask_path = os.path.join(path_masks, str(i)+'_mask.jpg')
        cv.imwrite(mask_path, binary_img)
        i += 1
        # plot_pair([im, binary_img], gray=True)
        # plt.show()
        # break
        

def get_poly(path_marked):
    with open(path_marked) as file:
        data = json.load(file)
    shape_dicts = data['shapes']
    return shape_dicts


def create_color_masks(img, shape_dicts):
    blank = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.int32)
    for shape in shape_dicts:
        points = np.array(shape['points'], dtype=np.int32)
        cv.fillPoly(blank, [points], COLORS[np.random.choice(len(COLORS))])  
    return blank

def create_binary_masks(img, shape_dicts):
    blank = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    for shape in shape_dicts:
        points = np.array(shape['points'], dtype=np.int32) 
        cv.fillPoly(blank, [points], 255)  
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


def adjustData(img,mask):
    if(np.max(img) > 1):
        img = img/255
        mask = mask/255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)


def data_augmentation(batch_size, folder_path, image_folder, mask_folder, aug_dict, image_color_mode ="grayscale",
                    mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                    save_to_dir=None, save_format=None,  target_size=(IMG_WIDTH,IMG_HEIGHT), seed=1):
   
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        directory=folder_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_format = save_format,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        directory=folder_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_format = save_format,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask)
        yield (img, mask)

def testGenerator(test_path, num_image=40, target_size=(IMG_WIDTH,IMG_HEIGHT), gray_flag=True):

    for i in range(num_image):
        if gray_flag == False:
            img = cv.imread(os.path.join(test_path,"%d.jpg"%i))
            img = img/255
            img = np.array(cv.resize(img, (IMG_WIDTH, IMG_HEIGHT)))
            img = np.reshape(img,(1,)+img.shape)
            yield img
        elif gray_flag == True:
            img = cv.imread(os.path.join(test_path,"%d.jpg"%i), 0)
            img = img/255
            img = np.array(cv.resize(img, (IMG_WIDTH, IMG_HEIGHT)))
            img = np.reshape(img,img.shape+(1,)) if (gray_flag) else img
            img = np.reshape(img,(1,)+img.shape)
            yield img


def saveResult(save_path, data):
    for i, img in enumerate(data): 
        img = img * 255
        cv.imwrite(os.path.join(save_path, "%d_predict.jpg"%i), img)