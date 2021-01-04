from model import * 
from utils import *
import tensorflow as tf

def main():

    # Data augmentation
    data_gen_args = dict(rotation_range=45,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest')
    trainGene = trainGenerator(2, '../', 'dataset', 'masks', 
                            data_gen_args, save_to_dir=None, save_format=None)
    
    # U-net architecture
    # model = unet_paper((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNLES))
    model = unet_test((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNLES))
    
    # Modelcheckpoint
    checkpointer = ModelCheckpoint('banana_unet.h5', verbose=1, save_best_only=True)
    callbacks = [EarlyStopping(patience=3, monitor='loss'),
                TensorBoard(log_dir='../logs'),
                checkpointer]
    
    # Training
    model.fit(trainGene, epochs=5, steps_per_epoch=700, callbacks=callbacks)

    
if __name__ == '__main__':
    # resize_images('../dataset', IMG_WIDTH, IMG_HEIGHT)
    # resize_images('../masks', IMG_WIDTH, IMG_HEIGHT)
    generate_masks()
    # main()
    

