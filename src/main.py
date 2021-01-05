from model import * 
from utils import *
import tensorflow as tf

def main():

    # Data augmentation
    train_gen_args = dict(rotation_range=30,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest')

    val_gen_args = dict(rotation_range=40,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest')

    trainGene = data_augmentation(4, '../dataset/train/', 'img', 'mask', 
                            train_gen_args, save_to_dir=None, save_format=None)
    
    valGene = data_augmentation(4, '../dataset/val/', 'img', 'mask', 
                            val_gen_args, save_to_dir=None, save_format=None)

    # U-net architecture
    # model = unet_paper((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNLES))
    model = unet_test((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNLES))
    
    # Modelcheckpoint
    checkpointer = ModelCheckpoint('../checkpoints/128x128x3_test_ep900_step_1000_vstep_300.h5', verbose=1, save_best_only=True)
    callbacks = [EarlyStopping(patience=8, monitor='val_loss'),
                TensorBoard(log_dir='../logs'),
                checkpointer]
    
    # Training
    model.fit(trainGene, epochs=900,
                         steps_per_epoch=1000,
                         validation_data=valGene,
                         validation_steps=300, 
                         callbacks=callbacks)

    # Save model and renaming
    model.save('../saved_models/128x128x3_unet_test_ep900_step_1000_vstep_300.h5')
    # rename_all('../saved_models/', '.h5')
    # rename_all('../checkpoints/', '.h5')

    
if __name__ == '__main__':
    # resize_images('../dataset', IMG_WIDTH, IMG_HEIGHT)
    # generate_masks()
    main()
    

