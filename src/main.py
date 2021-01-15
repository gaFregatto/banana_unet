from model import * 
from utils import *

def main():
    # Data augmentation
    train_gen_args = dict(rotation_range=40,
                    width_shift_range=0.45,
                    height_shift_range=0.45,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest')

    val_gen_args = dict(rotation_range=20,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest')

    trainGene = data_augmentation(BATCH_TRAIN, '../dataset/train/', 'img', 'mask', 
                            train_gen_args, save_to_dir=None, save_format=None)
    
    valGene = data_augmentation(BATCH_VAL, '../dataset/val/', 'img', 'mask', 
                            val_gen_args, save_to_dir=None, save_format=None)

    # U-net architecture
    # model = unet_reg((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    model = unet_test((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    
    # Modelcheckpoints
    checkpointer = ModelCheckpoint(os.path.join('../checkpoints', FILE), verbose=1, save_best_only=True)
    csv_logger = CSVLogger(os.path.join('../csv_logs', FILE_CSV), separator=',', append=False)
    early_stop = EarlyStopping(patience=5, monitor='val_loss')
    tensor_board = TensorBoard(log_dir='../logs')
    callbacks = [checkpointer, csv_logger, tensor_board, early_stop]
    
    # Training
    history = model.fit(trainGene, epochs=EP,
                         steps_per_epoch=STEP,
                         validation_data=valGene,
                         validation_steps=V_STEP, 
                         callbacks=callbacks)

    # Save model 
    model.save(os.path.join('../saved_models', FILE))
    
    # Evaluating 
    print("Training evaluate")
    train_loss, train_acc = model.evaluate(trainGene, steps=E_STEP)
    print(f"Train accuracy - loss: {train_acc} - {train_loss}")
    
    print("Validation evaluate")
    val_loss, val_acc = model.evaluate(valGene, steps=E_STEP)
    print(f"Validation accuracy - loss: {val_acc} - {val_loss}")

    # Predicting results
    testGene = testGenerator('../dataset/test', gray_flag=True)
    results = model.predict(testGene, 40, verbose=1)
    saveResult("../dataset/test",results)

    # Plotting
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig(os.path.join('../plots', FILE_PNG))
    # plt.show()


def retrain():
    train_gen_args = dict(rotation_range=40,
                    width_shift_range=0.45,
                    height_shift_range=0.45,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest')

    val_gen_args = dict(rotation_range=20,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest')

    trainGene = data_augmentation(BATCH_TRAIN, '../dataset/train/', 'img', 'mask', 
                            train_gen_args, save_to_dir=None, save_format=None)
    
    valGene = data_augmentation(BATCH_VAL, '../dataset/val/', 'img', 'mask', 
                            val_gen_args, save_to_dir=None, save_format=None)
    
    model = load_model('../saved_models/test_256x256x1_ep25_step400_vstep150.h5')
    
    results = model.evaluate(valGene, steps=E_STEP)
    print("validation loss and accuracy are", results)

    checkpointer = ModelCheckpoint(os.path.join('../checkpoints', FILE), verbose=1, save_best_only=True)
    csv_logger = CSVLogger(os.path.join('../csv_logs', FILE_CSV), separator=',', append=True)
    early_stop = EarlyStopping(patience=5, monitor='val_loss')
    tensor_board = TensorBoard(log_dir='../logs')
    callbacks = [checkpointer, csv_logger, tensor_board, early_stop]

    history = model.fit(trainGene, epochs=EP,
                         steps_per_epoch=STEP,
                         validation_data=valGene,
                         validation_steps=V_STEP, 
                         callbacks=callbacks)

    model.save(os.path.join('../retrained_models', FILE))


def gen_results():
    model = load_model('../checkpoints/F_test_256x256x1_ep30_step300_vstep60_bt8_bv2.h5')
    # model = load_model('../saved_models/F_test_256x256x1_ep30_step300_vstep60_bt8_bv2.h5')
    
    testGene = testGenerator('../dataset/test', gray_flag=True)
    results = model.predict(testGene, 40, verbose=1)
    saveResult("../dataset/test",results)
    
    # set_trace()


if __name__ == '__main__':
    # Para treinar um novo modelo
    main()
    
    # Para continuar treinando um modelo salvo
    # retrain()

    # Para gerar resultados a partir de um modelo salvo
    # gen_results()
    

