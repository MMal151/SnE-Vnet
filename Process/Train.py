import os
import tensorflow as tf
import logging
from time import time

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from Loader.DataLoader import load_data
from Model.Unet3D import Unet3D
from Model.Vnet import Vnet
from Process.ProcessUtils import get_loss, get_metrics, get_optimizer, load_model
from Utils.VisualisationUtils import show_history

CLASS_NAME = "[Process/Train]"


def configure_gpus(gpus):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    tf.keras.backend.clear_session()


def train(cfg):
    lgr = CLASS_NAME + "[train()]"

    train_gen, test_gen, valid_gen = load_data(cfg)

    strategy = None
    if cfg["alw_parallel_processing"]:
        configure_gpus(cfg["gpu"])
        strategy = tf.distribute.MirroredStrategy()

    if strategy is not None:
        with strategy.scope():
            fit_model(cfg, train_gen, valid_gen, test_gen)
    else:
        fit_model(cfg, train_gen, valid_gen, test_gen)


def fit_model(cfg, train_gen, valid_gen, test_gen):
    lgr = CLASS_NAME + "[fit_model()]"

    start_time = time()
    model = None
    if cfg["resume_train"]:
        model = load_model(cfg, cfg["load_path"])

    if cfg["model_type"].lower() == "unet" and model is None:
        model = Unet3D(cfg["data"]["input_shape"], cfg["activation"]).generate_model()
    elif cfg["model_type"].lower() == "vnet" and model is None:
        model = Vnet(cfg["data"]["input_shape"], cfg["activation"]).generate_model()

    if model is not None:
        monitor = 'loss'
        validation = False
        if valid_gen is not None:
            monitor = "val_loss"
            validation = True

        metrics = get_metrics(cfg["perf_metrics"])
        model.compile(optimizer=get_optimizer(cfg),
                      loss=get_loss(cfg["loss"].lower()), metrics=metrics.values())

        callbacks = []

        checkpoint = ModelCheckpoint(cfg["save_path"] + "{epoch:02d}.h5", monitor=monitor,
                                     save_best_only=cfg["save_best_only"],
                                     save_freq='epoch')
        callbacks.append(checkpoint)

        if cfg["aply_early_stpng"]:
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.005, mode="min",
                                           restore_best_weights=True)
            callbacks.append(early_stopping)

        history = model.fit(train_gen, validation_data=valid_gen, steps_per_epoch=len(train_gen),
                            epochs=cfg["epochs"], callbacks=callbacks)
        show_history(history, validation)
        logging.info(f"{lgr}: Total Training Time: [{time() - start_time}] seconds")

        if test_gen is not None:
            logging.info(f"{lgr}: Starting testing.")
            results = model.evaluate(test_gen, batch_size=1, steps=len(test_gen))
            log_test_results(metrics.keys(), results)
    else:
        logging.error(f"{lgr}: Invalid model_type. Aborting training process.")


def log_test_results(eval_list, results):
    lgr = CLASS_NAME + "[log_test_results()]"
    logging.info(f"{lgr}: Loss = [{results[0]}]")

    for i, metric in enumerate(eval_list, start=1):
        logging.info(f"{lgr}: {metric} = [{results[i]}]")
