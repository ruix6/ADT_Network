"""experiment for the ADT Network ablation study."""
import glob
import json
import logging
import os
# set gpu private
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
import tensorflow as tf


import numpy as np
import sys
# add base path to sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from task2_regression.models.adt_env import ertnetEnv, pearson_loss, pearson_metric, pearson_tf_non_averaged
from util.dataset_generator import DataGenerator, create_tf_dataset


def evaluate_model(model, test_dict):
    """Evaluate a model.

    Parameters
    ----------
    model: tf.keras.Model
        Model to evaluate.
    test_dict: dict
        Mapping between a subject and a tf.data.Dataset containing the test
        set for the subject.

    Returns
    -------
    dict
        Mapping between a subject and the loss/evaluation score on the test set
    """
    evaluation = {}
    for subject, ds_test in test_dict.items():
        logging.info(f"Scores for subject {subject}:")
        results = model.evaluate(ds_test, verbose=2)

        metrics = model.metrics_names
        evaluation[subject] = dict(zip(metrics, results))

    return evaluation



if __name__ == "__main__":
    # Parameters
    # Length of the decision window
    fs= 64
    window_length = 5 * fs  # 10 seconds
    # Hop length between two consecutive decision windows
    hop_length = 1*fs
    epochs = 100
    patience = 10
    batch_size = 64


   # Get the path to the config gile
    experiments_folder = os.path.dirname(__file__)
    task_folder = os.path.dirname(experiments_folder)
    util_folder = os.path.join(os.path.dirname(task_folder), "util")
    config_path = os.path.join(util_folder, 'config.json')

    # Load the config
    with open(config_path) as fp:
        config = json.load(fp)

    # Provide the path of the dataset
    # which is split already to train, val, test

    data_folder = os.path.join('./',  config["split_folder"])
    stimulus_features = ["envelope"]
    features = ["eeg"] + stimulus_features

    # Create a directory to store (intermediate) results
    results_folder = os.path.join(experiments_folder, f"results_ertnet_env_mask_best_ablation")
    os.makedirs(results_folder, exist_ok=True)

    # Evaluate the model on test set
    # Create a dataset generator for each test subject
    test_files = [x for x in glob.glob(os.path.join(data_folder, "test_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
    # Get all different subjects from the test set
    subjects = list(set([os.path.basename(x).split("_-_")[1] for x in test_files]))
    datasets_test = {}
    # Create a generator for each subject
    for sub in subjects:
        files_test_sub = [f for f in test_files if sub in os.path.basename(f)]
        test_generator = DataGenerator(files_test_sub, window_length)
        datasets_test[sub] = create_tf_dataset(test_generator, window_length, None, hop_length, batch_size=64, data_types=(tf.float32, tf.float32), feature_dims=(64, 1))


    # create the model
    model = ertnetEnv(chans=64, outputDims=1, F=8, T=16, D=4, heads=4, ff_dim=128, blocks=4, mask=True, use_bias=False, lrate=0.5)
    model.build(input_shape=(None, 320, 64))
    model.compile(tf.keras.optimizers.Adam(learning_rate=0.001), loss=pearson_loss, metrics=[pearson_metric])

    # #ablation study
    # for i in range(8):
    #     model.load_weights('task2_regression/experiments/results_ertnet_env_mask_best/model.h5')

    #     temp_conv = model.layers[0].get_weights()
    #     temp_conv[0][:, :, :, i] = 0.
    #     model.layers[0].set_weights(temp_conv)
        

    #     evaluation = evaluate_model(model, datasets_test)

    #     # We can save our results in a json encoded file
    #     results_path = os.path.join(results_folder, f'eval_temp{i}.json')
    #     with open(results_path, "w") as fp:
    #         json.dump(evaluation, fp)
    #     logging.info(f"Results saved at {results_path}")
        #ablation study
    model.load_weights('task2_regression/experiments/results_ertnet_env_mask_best/model.h5')

    temp_conv = model.layers[0].get_weights()
    temp_conv[0][:, :, :, [4, 6]] = 0.
    model.layers[0].set_weights(temp_conv)
    

    evaluation = evaluate_model(model, datasets_test)

    # We can save our results in a json encoded file
    results_path = os.path.join(results_folder, 'eval_temp46.json')
    with open(results_path, "w") as fp:
        json.dump(evaluation, fp)
    logging.info(f"Results saved at {results_path}")
