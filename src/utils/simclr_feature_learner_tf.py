#Author: Gentry Atkinson
#Organization: Texas University
#Data: 14 June, 2022
#Build and trained a self-supervised feature extractor using
#  SimCLR, following:
#  https://github.com/iantangc/ContrastiveLearningHAR

# @article{tang2020exploring,
#   title={Exploring Contrastive Learning in Human Activity Recognition for Healthcare},
#   author={Tang, Chi Ian and Perez-Pozuelo, Ignacio and Spathis, Dimitris and Mascolo, Cecilia},
#   journal={arXiv preprint arXiv:2011.11542},
#   year={2020}
# }

import numpy as np
import tensorflow as tf
import sklearn.metrics
import scipy

from utils.har_clr.transformations import rotation_transform_vectorized, generate_composite_transform_function_simple
from utils.har_clr.simclr_models import create_base_model, attach_simclr_head, simclr_train_model

def get_features_for_set(X, y=None, with_visual=False, with_summary=False):
    batch_size = 512
    decay_steps = 1000
    epochs = 200
    temperature = 0.1
    transform_funcs = [
        # transformations.scaling_transform_vectorized, # Use Scaling trasnformation
        rotation_transform_vectorized # Use rotation trasnformation
    ]
    transformation_function = generate_composite_transform_function_simple(transform_funcs)

    tf.keras.backend.set_floatx('float32')

    lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=0.1, decay_steps=decay_steps)
    optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)
    # transformation_function = simclr_utitlities.generate_combined_transform_function(trasnform_funcs_vectorized, indices=trasnformation_indices)

    base_model = create_base_model(X[0].shape, model_name="base_model")
    simclr_model = attach_simclr_head(base_model)
    simclr_model.summary()

    trained_simclr_model, epoch_losses = simclr_train_model(simclr_model, np_train[0], optimizer, batch_size, transformation_function, temperature=temperature, epochs=epochs, is_trasnform_function_vectorized=True, verbose=1)

    #simclr_model_save_path = f"{working_directory}{start_time_str}_simclr.hdf5"
    #trained_simclr_model.save(simclr_model_save_path)