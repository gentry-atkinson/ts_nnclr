#Author: Gentry Atkinson
#Organization: Texas University
#Data: 24 May, 2022
#Build and trained a self-supervise feature extractor using
#  nearest neighbor clr

#Source: https://keras.io/examples/vision/nnclr/

import tensorflow as tf
import numpy as np
from tensorflow import keras
from utils.augmentation import  rand_signal_drop, time_shift

width = 128
input_shape = (128,1)
kernel_size = 16

temperature = 0.1
queue_size = 1000

contrastive_augmenter = {
    "name": "contrastive_augmenter",
    "drop_chance": 0.1,
    "shift": 10
}
classification_augmenter = {
    "name": "classification_augmenter",
    "drop_chance": 0.1,
    "shift": 10
}

# class RandomResizedCrop(layers.Layer):
#     def __init__(self, scale, ratio):
#         super(RandomResizedCrop, self).__init__()
#         self.scale = scale
#         self.log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))

#     def call(self, images):
#         batch_size = tf.shape(images)[0]
#         height = tf.shape(images)[1]
#         width = tf.shape(images)[2]

#         random_scales = tf.random.uniform((batch_size,), self.scale[0], self.scale[1])
#         random_ratios = tf.exp(
#             tf.random.uniform((batch_size,), self.log_ratio[0], self.log_ratio[1])
#         )

#         new_heights = tf.clip_by_value(tf.sqrt(random_scales / random_ratios), 0, 1)
#         new_widths = tf.clip_by_value(tf.sqrt(random_scales * random_ratios), 0, 1)
#         height_offsets = tf.random.uniform((batch_size,), 0, 1 - new_heights)
#         width_offsets = tf.random.uniform((batch_size,), 0, 1 - new_widths)

#         bounding_boxes = tf.stack(
#             [
#                 height_offsets,
#                 width_offsets,
#                 height_offsets + new_heights,
#                 width_offsets + new_widths,
#             ],
#             axis=1,
#         )
#         images = tf.image.crop_and_resize(
#             images, bounding_boxes, tf.range(batch_size), (height, width)
#         )
#         return images

# def random_invert(factor=0.5):
#   return layers.Lambda(lambda x: random_invert_img(x, factor))

# random_invert = random_invert()

def random_dropout(chance=0.1):
    return keras.layers.Lambda(lambda x: rand_signal_drop(x, chance))

random_dropout = random_dropout()


class RandomDropout(keras.layers.Layer):
        def __init__(self, drop_chance):
            super(RandomDropout, self).__init__()
            self.drop_chance = drop_chance

        def call(self, signals):
            return random_dropout(x)


# class TimeShift(keras.layers.Layer):
#         def __init__(self, shift):
#             super(TimeShift, self).__init__()
#             self.shift = shift

#         def call(self, signals):
#             return [time_shift(s, self.shift) for s in signals]


def augmenter(name='None', drop_chance=0.1, shift=10):
    return keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.RandomFlip(mode='vertical')
            #RandomDropout(drop_chance=drop_chance),
            #TimeShift(shift=shift)
        ],
        name=name,
    )


def encoder():
    return keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv1D(64, kernel_size, activation='relu', data_format='channels_last'),
            keras.layers.Conv1D(64, kernel_size, activation='relu'),
            keras.layers.MaxPooling1D(kernel_size),
            keras.layers.Flatten(),
            keras.layers.Dense(width, activation='relu')
        ],
        name="encoder",
    )

class NNCLR(keras.Model):
    def __init__(
        self, temperature, queue_size,
    ):
        super(NNCLR, self).__init__()
        self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.correlation_accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.contrastive_augmenter = augmenter(**contrastive_augmenter)
        self.classification_augmenter = augmenter(**classification_augmenter)
        self.encoder = encoder()
        self.projection_head = keras.Sequential(
            [
                keras.layers.Input(shape=(width,)),
                keras.layers.Dense(width, activation="relu"),
                keras.layers.Dense(width),
            ],
            name="projection_head",
        )
        self.linear_probe = keras.Sequential(
            [keras.layers.Input(shape=(width,)), keras.layers.Dense(10)], name="linear_probe"
        )
        self.temperature = temperature

        feature_dimensions = self.encoder.output_shape[1]
        self.feature_queue = tf.Variable(
            tf.math.l2_normalize(
                tf.random.normal(shape=(queue_size, feature_dimensions)), axis=1
            ),
            trainable=False,
        )

    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        super(NNCLR, self).compile(**kwargs)
        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer

    def nearest_neighbour(self, projections):
        support_similarities = tf.matmul(
            projections, self.feature_queue, transpose_b=True
        )
        nn_projections = tf.gather(
            self.feature_queue, tf.argmax(support_similarities, axis=1), axis=0
        )
        return projections + tf.stop_gradient(nn_projections - projections)

    def update_contrastive_accuracy(self, features_1, features_2):
        features_1 = tf.math.l2_normalize(features_1, axis=1)
        features_2 = tf.math.l2_normalize(features_2, axis=1)
        similarities = tf.matmul(features_1, features_2, transpose_b=True)

        batch_size = tf.shape(features_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities, tf.transpose(similarities)], axis=0),
        )

    def update_correlation_accuracy(self, features_1, features_2):
        features_1 = (
            features_1 - tf.reduce_mean(features_1, axis=0)
        ) / tf.math.reduce_std(features_1, axis=0)
        features_2 = (
            features_2 - tf.reduce_mean(features_2, axis=0)
        ) / tf.math.reduce_std(features_2, axis=0)

        #batch_size = tf.shape(features_1, out_type=tf.float32)[0] original line
        batch_size = tf.shape(features_1, out_type=tf.int32)[0]
        print(type(features_1))
        print(type(features_2))
        cross_correlation = (
            tf.matmul(
                tf.cast(features_1, dtype=tf.int32), tf.cast(features_2, dtype=tf.int32), transpose_a=True) / batch_size
        )

        feature_dim = tf.shape(features_1)[1]
        correlation_labels = tf.range(feature_dim)
        self.correlation_accuracy.update_state(
            tf.concat([correlation_labels, correlation_labels], axis=0),
            tf.concat([cross_correlation, tf.transpose(cross_correlation)], axis=0),
        )

    def contrastive_loss(self, projections_1, projections_2):
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)

        similarities_1_2_1 = (
            tf.matmul(
                self.nearest_neighbour(projections_1), projections_2, transpose_b=True
            )
            / self.temperature
        )
        similarities_1_2_2 = (
            tf.matmul(
                projections_2, self.nearest_neighbour(projections_1), transpose_b=True
            )
            / self.temperature
        )

        similarities_2_1_1 = (
            tf.matmul(
                self.nearest_neighbour(projections_2), projections_1, transpose_b=True
            )
            / self.temperature
        )
        similarities_2_1_2 = (
            tf.matmul(
                projections_1, self.nearest_neighbour(projections_2), transpose_b=True
            )
            / self.temperature
        )

        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        loss = keras.losses.sparse_categorical_crossentropy(
            tf.concat(
                [
                    contrastive_labels,
                    contrastive_labels,
                    contrastive_labels,
                    contrastive_labels,
                ],
                axis=0,
            ),
            tf.concat(
                [
                    similarities_1_2_1,
                    similarities_1_2_2,
                    similarities_2_1_1,
                    similarities_2_1_2,
                ],
                axis=0,
            ),
            from_logits=True,
        )

        self.feature_queue.assign(
            tf.concat([projections_1, self.feature_queue[:-batch_size]], axis=0)
        )
        return loss

    def train_step(self, data):
        #(unlabeled_images, _), (labeled_images, labels) = data
        #images = tf.concat((unlabeled_images, labeled_images), axis=0)
        augmented_images_1 = self.contrastive_augmenter(data)
        augmented_images_2 = self.contrastive_augmenter(data)

        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1)
            features_2 = self.encoder(augmented_images_2)
            projections_1 = self.projection_head(features_1)
            projections_2 = self.projection_head(features_2)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.update_contrastive_accuracy(features_1, features_2)
        self.update_correlation_accuracy(features_1, features_2)
        preprocessed_images = self.classification_augmenter(labeled_images)

        with tf.GradientTape() as tape:
            features = self.encoder(preprocessed_images)
            class_logits = self.linear_probe(features)
            probe_loss = self.probe_loss(labels, class_logits)
        gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        self.probe_optimizer.apply_gradients(
            zip(gradients, self.linear_probe.trainable_weights)
        )
        self.probe_accuracy.update_state(labels, class_logits)

        return {
            "c_loss": contrastive_loss,
            "c_acc": self.contrastive_accuracy.result(),
            "r_acc": self.correlation_accuracy.result(),
            "p_loss": probe_loss,
            "p_acc": self.probe_accuracy.result(),
        }

    def test_step(self, data):
        labeled_images, labels = data

        preprocessed_images = self.classification_augmenter(
            labeled_images, training=False
        )
        features = self.encoder(preprocessed_images, training=False)
        class_logits = self.linear_probe(features, training=False)
        probe_loss = self.probe_loss(labels, class_logits)

        self.probe_accuracy.update_state(labels, class_logits)
        return {"p_loss": probe_loss, "p_acc": self.probe_accuracy.result()}

def get_features_for_set(X, with_visual=False, with_summary=False):
    global temperature
    global queue_size
    global width
    global input_shape

    width = len(X[0])
    input_shape = (width,1)

    nnclr = NNCLR(temperature, queue_size)
    opti = keras.optimizers.Adam()
    nnclr.compile(contrastive_optimizer=opti, probe_optimizer=opti,loss='mse')
    # unlabeled_train_dataset = tf.convert_to_tensor(X)
    # labels = tf.convert_to_tensor(np.zeros(len(X), dtype='int16'))
    # labeled_train_dataset = tf.data.Dataset.zip(
    #     tf.data.Dataset.from_tensors(unlabeled_train_dataset), tf.data.Dataset.from_tensors(labels)
    # ).prefetch(buffer_size=tf.data.AUTOTUNE)
    # zipped_X = tf.data.Dataset.zip(
    #     (tf.data.Dataset.from_tensors(unlabeled_train_dataset), tf.data.Dataset.from_tensors(labeled_train_dataset))
    # ).prefetch(buffer_size=tf.data.AUTOTUNE)
    nnclr.fit(X, epochs=5)
