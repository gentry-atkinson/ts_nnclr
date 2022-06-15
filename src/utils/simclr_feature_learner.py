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

def create_base_model(input_shape, model_name="base_model"):
    """
    Create the base model for activity recognition
    Reference (TPN model):
        Saeed, A., Ozcelebi, T., & Lukkien, J. (2019). Multi-task self-supervised learning for human activity detection. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 3(2), 1-30.
    Architecture:
        Input
        -> Conv 1D: 32 filters, 24 kernel_size, relu, L2 regularizer
        -> Dropout: 10%
        -> Conv 1D: 64 filters, 16 kernel_size, relu, L2 regularizer
        -> Dropout: 10%
        -> Conv 1D: 96 filters, 8 kernel_size, relu, L2 regularizer
        -> Dropout: 10%
        -> Global Maximum Pooling 1D
    
    Parameters:
        input_shape
            the input shape for the model, should be (window_size, num_channels)
    
    Returns:
        model (tf.keras.Model)
    """

    inputs = tf.keras.Input(shape=input_shape, name='input')
    x = inputs
    x = tf.keras.layers.Conv1D(
            32, 24,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4)
        )(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Conv1D(
            64, 16,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
        )(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    x = tf.keras.layers.Conv1D(
        96, 8,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
        )(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    x = tf.keras.layers.GlobalMaxPool1D(data_format='channels_last', name='global_max_pooling1d')(x)

    return tf.keras.Model(inputs, x, name=model_name)

def attach_simclr_head(base_model, hidden_1=256, hidden_2=128, hidden_3=50):
    """
    Attach a 3-layer fully-connected encoding head
    Architecture:
        base_model
        -> Dense: hidden_1 units
        -> ReLU
        -> Dense: hidden_2 units
        -> ReLU
        -> Dense: hidden_3 units
    """

    input = base_model.input
    x = base_model.output

    projection_1 = tf.keras.layers.Dense(hidden_1)(x)
    projection_1 = tf.keras.layers.Activation("relu")(projection_1)
    projection_2 = tf.keras.layers.Dense(hidden_2)(projection_1)
    projection_2 = tf.keras.layers.Activation("relu")(projection_2)
    projection_3 = tf.keras.layers.Dense(hidden_3)(projection_2)

    simclr_model = tf.keras.Model(input, projection_3, name= base_model.name + "_simclr")

    return simclr_model


def create_linear_model_from_base_model(base_model, output_shape, intermediate_layer=7):

    """
    Create a linear classification model from the base mode, using activitations from an intermediate layer
    Architecture:
        base_model-intermediate_layer
        -> Dense: output_shape units
        -> Softmax
    
    Optimizer: SGD
    Loss: CategoricalCrossentropy
    Parameters:
        base_model
            the base model from which the activations are extracted
        
        output_shape
            number of output classifiction categories
        intermediate_layer
            the index of the intermediate layer from which the activations are extracted
    
    Returns:
        trainable_model (tf.keras.Model)
    """

    inputs = base_model.inputs
    x = base_model.layers[intermediate_layer].output
    x = tf.keras.layers.Dense(output_shape, kernel_initializer=tf.random_normal_initializer(stddev=.01))(x)
    outputs = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=base_model.name + "linear")

    for layer in model.layers[:intermediate_layer+1]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.03),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"), tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )
    return model


def create_full_classification_model_from_base_model(base_model, output_shape, model_name="TPN", intermediate_layer=7, last_freeze_layer=4):
    """
    Create a full 2-layer classification model from the base mode, using activitations from an intermediate layer with partial freezing
    Architecture:
        base_model-intermediate_layer
        -> Dense: 1024 units
        -> ReLU
        -> Dense: output_shape units
        -> Softmax
    
    Optimizer: Adam
    Loss: CategoricalCrossentropy
    Parameters:
        base_model
            the base model from which the activations are extracted
        
        output_shape
            number of output classifiction categories
        model_name
            name of the output model
        intermediate_layer
            the index of the intermediate layer from which the activations are extracted
        last_freeze_layer
            the index of the last layer to be frozen for fine-tuning (including the layer with the index)
    
    Returns:
        trainable_model (tf.keras.Model)
    """

    # inputs = base_model.inputs
    intermediate_x = base_model.layers[intermediate_layer].output

    x = tf.keras.layers.Dense(1024, activation='relu')(intermediate_x)
    x = tf.keras.layers.Dense(output_shape)(x)
    outputs = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=base_model.inputs, outputs=outputs, name=model_name)

    for layer in model.layers:
        layer.trainable = False
    
    for layer in model.layers[last_freeze_layer+1:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"), tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    return model


def extract_intermediate_model_from_base_model(base_model, intermediate_layer=7):
    """
    Create an intermediate model from base mode, which outputs embeddings of the intermediate layer
    Parameters:
        base_model
            the base model from which the intermediate model is built
        
        intermediate_layer
            the index of the intermediate layer from which the activations are extracted
    Returns:
        model (tf.keras.Model)
    """

    model = tf.keras.Model(inputs=base_model.inputs, outputs=base_model.layers[intermediate_layer].output, name=base_model.name + "_layer_" + str(intermediate_layer))
    return model

def get_mode(np_array):
    """
    Get the mode (majority/most frequent value) from a 1D array
    """
    return scipy.stats.mode(np_array)[0]

def sliding_window_np(X, window_size, shift, stride, offset=0, flatten=None):
    """
    Create sliding windows from an ndarray
    Parameters:
    
        X (numpy-array)
            The numpy array to be windowed
        
        shift (int)
            number of timestamps to shift for each window
            (200 here refers to 50% overlap, no overlap if =400)
        stride (int)
            stride of the window (dilation)
        offset (int)
            starting index of the first window
        
        flatten (function (array) -> (value or array) )
            the function to be applied to a window after it is extracted
            can be used with get_mode (see above) for extracting the label by majority voting
            ignored if is None
    Return:
        Windowed ndarray
            shape[0] is the number of windows
    """

    overall_window_size = (window_size - 1) * stride + 1
    num_windows = (X.shape[0] - offset - (overall_window_size)) // shift + 1
    windows = []
    for i in range(num_windows):
        start_index = i * shift + offset
        this_window = X[start_index : start_index + overall_window_size : stride]
        if flatten is not None:
            this_window = flatten(this_window)
        windows.append(this_window)
    return np.array(windows)

def get_windows_dataset_from_user_list_format(user_datasets, window_size=400, shift=200, stride=1, verbose=0):
    """
    Create windows dataset in 'user-list' format using sliding windows
    Parameters:
        user_datasets
            dataset in the 'user-list' format {user_id: [(sensor_values, activity_labels)]}
        
        window_size = 400
            size of the window (output)
        shift = 200
            number of timestamps to shift for each window
            (200 here refers to 50% overlap, no overlap if =400)
        stride = 1
            stride of the window (dilation)
        verbose = 0
            debug messages are printed if > 0
    
    Return:
        user_dataset_windowed
            Windowed version of the user_datasets
            Windows from different trials are combined into one array
            type: {user_id: ( windowed_sensor_values, windowed_activity_labels)}
            windowed_sensor_values have shape (num_window, window_size, channels)
            windowed_activity_labels have shape (num_window)
            Labels are decided by majority vote
    """
    
    user_dataset_windowed = {}

    for user_id in user_datasets:
        if verbose > 0:
            print(f"Processing {user_id}")
        x = []
        y = []

        # Loop through each trail of each user
        for v,l in user_datasets[user_id]:
            v_windowed = sliding_window_np(v, window_size, shift, stride)
            
            # flatten the window by majority vote (1 value for each window)
            l_flattened = sliding_window_np(l, window_size, shift, stride, flatten=get_mode)
            if len(v_windowed) > 0:
                x.append(v_windowed)
                y.append(l_flattened)
            if verbose > 0:
                print(f"Data: {v_windowed.shape}, Labels: {l_flattened.shape}")

        # combine all trials
        user_dataset_windowed[user_id] = (np.concatenate(x), np.concatenate(y).squeeze())
    return user_dataset_windowed

def combine_windowed_dataset(user_datasets_windowed, train_users, test_users=None, verbose=0):
    """
    Combine a windowed 'user-list' dataset into training and test sets
    Parameters:
        user_dataset_windowed
            dataset in the windowed 'user-list' format {user_id: ( windowed_sensor_values, windowed_activity_labels)}
        
        train_users
            list or set of users (corresponding to the user_id) to be used as training data
        test_users = None
            list or set of users (corresponding to the user_id) to be used as testing data
            if is None, then all users not in train_users will be treated as test users 
        verbose = 0
            debug messages are printed if > 0
    Return:
        (train_x, train_y, test_x, test_y)
            train_x, train_y
                the resulting training/test input values as a single numpy array
            test_x, test_y
                the resulting training/test labels as a single (1D) numpy array
    """
    
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for user_id in user_datasets_windowed:
        
        v,l = user_datasets_windowed[user_id]
        if user_id in train_users:
            if verbose > 0:
                print(f"{user_id} Train")
            train_x.append(v)
            train_y.append(l)
        elif test_users is None or user_id in test_users:
            if verbose > 0:
                print(f"{user_id} Test")
            test_x.append(v)
            test_y.append(l)
    

    if len(train_x) == 0:
        train_x = np.array([])
        train_y = np.array([])
    else:
        train_x = np.concatenate(train_x)
        train_y = np.concatenate(train_y).squeeze()
    
    if len(test_x) == 0:
        test_x = np.array([])
        test_y = np.array([])
    else:
        test_x = np.concatenate(test_x)
        test_y = np.concatenate(test_y).squeeze()

    return train_x, train_y, test_x, test_y

def get_mean_std_from_user_list_format(user_datasets, train_users):
    """
    Obtain and means and standard deviations from a 'user-list' dataset (channel-wise)
    from training users only
    Parameters:
        user_datasets
            dataset in the 'user-list' format {user_id: [(sensor_values, activity_labels)]}
        
        train_users
            list or set of users (corresponding to the user_ids) from which the mean and std are extracted
    Return:
        (means, stds)
            means and stds of the particular users (channel-wise)
            shape: (num_channels)
    """
    
    mean_std_data = []
    for u in train_users:
        for data, _ in user_datasets[u]:
            mean_std_data.append(data)
    mean_std_data_combined = np.concatenate(mean_std_data)
    means = np.mean(mean_std_data_combined, axis=0)
    stds = np.std(mean_std_data_combined, axis=0)
    return (means, stds)

def normalise(data, mean, std):
    """
    Normalise data (Z-normalisation)
    """

    return ((data - mean) / std)

def apply_label_map(y, label_map):
    """
    Apply a dictionary mapping to an array of labels
    Can be used to convert str labels to int labels
    Parameters:
        y
            1D array of labels
        label_map
            a label dictionary of (label_original -> label_new)
    Return:
        y_mapped
            1D array of mapped labels
            None values are present if there is no entry in the dictionary
    """

    y_mapped = []
    for l in y:
        y_mapped.append(label_map.get(l))
    return np.array(y_mapped)


def filter_none_label(X, y):
    """
    Filter samples of the value None
    Can be used to exclude non-mapped values from apply_label_map
    Parameters:
        X
            data values
        y
            labels (1D)
    Return:
        (X_filtered, y_filtered)
            X_filtered
                filtered data values
            
            y_filtered
                filtered labels (of type int)
    """

    valid_mask = np.where(y != None)
    return (np.array(X[valid_mask]), np.array(y[valid_mask], dtype=int))

def pre_process_dataset_composite(user_datasets, label_map, output_shape, train_users, test_users, window_size, shift, normalise_dataset=True, validation_split_proportion=0.2, verbose=0):
    """
    A composite function to process a dataset
    Steps
        1: Use sliding window to make a windowed dataset (see get_windows_dataset_from_user_list_format)
        2: Split the dataset into training and test set (see combine_windowed_dataset)
        3: Normalise the datasets (see get_mean_std_from_user_list_format)
        4: Apply the label map and filter labels (see apply_label_map, filter_none_label)
        5: One-hot encode the labels (see tf.keras.utils.to_categorical)
        6: Split the training set into training and validation sets (see sklearn.model_selection.train_test_split)
    
    Parameters:
        user_datasets
            dataset in the 'user-list' format {user_id: [(sensor_values, activity_labels)]}
        label_map
            a mapping of the labels
            can be used to filter labels
            (see apply_label_map and filter_none_label)
        output_shape
            number of output classifiction categories
            used in one hot encoding of the labels
            (see tf.keras.utils.to_categorical)
        train_users
            list or set of users (corresponding to the user_id) to be used as training data
        test_users
            list or set of users (corresponding to the user_id) to be used as testing data
        window_size
            size of the data windows
            (see get_windows_dataset_from_user_list_format)
        shift
            number of timestamps to shift for each window
            (see get_windows_dataset_from_user_list_format)
        normalise_dataset = True
            applies Z-normalisation if True
        validation_split_proportion = 0.2
            if not None, the proportion for splitting the full training set further into training and validation set using random sampling
            (see sklearn.model_selection.train_test_split)
            if is None, the training set will not be split - the return value np_val will also be none
        verbose = 0
            debug messages are printed if > 0
    
    Return:
        (np_train, np_val, np_test)
            three pairs of (X, y)
            X is a windowed set of data points
            y is an array of one-hot encoded labels
            if validation_split_proportion is None, np_val is None
    """

    # Step 1
    user_datasets_windowed = get_windows_dataset_from_user_list_format(user_datasets, window_size=window_size, shift=shift)

    # Step 2
    train_x, train_y, test_x, test_y = combine_windowed_dataset(user_datasets_windowed, train_users)

    # Step 3
    if normalise_dataset:
        means, stds = get_mean_std_from_user_list_format(user_datasets, train_users)
        train_x = normalise(train_x, means, stds)
        test_x = normalise(test_x, means, stds)

    # Step 4
    train_y_mapped = apply_label_map(train_y, label_map)
    test_y_mapped = apply_label_map(test_y, label_map)

    train_x, train_y_mapped = filter_none_label(train_x, train_y_mapped)
    test_x, test_y_mapped = filter_none_label(test_x, test_y_mapped)

    if verbose > 0:
        print("Test")
        print(np.unique(test_y, return_counts=True))
        print(np.unique(test_y_mapped, return_counts=True))
        print("-----------------")

        print("Train")
        print(np.unique(train_y, return_counts=True))
        print(np.unique(train_y_mapped, return_counts=True))
        print("-----------------")

    # Step 5
    train_y_one_hot = tf.keras.utils.to_categorical(train_y_mapped, num_classes=output_shape)
    test_y_one_hot = tf.keras.utils.to_categorical(test_y_mapped, num_classes=output_shape)

    r = np.random.randint(len(train_y_mapped))
    assert train_y_one_hot[r].argmax() == train_y_mapped[r]
    r = np.random.randint(len(test_y_mapped))
    assert test_y_one_hot[r].argmax() == test_y_mapped[r]

    # Step 6
    if validation_split_proportion is not None and validation_split_proportion > 0:
        train_x_split, val_x_split, train_y_split, val_y_split = sklearn.model_selection.train_test_split(train_x, train_y_one_hot, test_size=validation_split_proportion, random_state=42)
    else:
        train_x_split = train_x
        train_y_split = train_y_one_hot
        val_x_split = None
        val_y_split = None
        

    if verbose > 0:
        print("Training data shape:", train_x_split.shape)
        print("Validation data shape:", val_x_split.shape if val_x_split is not None else "None")
        print("Testing data shape:", test_x.shape)

    np_train = (train_x_split, train_y_split)
    np_val = (val_x_split, val_y_split) if val_x_split is not None else None
    np_test = (test_x, test_y_one_hot)

    # original_np_train = np_train
    # original_np_val = np_val
    # original_np_test = np_test

    return (np_train, np_val, np_test)

def pre_process_dataset_composite_in_user_format(user_datasets, label_map, output_shape, train_users, window_size, shift, normalise_dataset=True, verbose=0):
    """
    A composite function to process a dataset which outputs processed datasets separately for each user (of type: {user_id: ( windowed_sensor_values, windowed_activity_labels)}).
    This is different from pre_process_dataset_composite where the data from the training and testing users are not combined into one object.
    Steps
        1: Use sliding window to make a windowed dataset (see get_windows_dataset_from_user_list_format)
        For each user:
            2: Apply the label map and filter labels (see apply_label_map, filter_none_label)
            3: One-hot encode the labels (see tf.keras.utils.to_categorical)
            4: Normalise the data (see get_mean_std_from_user_list_format)
    
    Parameters:
        user_datasets
            dataset in the 'user-list' format {user_id: [(sensor_values, activity_labels)]}
        label_map
            a mapping of the labels
            can be used to filter labels
            (see apply_label_map and filter_none_label)
        output_shape
            number of output classifiction categories
            used in one hot encoding of the labels
            (see tf.keras.utils.to_categorical)
        train_users
            list or set of users (corresponding to the user_id) to be used for normalising the dataset
        window_size
            size of the data windows
            (see get_windows_dataset_from_user_list_format)
        shift
            number of timestamps to shift for each window
            (see get_windows_dataset_from_user_list_format)
        normalise_dataset = True
            applies Z-normalisation if True
        verbose = 0
            debug messages are printed if > 0
    
    Return:
        user_datasets_processed
            Processed version of the user_datasets in the windowed format
            type: {user_id: (windowed_sensor_values, windowed_activity_labels)}
    """

    # Preparation for step 2
    if normalise_dataset:
        means, stds = get_mean_std_from_user_list_format(user_datasets, train_users)

    # Step 1
    user_datasets_windowed = get_windows_dataset_from_user_list_format(user_datasets, window_size=window_size, shift=shift)

    
    user_datasets_processed = {}
    for user, user_dataset in user_datasets_windowed.items():
        data, labels = user_dataset

        # Step 2
        labels_mapped = apply_label_map(labels, label_map)
        data_filtered, labels_filtered = filter_none_label(data, labels_mapped)

        # Step 3
        labels_one_hot = tf.keras.utils.to_categorical(labels_filtered, num_classes=output_shape)

        # random check
        r = np.random.randint(len(labels_filtered))
        assert labels_one_hot[r].argmax() == labels_filtered[r]

        # Step 4
        if normalise_dataset:
            data_filtered = normalise(data_filtered, means, stds)

        user_datasets_processed[user] = (data_filtered, labels_one_hot)

        if verbose > 0:
            print("Data shape of user", user, ":", data_filtered.shape)
    
    return user_datasets_processed

def add_user_id_to_windowed_dataset(user_datasets_windowed, encode_user_id=True, as_feature=False, as_label=True, verbose=0):
    """
    Add user ids as features or labels to a windowed dataset
    The user ids are appended to the last dimension of the arrays
    E.g. sensor values of shape (100, 400, 3) will become (100, 400, 4), and data[:, :, -1] will contain the user id
    Similarly labels of shape (100, 5) will become (100, 6), and labels[:, -1] will contain the user id
    
    Parameters:
        user_datasets_windowed
            dataset in the 'windowed-user' format type: {user_id: (windowed_sensor_values, windowed_activity_labels)}
        encode_user_id = True
            whether to encode the user ids as integers
            if True: 
                encode all user ids as integers when being appended to the np arrays
                return the map from user id to integer as an output
                note that the dtype of the output np arrays will be kept as float if they are originally of type float
            if False:
                user ids will be kept as is when being appended to the np arrays
                WARNING: if the user id is of type string, the output arrays will also be converted to type string, which might be difficult to work with
        as_feature = False
            user ids will be added to the windowed_sensor_values arrays as extra features if True
        as_label = False
            user ids will be added to the windowed_activity_labels arrays as extra labels if True
        verbose = 0
            debug messages are printed if > 0
    Return:
        user_datasets_modified, user_id_encoder
            user_datasets_modified
                the modified version of the input (user_datasets_windowed)
                with the same type {user_id: ( windowed_sensor_values, windowed_activity_labels)}
            user_id_encoder
                the encoder which maps user ids to integers
                type: {user_id: encoded_user_id}
                None if encode_user_id is False
    """

    # Create the mapping from user_id to integers
    if encode_user_id:
        all_users = sorted(list(user_datasets_windowed.keys()))
        user_id_encoder = dict([(u, i) for i, u in enumerate(all_users)])
    else:
        user_id_encoder = None

    # if none of the options are enabled, return the input
    if not as_feature and not as_label:
        return user_datasets_windowed, user_id_encoder

    user_datasets_modified = {}
    for user, user_dataset in user_datasets_windowed.items():
        data, labels = user_dataset

        # Get the encoded user_id
        if encode_user_id:
            user_id = user_id_encoder[user]
        else:
            user_id = user

        # Add user_id as an extra feature
        if as_feature:
            user_feature = np.expand_dims(np.full(data.shape[:-1], user_id), axis=-1)
            data_modified = np.append(data, user_feature, axis=-1)
        else:
            data_modified = data
        
        # Add user_id as an extra label
        if as_label:
            user_labels = np.expand_dims(np.full(labels.shape[:-1], user_id), axis=-1)
            labels_modified = np.append(labels, user_labels, axis=-1)
        else:
            labels_modified = labels

        if verbose > 0:
            print(f"User {user}: id {repr(user)} -> {repr(user_id)}, data shape {data.shape} -> {data_modified.shape}, labels shape {labels.shape} -> {labels_modified.shape}")

        user_datasets_modified[user] = (data_modified, labels_modified)
    
    return user_datasets_modified, user_id_encoder

def make_batches_reshape(data, batch_size):
    """
    Make a batched dataset from a windowed time-series by simple reshaping
    Note that the last batch is dropped if incomplete
    Parameters:
        data
            A 3D numpy array in the shape (num_windows, window_size, num_channels)
        batch_size
            the (maximum) size of the batches
    Returns:
        batched_data
            A 4D numpy array in the shape (num_batches, batch_size, window_size, num_channels)
    """

    max_len = (data.shape[0]) // batch_size * batch_size
    return data[:max_len].reshape((-1, batch_size, data.shape[-2], data.shape[-1]))

def np_random_shuffle_index(length):
    """
    Get a list of randomly shuffled indices
    """
    indices = np.arange(length)
    np.random.shuffle(indices)
    return indices

def ceiling_division(n, d):
    """
    Ceiling integer division
    """
    return -(n // -d)

def get_batched_dataset_generator(data, batch_size):
    """
    Create a data batch generator
    Note that the last batch might not be full
    Parameters:
        data
            A numpy array of data
        batch_size
            the (maximum) size of the batches
    Returns:
        generator<numpy array>
            a batch of the data with the same shape except the first dimension, which is now the batch size
    """

    num_bathes = ceiling_division(data.shape[0], batch_size)
    for i in range(num_bathes):
        yield data[i * batch_size : (i + 1) * batch_size]