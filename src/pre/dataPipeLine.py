import tensorflow as tf
from pre.videoPreperation import mappable_function


# PATH = "../../data/s1/*.mgp"


def getProcesedData(PATH):
    # Load data files from the specified directory
    data = tf.data.Dataset.list_files(PATH)

    # Shuffle the dataset with a buffer size of 500
    data = data.shuffle(500, reshuffle_each_iteration=False)

    # Map the mappable_function to load data for each file in the dataset
    data = data.map(mappable_function)

    # Pad and batch the dataset with group size 2, padding frames to have shapes ([75,None,None,None]) and alignments to have shape ([40])
    data = data.padded_batch(2, padded_shapes=([75,None,None,None],[40]))

    # Prefetch data to improve pipeline performance
    data = data.prefetch(tf.data.AUTOTUNE)
    
    return data