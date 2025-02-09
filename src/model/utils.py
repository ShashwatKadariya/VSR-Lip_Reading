import tensorflow as tf


vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
# String to number mapping
char_to_num = tf.keras.layers.StringLookup(
    vocabulary=vocab, oov_token=""
)

# Number to string mapping
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


def scheduler(epoch, lr):
    """
    Learning rate scheduler function.

    Args:
    - epoch (int): The current epoch number.
    - lr (float): The current learning rate.

    Returns:
    - float: The updated learning rate based on the epoch number.
    """

    if epoch < 30:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def CTCLoss(y_true, y_pred):
    """
    Compute the Connectionist Temporal Classification (CTC) loss.

    Args:
    - y_true (tensor): True labels. Expected to have shape (batch_size, max_label_length).
    - y_pred (tensor): Predicted logits. Expected to have shape (batch_size, max_input_length, num_classes).

    Returns:
    - tensor: CTC loss.
    """

    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

class ProduceExample(tf.keras.callbacks.Callback):
    """
    Callback to produce examples of model predictions at the end of each epoch.

    Args:
    - dataset (tf.data.Dataset): Dataset used for evaluation.

    Methods:
    - on_epoch_end(epoch, logs=None): Called at the end of each epoch to generate and print model predictions.
    """
    def __init__(self, dataset) -> None:
        """
        Initialize the callback.

        Args:
        - dataset (tf.data.Dataset): Dataset used for evaluation.
        """
        self.dataset = dataset.as_numpy_iterator()

    def on_epoch_end(self, epoch, logs=None) -> None:
        """
        Generate and print model predictions at the end of each epoch.

        Args:
        - epoch (int): Current epoch number.
        - logs (dict): Dictionary containing the loss value and any other metrics during training.
        """
        data = self.dataset.next()
        yhat = self.model.predict(data[0])
        decoded = tf.keras.backend.ctc_decode(yhat, [75,75], greedy=False)[0][0].numpy()
        for x in range(len(yhat)):
            print('Original:', tf.strings.reduce_join(num_to_char(data[1][x])).numpy().decode('utf-8'))
            print('Prediction:', tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8'))
            print('~'*100)

class ProduceExample(tf.keras.callbacks.Callback):
    """
    Callback to produce examples of model predictions at the end of each epoch.

    Args:
    - dataset (tf.data.Dataset): Dataset used for evaluation.

    Methods:
    - on_epoch_end(epoch, logs=None): Called at the end of each epoch to generate and print model predictions.
    """
    def __init__(self, dataset) -> None:
        """
        Initialize the callback.

        Args:
        - dataset (tf.data.Dataset): Dataset used for evaluation.
        """
        self.dataset = dataset.as_numpy_iterator()

    def on_epoch_end(self, epoch, logs=None) -> None:
        """
        Generate and print model predictions at the end of each epoch.

        Args:
        - epoch (int): Current epoch number.
        - logs (dict): Dictionary containing the loss value and any other metrics during training.
        """
        data = self.dataset.next()
        yhat = self.model.predict(data[0])
        decoded = tf.keras.backend.ctc_decode(yhat, [75,75], greedy=False)[0][0].numpy()
        for x in range(len(yhat)):
            print('Original:', tf.strings.reduce_join(num_to_char(data[1][x])).numpy().decode('utf-8'))
            print('Prediction:', tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8'))
            print('~'*100)
