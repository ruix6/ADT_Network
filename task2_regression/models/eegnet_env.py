import tensorflow as tf

class EEGNet(tf.keras.models.Model):
    def __init__(self, num_channels, num_samples, dropout_rate=0.5):
        super(EEGNet, self).__init__()
        self.firstconv = tf.keras.layers.Conv2D(4, (3, 1), padding='valid', input_shape=(num_samples, num_channels, 1), use_bias=False)
        self.depthwiseconv = tf.keras.layers.DepthwiseConv2D((1, num_channels), use_bias=False, depth_multiplier=8, depthwise_initializer='glorot_uniform', padding='valid')
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.activation1 = tf.keras.layers.ELU()
        self.avgpool1 = tf.keras.layers.AveragePooling2D((2, 1))
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.separableconv1 = tf.keras.layers.SeparableConv2D(32, (3, 1), use_bias=False, padding='valid')
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.activation2 = tf.keras.layers.ELU()
        self.avgpool2 = tf.keras.layers.AveragePooling2D((2, 1))
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(num_samples, activation='linear', use_bias=False)

    def call(self, inputs):
        # expand dimension
        x = tf.expand_dims(inputs, axis=-1)
        x = self.firstconv(x)
        x = self.depthwiseconv(x)
        x = self.batchnorm1(x)
        x = self.activation1(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        x = self.separableconv1(x)
        x = self.batchnorm2(x)
        x = self.activation2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.dense(x)
 
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'firstconv': self.firstconv,
            'depthwiseconv': self.depthwiseconv,
            'batchnorm1': self.batchnorm1,
            'activation1': self.activation1,
            'avgpool1': self.avgpool1,
            'dropout1': self.dropout1,
            'separableconv1': self.separableconv1,
            'batchnorm2': self.batchnorm2,
            'activation2': self.activation2,
            'avgpool2': self.avgpool2,
            'dropout2': self.dropout2,
            'flatten': self.flatten,
            'dense': self.dense
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def pearson_tf(y_true, y_pred, axis=1):
    """Pearson correlation function implemented in tensorflow.

    Parameters
    ----------
    y_true: tf.Tensor
        Ground truth labels. Shape is (batch_size, time_steps, n_features)
    y_pred: tf.Tensor
        Predicted labels. Shape is (batch_size, time_steps, n_features)
    axis: int
        Axis along which to compute the pearson correlation. Default is 1.

    Returns
    -------
    tf.Tensor
        Pearson correlation.
        Shape is (batch_size, 1, n_features) if axis is 1.
    """
    # Compute the mean of the true and predicted values
    y_true_mean = tf.reduce_mean(y_true, axis=axis, keepdims=True)
    y_pred_mean = tf.reduce_mean(y_pred, axis=axis, keepdims=True)

    # Compute the numerator and denominator of the pearson correlation
    numerator = tf.reduce_sum(
        (y_true - y_true_mean) * (y_pred - y_pred_mean),
        axis=axis,
        keepdims=True,
    )
    std_true = tf.reduce_sum(tf.square(y_true - y_true_mean), axis=axis, keepdims=True)
    std_pred = tf.reduce_sum(tf.square(y_pred - y_pred_mean), axis=axis, keepdims=True)
    denominator = tf.sqrt(std_true * std_pred)

    # Compute the pearson correlation
    return tf.reduce_mean(tf.math.divide_no_nan(numerator, denominator), axis=-1)

def pearson_tf_non_averaged(y_true, y_pred, axis=1):
    """Pearson correlation function implemented in tensorflow.

    Parameters
    ----------
    y_true: tf.Tensor
        Ground truth labels. Shape is (batch_size, time_steps, n_features)
    y_pred: tf.Tensor
        Predicted labels. Shape is (batch_size, time_steps, n_features)
    axis: int
        Axis along which to compute the pearson correlation. Default is 1.

    Returns
    -------
    tf.Tensor
        Pearson correlation.
        Shape is (batch_size, 1, n_features) if axis is 1.
    """
    # Compute the mean of the true and predicted values
    y_true_mean = tf.reduce_mean(y_true, axis=axis, keepdims=True)
    y_pred_mean = tf.reduce_mean(y_pred, axis=axis, keepdims=True)

    # Compute the numerator and denominator of the pearson correlation
    numerator = tf.reduce_sum(
        (y_true - y_true_mean) * (y_pred - y_pred_mean),
        axis=axis,
        keepdims=True,
    )
    std_true = tf.reduce_sum(tf.square(y_true - y_true_mean), axis=axis, keepdims=True)
    std_pred = tf.reduce_sum(tf.square(y_pred - y_pred_mean), axis=axis, keepdims=True)
    denominator = tf.sqrt(std_true * std_pred)

    # Compute the pearson correlation
    return tf.math.divide_no_nan(numerator, denominator)


@tf.function
def pearson_loss(y_true, y_pred, axis=1):
    """Pearson loss function.

    Parameters
    ----------
    y_true: tf.Tensor
        True values. Shape is (batch_size, time_steps, n_features)
    y_pred: tf.Tensor
        Predicted values. Shape is (batch_size, time_steps, n_features)

    Returns
    -------
    tf.Tensor
        Pearson loss.
        Shape is (batch_size, 1, n_features)
    """
    return -pearson_tf(y_true, y_pred, axis=axis)


@tf.function
def pearson_metric(y_true, y_pred, axis=1):
    """Pearson metric function.

    Parameters
    ----------
    y_true: tf.Tensor
        True values. Shape is (batch_size, time_steps, n_features)
    y_pred: tf.Tensor
        Predicted values. Shape is (batch_size, time_steps, n_features)

    Returns
    -------
    tf.Tensor
        Pearson metric.
        Shape is (batch_size, 1, n_features)
    """
    return pearson_tf(y_true, y_pred, axis=axis)
    
if __name__ == '__main__':

    model = EEGNet(64, 320)
    model.build(input_shape=(None, 320, 64))
    model.summary()
    x = tf.random.uniform((100, 320, 64))
    y = tf.random.uniform((100, 320, 1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=2, batch_size=32)
