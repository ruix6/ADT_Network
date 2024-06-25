import tensorflow as tf
import numpy as np
tf.keras.backend.clear_session()

class featureExtraction(tf.keras.layers.Layer):
    def __init__(self, chans, F, T, D, use_bias=False, **kwargs):
        super(featureExtraction, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(filters=F, kernel_size=(T, 1), use_bias=use_bias, padding='same')
        self.norm = tf.keras.layers.LayerNormalization()
        self.conv2 = tf.keras.layers.DepthwiseConv2D(kernel_size=(1, chans), use_bias=use_bias, depth_multiplier=D)
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.activation = tf.keras.layers.Activation('elu')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        return tf.cast(x, tf.float32)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'conv1': self.conv1,
            'norm': self.norm,
            'conv2': self.conv2,
            'norm2': self.norm2,
            'activation': self.activation
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def positionalEncoding(position, d_model):
    def get_angles(pos, i, d_model):
        angle_rates = 1 / tf.pow(10000., (2. * (tf.cast(i, tf.float32) // 2.)) / tf.cast(d_model, tf.float32))
        return tf.cast(pos, dtype=tf.float32) * angle_rates

    angle_rads = get_angles(tf.range(position)[:, tf.newaxis],
                            tf.range(d_model)[tf.newaxis, :],
                            d_model)

    sine_mask = tf.cast(tf.range(d_model) % 2 == 0, tf.float32)
    cosine_mask = tf.cast(tf.range(d_model) % 2 == 1, tf.float32)

    sines = tf.math.sin(angle_rads) * sine_mask
    cosines = tf.math.cos(angle_rads) * cosine_mask

    pos_encoding = sines + cosines
    pos_encoding = pos_encoding[tf.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

class transformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, rate=0.5, ff_dim=128, use_bias=True, mask=True, **kwargs):
        super(transformerBlock, self).__init__(**kwargs)
        if mask:
            self.mask_ = 1 - tf.linalg.band_part(tf.ones((320, 320)), -1, 0)# 因果掩码：tf.linalg.band_part(tf.ones((320, 320)), -1, 0)
            self.mask_ = tf.expand_dims(self.mask_, axis=0)
        else:
            self.mask_ = None
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, use_bias=use_bias, dropout=rate)
        self.ffnsub = tf.keras.layers.Conv1D(ff_dim, 3, activation='elu', use_bias=False, padding='same')
        self.ffn = tf.keras.layers.Conv1D(embed_dim, 3, activation='elu', use_bias=False, padding='same')
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs, attention_mask=self.mask_)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffnsub(out1)
        ffn_output = self.dropout(ffn_output)
        ffn_output = self.ffn(ffn_output)
        out = self.layernorm2(out1 + ffn_output)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'att': self.att,
            'ffnsub': self.ffnsub,
            'ffn': self.ffn,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'dropout': self.dropout
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class SqueezeLayer(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super(SqueezeLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'axis': self.axis
        })
        return config

class ExpandDimsLayer(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'axis': self.axis
        })
        return config


class ADT(tf.keras.Model):
    def __init__(self,chans, outputDims, F, T, D, heads, ff_dim, blocks, mask=True, use_bias=False, lrate=0.5, **kwargs):
        super(ADT, self).__init__(**kwargs)
        self.featureExtraction = featureExtraction(chans, F, T, D, use_bias=use_bias)
        self.expandDims = ExpandDimsLayer(axis=-1)
        self.squeezeDims = SqueezeLayer(axis=-2)
        self.positionalEncoding = positionalEncoding(320, F*D)
        self.transformer = [transformerBlock(embed_dim=F*D, num_heads=heads, rate=lrate, ff_dim=ff_dim, use_bias=use_bias, mask=mask) for _ in range(blocks)]
        self.linearProjection = tf.keras.layers.Dense(outputDims, use_bias=use_bias)

    def call(self, inputs):
        x = self.expandDims(inputs)
        x = self.featureExtraction(x)
        x = self.squeezeDims(x)
        x = x + self.positionalEncoding
        for i in range(len(self.transformer)):
            x = self.transformer[i](x)
        x = self.linearProjection(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'featureExtraction': self.featureExtraction,
            'expandDims': self.expandDims,
            'squeezeDims': self.squeezeDims,
            'positionalEncoding': self.positionalEncoding,
            'transformer': self.transformer,
            'linearProjection': self.linearProjection
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
    model = ADT(chans=64, outputDims=10, F=16, T=16, D=4, heads=4, ff_dim=128, blocks=2, mask=True, use_bias=False, lrate=0.5)
    model.build(input_shape=(None, 320, 64))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=pearson_loss, metrics=[pearson_metric])
    x = np.random.random((10, 320, 64))
    y = np.random.random((10, 320, 10))
    model.fit(x, y, batch_size=2, epochs=1)
    model.summary()
    # #model.save_weights('model_weights.h5')
    # 打印模型的所有可训练参数
    for weight in model.trainable_weights:
        print(weight.name, weight.shape)