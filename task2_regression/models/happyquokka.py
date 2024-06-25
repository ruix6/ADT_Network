import tensorflow as tf

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, d_hid, rate=0.5, use_bias=False, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, use_bias=use_bias)
        self.ffnsub = tf.keras.layers.Conv1D(d_hid, 9, activation='elu', use_bias=False, padding='same')
        self.ffn = tf.keras.layers.Conv1D(embed_dim, 9, activation='elu', use_bias=False, padding='same')
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs):
        attn_input = self.layernorm1(inputs)
        attn_output = self.att(attn_input, attn_input)
        ffn_input = self.layernorm2(attn_input + attn_output)
        ffn_output = self.ffnsub(ffn_input)
        ffn_output = self.dropout(ffn_output)
        ffn_output = self.ffn(ffn_output)
        out = ffn_input + ffn_output
        
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
    
# def positional_encoding(position, d_model):
#     def get_angles(pos, i, d_model):
#         angle_rates = 1 / tf.pow(10000., (2. * (tf.cast(i, tf.float32) // 2.)) / tf.cast(d_model, tf.float32))
#         return tf.cast(pos, dtype=tf.float32) * angle_rates

#     angle_rads = get_angles(tf.range(position)[:, tf.newaxis],
#                             tf.range(d_model)[tf.newaxis, :],
#                             d_model)

#     sine_mask = tf.cast(tf.range(d_model) % 2 == 0, tf.float32)
#     cosine_mask = tf.cast(tf.range(d_model) % 2 == 1, tf.float32)

#     sines = tf.math.sin(angle_rads) * sine_mask
#     cosines = tf.math.cos(angle_rads) * cosine_mask

#     pos_encoding = sines + cosines
#     pos_encoding = pos_encoding[tf.newaxis, ...]

#     return tf.cast(pos_encoding, dtype=tf.float32)

class happyquokka(tf.keras.models.Model):
    def __init__(self, num_layers, embed_dim, num_heads, d_hid, rate=0.5, use_bias=False, **kwargs):
        super(happyquokka, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.rate = rate
        self.use_bias = use_bias
        #self.pos_encoding = positional_encoding(320, embed_dim)
        self.input_proj = tf.keras.layers.Conv1D(embed_dim, kernel_size=7, use_bias=False, padding='same')
        self.input_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.input_act = tf.keras.layers.Activation('elu')
        self.blocks = [TransformerBlock(embed_dim, num_heads, d_hid, rate, use_bias) for _ in range(num_layers)]
        self.linear_proj = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.input_proj(inputs)
        #x = x + self.pos_encoding
        x = self.input_norm(x)
        x = self.input_act(x)
        for block in self.blocks:
            x = block(x)
        x = self.linear_proj(x)
        return x
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_layers': self.num_layers,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'rate': self.rate,
            'use_bias': self.use_bias,
            'pos_encoding': self.pos_encoding,
            'input_proj': self.input_proj,
            'input_norm': self.input_norm,
            'input_act': self.input_act,
            'blocks': self.blocks,
            'linear_proj': self.linear_proj
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
    model = happyquokka(num_layers=8, embed_dim=128, num_heads=2, d_hid=1024)
    model.build(input_shape=(None, 320, 64))
    model.summary()
    x = tf.random.uniform((100, 320, 64))
    y = tf.random.uniform((100, 320, 1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=2, batch_size=32)
