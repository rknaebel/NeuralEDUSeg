import tensorflow as tf


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)


def trilinear_similarity(x1, x2, scope='trilinear', reuse=None):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        x1_shape = x1.shape.as_list()
        x2_shape = x2.shape.as_list()
        if len(x1_shape) != 3 or len(x2_shape) != 3:
            raise ValueError(
                '`args` must be 3 dims (batch_size, len, dimension)')
        if x1_shape[2] != x2_shape[2]:
            raise ValueError('the last dimension of `args` must equal')
        weights_x1 = tf.compat.v1.get_variable('kernel_x1', [x1_shape[2], 1],
                                               dtype=x1.dtype)
        weights_x2 = tf.compat.v1.get_variable('kernel_x2', [x2_shape[2], 1],
                                               dtype=x2.dtype)
        weights_mul = tf.compat.v1.get_variable('kernel_mul', [1, 1, x1_shape[2]],
                                                dtype=x2.dtype)
        bias = tf.compat.v1.get_variable('bias', [1], dtype=x1.dtype,
                                         initializer=tf.zeros_initializer)
        subres0 = tf.tile(tf.keras.backend.dot(x1, weights_x1),
                          [1, 1, tf.shape(x2)[1]])
        subres1 = tf.tile(tf.transpose(tf.keras.backend.dot(x2, weights_x2),
                                       (0, 2, 1)), [1, tf.shape(x1)[1], 1])
        subres2 = tf.keras.backend.batch_dot(x1 * weights_mul,
                                             tf.transpose(x2, perm=(0, 2, 1)))
        return subres0 + subres1 + subres2 + tf.tile(bias, [tf.shape(x2)[1]])


def self_attention(inputs, lengths, window_size=-1, scope='bilinear_attention',
                   reuse=None):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # logits = tf.matmul(inputs, inputs, transpose_b=True)  # Q * K
        logits = trilinear_similarity(inputs, inputs)
        mask = tf.sequence_mask(lengths, tf.shape(inputs)[1], tf.float32)
        mask = tf.expand_dims(mask, 1)
        if window_size > 0:
            restricted_mask = tf.compat.v1.matrix_band_part(
                tf.ones_like(logits, dtype=tf.float32),
                window_size, window_size)
            mask = mask * restricted_mask
        logits = mask_logits(logits, mask)
        weights = tf.nn.softmax(logits, name='attn_weights')
        return tf.matmul(weights, inputs), weights
