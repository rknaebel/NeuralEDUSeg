import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow_hub as hub

from neuraleduseg.model.base_seg import BaseSegModel
from neuraleduseg.model.layers import self_attention
from neuraleduseg.model.rnn import rnn


class AttnSegModel(BaseSegModel):

    def __init__(self, args):
        super().__init__(args)

    def _build_graph(self):
        self._setup_placeholders()
        self.sequence_mask = tf.sequence_mask(
            self.placeholders['input_length'], dtype=tf.float32
        )
        with tf.compat.v1.variable_scope('embedding'):
            self._embed()
        with tf.compat.v1.variable_scope('encoding'):
            self._encode()
        with tf.compat.v1.variable_scope('output'):
            self._output()
        with tf.compat.v1.variable_scope('loss'):
            self._compute_loss()
        self.grads, self.grad_norm, self.train_op = self._get_train_op(
            self.loss
        )

    def _setup_placeholders(self):
        self.placeholders = {
            'input_length': tf.compat.v1.placeholder(tf.int32, shape=[None]),
            'elmo_input_words': tf.compat.v1.placeholder(tf.string, shape=[None, None]),
            'seg_labels': tf.compat.v1.placeholder(tf.float32, shape=[None, None]),
            'dropout_keep_prob': tf.compat.v1.placeholder(tf.float32)
        }

    def _embed(self):
        elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
        self.elmo = elmo(
            inputs={
                "tokens": self.placeholders['elmo_input_words'],
                "sequence_len": self.placeholders['input_length']
            },
            signature="tokens",
            as_dict=True)["elmo"]
        self.embedded_inputs = self.elmo

    def _output(self):
        self.scores = tc.layers.fully_connected(self.encoded_sent, 2,
                                                activation_fn=None,
                                                scope='output_fc1')
        self.log_likelyhood, self.trans_params = tc.crf.crf_log_likelihood(
            self.scores,
            tf.cast(self.placeholders['seg_labels'], tf.int32),
            self.placeholders['input_length']
        )

    def _compute_loss(self):
        self.loss = tf.reduce_mean(-self.log_likelyhood, 0)
        if self.weight_decay > 0:
            with tf.compat.v1.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v)
                                    for v in tf.compat.v1.trainable_variables()
                                    if 'bias' not in v.name])
            self.loss += self.weight_decay * l2_loss

    def _train_epoch(self, train_batches, print_every_n_batch):
        total_loss, total_batch_num = 0, 0
        for bitx, batch in enumerate(train_batches):
            feed_dict = {
                self.placeholders['input_length']: batch['length'],
                self.placeholders['elmo_input_words']: batch['words'],
                self.placeholders['seg_labels']: batch['seg_labels'],
                self.placeholders['dropout_keep_prob']: self.dropout_keep_prob,
            }
            _, loss, grad_norm = self.sess.run([self.train_op, self.loss,
                                                self.grad_norm], feed_dict)

            if (bitx != 0 and print_every_n_batch > 0
                    and bitx % print_every_n_batch == 0):
                self.logger.info('bitx: {}, loss: {}, grad: {}'.format(
                    bitx, loss, grad_norm
                )
                )
            total_loss += loss
            total_batch_num += 1
        return total_loss / total_batch_num

    def segment(self, batch):
        feed_dict = {
            self.placeholders['input_length']: batch['length'],
            self.placeholders['elmo_input_words']: batch['words'],
            self.placeholders['dropout_keep_prob']: 1.0,
        }
        scores, trans_params = self.sess.run([self.scores, self.trans_params],
                                             feed_dict)

        batch_pred_segs = []
        for sample_idx in range(len(batch['raw_data'])):
            length = batch['length'][sample_idx]
            viterbi_seq, viterbi_score = tc.crf.viterbi_decode(
                scores[sample_idx][:length], trans_params
            )
            pred_segs = []
            for word_idx, label in enumerate(viterbi_seq):
                if label == 1:
                    pred_segs.append(word_idx)
            batch_pred_segs.append(pred_segs)
        return batch_pred_segs

    def _encode(self):
        with tf.compat.v1.variable_scope('rnn_1'):
            self.encoded_sent, _ = rnn('bi-lstm', self.embedded_inputs,
                                       self.placeholders['input_length'],
                                       hidden_size=self.hidden_size,
                                       layer_num=1, concat=True)
            self.encoded_sent = tf.nn.dropout(
                self.encoded_sent, self.placeholders['dropout_keep_prob']
            )
        self.attn_outputs, self.attn_weights = self_attention(
            self.encoded_sent, self.placeholders['input_length'],
            self.window_size)
        self.attn_outputs = tf.nn.dropout(
            self.attn_outputs,
            self.placeholders['dropout_keep_prob']
        )
        self.encoded_sent = tf.concat([self.encoded_sent, self.attn_outputs], -1)
        with tf.compat.v1.variable_scope('rnn_2'):
            self.encoded_sent, _ = rnn('bi-lstm',
                                       self.encoded_sent,
                                       self.placeholders['input_length'],
                                       hidden_size=self.hidden_size, layer_num=1, concat=True)
            self.encoded_sent = tf.nn.dropout(
                self.encoded_sent,
                self.placeholders['dropout_keep_prob'])
