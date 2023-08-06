import tensorflow as tf

characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)


def decode_batch_predictions(pred):
    # Batch Prediction:
    # input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # # Use greedy search. For complex tasks, you can use beam search
    # results = tensorflow.keras.backend.ctc_decode(pred, input_length=input_len, greedy=False)[0][0]
    # # Iterate over the results and get back the text
    # output_text = []
    # for result in results:
    #     result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
    #     output_text.append(result)
    input_len = pred.shape[1]
    results = tf.keras.backend.ctc_decode([pred], input_length=[input_len], greedy=False)[0][0]
    output_text = tf.strings.reduce_join(num_to_char(results)).numpy().decode("utf-8")
    return output_text


def load_alignments(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens = [*tokens, ' ', line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]
