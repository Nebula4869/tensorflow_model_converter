import tensorflow as tf


def show_graph(pb_model_file):
    with tf.Session() as sess:
        with open(pb_model_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        tf.summary.FileWriter('./log/', sess.graph)


def convert_model(pb_model_file, tflite_model_file):
    converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(
        pb_model_file,
        input_arrays=['input'],
        output_arrays=['embeddings'],
        input_shapes={'input': [1, 160, 160, 3]})

    tflite_model = converter.convert()

    with open(tflite_model_file, 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    show_graph('./pb/model.pb')
    # convert_model('./pb/model.pb', 'model.tflite')
