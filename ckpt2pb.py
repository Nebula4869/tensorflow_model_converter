import tensorflow as tf


def show_graph(input_checkpoint):
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        for variable in tf.global_variables():
            print(variable)
        tf.summary.FileWriter('./log/', sess.graph)


def freeze_graph(input_checkpoint, output_node_names, output_graph):
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']

    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, '', './pb/model.pbtxt')
        saver.restore(sess, input_checkpoint)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names.split(","))

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

        # for operation in output_graph_def.node:
        #     print(operation)


if __name__ == '__main__':
    show_graph('./ckpt/model')
    # freeze_graph('./ckpt/model', 'embeddings', './pb/model.pb')
