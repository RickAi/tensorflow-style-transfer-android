# coding=UTF-8
import tensorflow as tf
import os.path
import argparse
from tensorflow.python.framework import graph_util

# convert existing checkpoint to forzen model .pb

MODEL_DIR = "outputs/pb"
MODEL_NAME = "frozen_model.pb"

if not tf.gfile.Exists(MODEL_DIR):  # 创建目录
    tf.gfile.MakeDirs(MODEL_DIR)


def freeze_graph(aggs):
    checkpoint = tf.train.get_checkpoint_state(aggs.model_folder)  # 检查目录下ckpt文件状态是否可用
    input_checkpoint = checkpoint.model_checkpoint_path  # 得ckpt文件路径
    output_graph = os.path.join(MODEL_DIR, MODEL_NAME)  # PB模型保存路径

    output_node_names = aggs.output_names  # 原模型输出操作节点的名字
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                       clear_devices=True)

    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    print(">> Start to save frozen model.")
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据

        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess,
            input_graph_def,
            output_node_names.split(",")  # 如果有多个输出节点，以逗号隔开
        )
        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
            print(">> Save frozen model done at:%s" % output_graph)
        # print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点

        # for op in graph.get_operations():
        #     print(op.name, op.values())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_folder", default="./models", nargs='?', type=str, help="input ckpt model dir")  # 命令行解析，help是提示符，type是输入的类型，
    parser.add_argument("output_names", default='clip_by_value', nargs='?', type=str, help="output names")
    aggs = parser.parse_args()
    freeze_graph(aggs)
