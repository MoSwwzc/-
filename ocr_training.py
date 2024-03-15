#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['NLS_LANG'] = 'AMERICAN_AMERICA.AL32UTF8'
import random
import tf_slim as slim
import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image
import cv2
from tensorflow.python.ops import control_flow_ops
import label_dict
import sys
import time
from importlib import reload

stdo = sys.stdout
reload(sys)
sys.stdout = stdo
label_dict = label_dict.label_dict

# 输入参数解析
tf.app.flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
tf.app.flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness")
tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")

tf.app.flags.DEFINE_integer('charset_size', len(label_dict), "Choose the first `charset_size` characters only.")
tf.app.flags.DEFINE_integer('image_size', 64, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")
tf.app.flags.DEFINE_integer('max_steps', 16002, 'the max training steps ')
tf.app.flags.DEFINE_integer('eval_steps', 100, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 500, "the steps to save")

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', './dataset/train/', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', './dataset/test/', 'the test dataset dir')
tf.app.flags.DEFINE_string('logging_dir', './log', 'the logging dir')

tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_boolean('epoch', 1, 'Number of epoches')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Validation batch size')
tf.app.flags.DEFINE_string('mode', 'inference', 'Running mode. One of {"train", "valid", "test"}')
"""tf.app.flags.DEFINE_string() ：定义一个用于接收 string 类型数值的变量;
tf.app.flags.DEFINE_integer() : 定义一个用于接收 int 类型数值的变量;
tf.app.flags.DEFINE_float() ： 定义一个用于接收 float 类型数值的变量;
tf.app.flags.DEFINE_boolean() : 定义一个用于接收 bool 类型数值的变量;"""

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
FLAGS = tf.app.flags.FLAGS

# random_flip_up_down: 是否随机上下翻转图像。默认值为False。
# random_brightness: 是否调整图像的亮度。默认值为True。
# random_contrast: 是否随机调整图像的对比度。默认值为True。
# charset_size: 选择前多少个字符。默认值为label_dict的长度。
# image_size: 图像的大小。需要与训练时使用的值相同。默认值为64。
# gray: 是否将RGB图像转换为灰度图像。默认值为True。
# max_steps: 最大训练步数。默认值为16002。
# eval_steps: 评估的步数。默认值为100。
# save_steps: 保存的步数。默认值为500。
# checkpoint_dir: 检查点目录。默认值为'./checkpoint/'。
# train_data_dir: 训练数据集目录。默认值为'./dataset/train/'。
# test_data_dir: 测试数据集目录。默认值为'./dataset/test/'。
# logging_dir: 日志目录。默认值为'./log'。
# restore: 是否从检查点恢复。默认值为False。
# epoch: 迭代次数。默认值为1。
# batch_size: 验证批次大小。默认值为128。
# mode: 运行模式。可选值为{"train", "valid", "test"}。默认值为'inference'。
# 此外，还定义了一个gpu_options变量，用于设置GPU内存占用比例为90%。最后，将这些参数存储在FLAGS对象中，以便在后续的代码中使用。


class DataIterator:
    def __init__(self, data_dir):
        # Set FLAGS.charset_size to a small value if available computation power is limited.
        truncate_path = data_dir + ('%05d' % FLAGS.charset_size)
        print(truncate_path)
        # 遍历训练集所有图像的路径，存储在image_names内
        self.image_names = []
        for root, sub_folder, file_list in os.walk(data_dir):
            if root < truncate_path:
                self.image_names += [os.path.join(root, file_path) for file_path in file_list]


        random.shuffle(self.image_names)  # 打乱
        # 例如image_name为./train/00001/2.png，提取00001就是其label
        self.labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.image_names]

    # __init__方法：初始化函数，接受一个参数data_dir，表示训练集所在的目录。
    # truncate_path变量：根据FLAGS.charset_size的值生成截断路径，用于限制遍历的范围。
    # image_names列表：存储训练集中所有图像的路径。
    # labels列表：存储每个图像对应的标签。



    @property
    def size(self):
        return len(self.labels)

    @staticmethod
    def data_augmentation(images):
        # 镜像变换
        if FLAGS.random_flip_up_down:
            images = tf.image.random_flip_up_down(images)
        # 图像亮度变化
        if FLAGS.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.3)
        # 对比度变化
        if FLAGS.random_contrast:
            images = tf.image.random_contrast(images, 0.8, 1.2)
        return images

    # batch的生成
    def input_pipeline(self, batch_size, num_epochs=None, aug=False):
        # numpy array 转 tensor
        images_tensor = tf.convert_to_tensor(self.image_names, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)
        # 将image_list ,label_list做一个slice处理
        input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs)

        labels = input_queue[1]
        images_content = tf.read_file(input_queue[0])
        images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)
        if aug:
            images = self.data_augmentation(images)
        new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
        images = tf.image.resize_images(images, new_size)
        image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,
                                                          min_after_dequeue=10000)
        # print 'image_batch', image_batch.get_shape()
        return image_batch, label_batch

# 这段代码是一个Python类的一部分，其中包含了一些属性和方法。下面是对这段代码的解析：
#
# 1. `@property` 装饰器用于将一个方法转换为属性，使得可以通过对象的属性访问方式来调用该方法。在这里，`size` 方法被定义为一个属性，返回 `self.labels` 的长度。
#
# 2. `data_augmentation` 方法是一个静态方法，用于对输入的图像进行数据增强操作。根据不同的标志（FLAGS），可以对图像进行镜像变换、亮度变化和对比度变化等操作。
#
# 3. `input_pipeline` 方法用于生成数据的输入管道。它接受三个参数：`batch_size`（每个批次的大小）、`num_epochs`（迭代次数）和 `aug`（是否进行数据增强）。
# 在这个方法中，首先将 `image_names` 和 `labels` 转换为张量（Tensor），然后使用 `tf.train.slice_input_producer` 创建一个输入队列。
# 接着，从输入队列中读取图像内容并进行解码，将其转换为浮点数类型的图像。如果 `aug` 为真，则对图像进行数据增强操作。
# 最后，调整图像大小并使用 `tf.train.shuffle_batch` 对图像和标签进行随机打乱和批处理。最终返回批处理后的图像和标签。
#
# 以上是对给定代码段的解析。


def build_graph(top_k):
    tf.compat.v1.disable_eager_execution()
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')  # dropout打开概率
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='train_flag')
    with tf.device('/gpu:0'):
        # network: conv2d->max_pool2d->conv2d->max_pool2d->conv2d->max_pool2d->conv2d->conv2d->
        # max_pool2d->fully_connected->fully_connected
        # 给slim.conv2d和slim.fully_connected准备了默认参数：batch_norm
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training, 'decay': 0.95}):
            conv3_1 = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv3_1')
            max_pool_1 = slim.max_pool2d(conv3_1, [2, 2], [2, 2], padding='SAME', scope='pool1')
            conv3_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding='SAME', scope='conv3_2')
            max_pool_2 = slim.max_pool2d(conv3_2, [2, 2], [2, 2], padding='SAME', scope='pool2')
            conv3_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding='SAME', scope='conv3_3')
            max_pool_3 = slim.max_pool2d(conv3_3, [2, 2], [2, 2], padding='SAME', scope='pool3')
            conv3_4 = slim.conv2d(max_pool_3, 512, [3, 3], padding='SAME', scope='conv3_4')
            conv3_5 = slim.conv2d(conv3_4, 512, [3, 3], padding='SAME', scope='conv3_5')
            max_pool_4 = slim.max_pool2d(conv3_5, [2, 2], [2, 2], padding='SAME', scope='pool4')

            flatten = slim.flatten(max_pool_4)
            fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024,
                                       activation_fn=tf.nn.relu, scope='fc1')
            logits = slim.fully_connected(slim.dropout(fc1, keep_prob), FLAGS.charset_size, activation_fn=None,
                                          scope='fc2')
        # 因为我们没有做热编码，所以使用sparse_softmax_cross_entropy_with_logits
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))

        #         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #         if update_ops:
        #             updates = tf.group(*update_ops)
        #             loss = control_flow_ops.with_dependencies([updates], loss)

        #         global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
        #         optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
        #         train_op = slim.learning.create_train_op(loss, optimizer, global_step=global_step)

        global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        train_op = slim.learning.create_train_op(loss, optimizer, global_step=global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            train_op = control_flow_ops.with_dependencies([updates], train_op)

        probabilities = tf.nn.softmax(logits)

        # 绘制loss accuracy曲线
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        merged_summary_op = tf.summary.merge_all()
        # 返回top k 个预测结果及其概率；返回top K accuracy
        predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
        accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))

    return {'images': images,
            'labels': labels,
            'keep_prob': keep_prob,
            'top_k': top_k,
            'global_step': global_step,
            'train_op': train_op,
            'loss': loss,
            'is_training': is_training,
            'accuracy': accuracy,
            'accuracy_top_k': accuracy_in_top_k,
            'merged_summary_op': merged_summary_op,
            'predicted_distribution': probabilities,
            'predicted_index_top_k': predicted_index_top_k,
            'predicted_val_top_k': predicted_val_top_k}


# 获待预测图像文件夹内的图像名字
def get_file_list(path):
    list_name = []
    files = os.listdir(path)
    files.sort()
    for file in files:
        file_path = os.path.join(path, file)
        list_name.append(file_path)
    return list_name


is_build = False
graph = None
saver = None
ckpt = None
sess = None


# 预测
def inference(name_list):
    global is_build, graph, saver, ckpt, sess
    image_set = []
    # 对每张图进行尺寸标准化和归一化
    for image in name_list:
        temp_image = Image.open(image).convert('L')
        temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
        temp_image = np.asarray(temp_image) / 255.0
        temp_image = temp_image.reshape([-1, 64, 64, 1])
        image_set.append(temp_image)

    # allow_soft_placement 如果你指定的设备不存在，允许TF自动分配设备
    # print('========start inference============')
    # images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
    # Pass a shadow label 0. This label will not affect the computation graph.
    if not is_build:
        # tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
        graph = build_graph(top_k=1)
        saver = tf.train.Saver()
        # 自动获取最后一次保存的模型
        current_path = os.getcwd()
        if current_path.endswith('web'):
            # 运行在web环境中
            current_path = current_path + '/xuexin/checkpoint'
        else:
            current_path = './checkpoint'
        ckpt = tf.train.latest_checkpoint(current_path)
        if ckpt:
            saver.restore(sess, ckpt)
            # save_pb()
        is_build = True
    val_list = []
    idx_list = []
    # 预测每一张图
    for item in image_set:
        temp_image = item
        # print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
        predict_val, predict_index = sess.run([graph['predicted_val_top_k'], graph['predicted_index_top_k']],
                                              feed_dict={graph['images']: temp_image,
                                                         graph['keep_prob']: 1.0,
                                                         graph['is_training']: False})
        val_list.append(predict_val)
        idx_list.append(predict_index)
    # return predict_val, predict_index
    return val_list, idx_list


# 训练
def train():
    print('Begin training')
    # 填好数据读取的路径
    train_feeder = DataIterator(data_dir='./dataset/train/')
    test_feeder = DataIterator(data_dir='./dataset/test/')
    model_name = 'chinese-rec-model'
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        # batch data 获取
        train_images, train_labels = train_feeder.input_pipeline(batch_size=FLAGS.batch_size, aug=True)
        test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size)
        graph = build_graph(top_k=1)  # 训练时top k = 1
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        # 设置多线程协调器
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        # test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val')
        start_step = 0
        # 可以从某个step下的模型继续训练
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint {0}".format(ckpt))
                start_step += int(ckpt.split('-')[-1])

        # 这段代码是用于训练一个名为
        # "chinese-rec-model"
        # 的模型。首先，它创建了两个数据迭代器（DataIterator），分别用于读取训练和测试数据。
        # 然后，它使用TensorFlow会话（tf.Session）进行训练。在
        # 训练过程中，它会定期保存模型检查点（checkpoint），并在每个评估步骤（eval_steps）后对测试数据进行评估。
        # 当达到最大训练步数（max_steps）或测试准确率超过0.999
        # 时，训练将停止。

        print(':::Training Start:::')
        try:
            i = 0
            while not coord.should_stop():
                i += 1
                start_time = time.time()
                train_images_batch, train_labels_batch = sess.run([train_images, train_labels])
                feed_dict = {graph['images']: train_images_batch,
                             graph['labels']: train_labels_batch,
                             graph['keep_prob']: 0.8,
                             graph['is_training']: True}
                _, loss_val, train_summary, step = sess.run(
                    [graph['train_op'], graph['loss'], graph['merged_summary_op'], graph['global_step']],
                    feed_dict=feed_dict)
                #                 train_writer.add_summary(train_summary, step)
                end_time = time.time()
                print("the step {0} takes {1} loss {2}".format(step, end_time - start_time, loss_val))
                if step > FLAGS.max_steps:
                    break
                if step % FLAGS.eval_steps == 1:
                    test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                    feed_dict = {graph['images']: test_images_batch,
                                 graph['labels']: test_labels_batch,
                                 graph['keep_prob']: 1.0,
                                 graph['is_training']: False}
                    accuracy_test, test_summary = sess.run([graph['accuracy'], graph['merged_summary_op']],
                                                           feed_dict=feed_dict)

                    # if step > 300:
                    #     test_writer.add_summary(test_summary, step)
                    print('===============Eval a batch=======================')
                    print('the step {0} test accuracy: {1}'.format(step, accuracy_test))
                    print('===============Eval a batch=======================')
                if step % FLAGS.save_steps == 1:
                    print('Save the ckpt of {0}'.format(step))
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, model_name),
                               global_step=graph['global_step'])
                    if accuracy_test > 0.999:
                        break
        # 这段代码是一个训练模型的循环，其中包含了以下步骤：
        #
        # 初始化计数器 i 为0。
        # 进入一个循环，直到 coord.should_stop() 返回 True。
        # 在每次循环中，增加计数器 i 的值。
        # 记录当前时间作为开始时间。
        # 从数据迭代器中获取一批训练图像和标签。
        # 构建一个字典 feed_dict，包含输入到模型中的图像、标签、保持概率和是否处于训练状态的信息。
        # 使用 sess.run() 运行模型的训练操作，并获取损失值、训练摘要和全局步数。
        # 计算当前步骤所花费的时间，并打印出步骤、时间和损失值。
        # 如果当前步骤超过了最大步数 FLAGS.max_steps，则跳出循环。
        # 如果当前步骤是评估步数 FLAGS.eval_steps 的倍数，则进行测试集上的评估。
        # 从数据迭代器中获取一批测试图像和标签。
        # 构建一个字典 feed_dict，包含输入到模型中的图像、标签、保持概率和是否处于训练状态的信息。
        # 使用 sess.run() 运行模型的准确率和测试摘要。
        # 打印出当前步骤的测试准确率。
        # 如果当前步骤是保存步数 FLAGS.save_steps 的倍数，则保存模型的检查点。
        # 打印出保存的检查点信息。
        # 使用 saver.save() 保存模型的检查点。
        # 如果测试准确率超过0.999，则跳出循环。
        # 这段代码的作用是在一个循环中训练模型，并在每个评估步和保存步进行相应的操作，包括评估模型在测试集上的性能和保存模型的检查点。
        except tf.errors.OutOfRangeError:
            print('==================Train Finished================')
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, model_name), global_step=graph['global_step'])
        """finally:
            # 达到最大训练迭代数的时候清理关闭线程
            coord.request_stop()"""
        coord.join(threads)
    # 这段代码是一个异常处理的代码块，用于捕获tf.errors.OutOfRangeError异常。当训练数据迭代器中没有更多的数据可供读取时，会抛出这个异常。
    # 在异常处理块中，首先打印出
    # "==================Train Finished================"，表示训练已经完成。然后使用saver.save()
    # 保存模型的检查点到指定的目录中，文件名为model_name，并使用全局步数作为文件名的一部分。
    # 最后，调用coord.join(threads)来等待所有线程完成，确保所有的资源都被正确释放

def pred(path,space_number=[]):
    name_list = get_file_list(path)
    # binary_pic(name_list)
    # tmp_name_list = get_file_list('../data/tmp')
    # 将待预测的图片名字列表送入predict()进行预测，得到预测的结果及其index
    final_predict_val, final_predict_index = inference(name_list)
    final_reco_text = []  # 存储最后识别出来的文字串
    # 给出top 3预测，candidate1是概率最高的预测
    pred_val_list = []
    for i in range(len(final_predict_val)):
        candidate1 = final_predict_index[i][0][0]
        # candidate2 = final_predict_index[i][0][1]
        # candidate3 = final_predict_index[i][0][2]
        r = label_dict[int(candidate1)].replace('（', '(').replace("）", ")")
        final_reco_text.append(r)




        if space_number!=None and i in space_number:
            final_reco_text.append(" ")
        """print('[the result info] image: {0} predict: {1} ; predict index {2} predict_val {3}'.format(
            name_list[i],
            label_dict[int(candidate1)],
            final_predict_index[i], final_predict_val[i]))"""
        pred_dict = {'accu': final_predict_val[i]
            , 'shape': cv2.imread(name_list[i]).shape, 'result': r}
        pred_val_list.append(pred_dict)
    # print ('=====================OCR RESULT=======================')
    # 打印出所有识别出来的结果（取top 1）+
    result = []
    for i in range(len(final_reco_text)):
        result.append(final_reco_text[i]),
    print(''.join(result))
    return ''.join(result), pred_val_list
# 这段代码是一个名为`pred`的函数，它接受两个参数：`path`和`space_number`。`path`是一个文件夹路径，`space_number`是一个可选的列表，默认为空列表。
#
# 函数的主要功能是对给定路径下的图像进行预测，并返回识别出的文字串和预测结果的详细信息。
#
# 以下是代码的解析：
#
# 1. 首先，通过调用`get_file_list(path)`函数获取指定路径下的文件名列表，并将其存储在`name_list`变量中。
# 2. 然后，调用`inference(name_list)`函数对图像进行预测，得到预测结果及其索引，分别存储在`final_predict_val`和`final_predict_index`变量中。
# 3. 接下来，创建一个空列表`final_reco_text`，用于存储最后识别出来的文字串。
# 4. 创建一个空列表`pred_val_list`，用于存储预测结果的详细信息。
# 5. 使用一个循环遍历`final_predict_val`的长度，对于每个索引`i`：
#    - 获取概率最高的预测结果`candidate1`。
#    - 根据`candidate1`从`label_dict`中获取对应的标签，并将其中的括号替换为英文括号，然后将其添加到`final_reco_text`列表中。
#    - 如果`space_number`不为空且当前索引`i`在`space_number`中，则在`final_reco_text`中添加一个空格。
#    - 创建一个字典`pred_dict`，包含预测准确率、图像形状和识别结果等信息，并将其添加到`pred_val_list`列表中。
# 6. 创建一个空列表`result`，用于存储所有识别出来的结果。
# 7. 使用一个循环遍历`final_reco_text`的长度，将每个元素添加到`result`列表中。
# 8. 打印出拼接后的`result`列表。
# 9. 返回拼接后的`result`列表和`pred_val_list`列表。

if __name__ == "__main__":
    train()
    """files = os.listdir("./test/")
    num_png = len(files)
    for i in range(num_png):
        files = os.listdir("./test/"+str(i))
        num_png = len(files)
        pred("./test/"+str(i))"""