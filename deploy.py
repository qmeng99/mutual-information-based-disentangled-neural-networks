import os
import tensorflow as tf
import numpy as np
import residual_def
import random
from sklearn.metrics import precision_recall_fscore_support, recall_score, precision_score, accuracy_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels


height = 224
width = 288
ckpt_dir = './model'
anat_num = 4



def load_data(flag):
    if flag == 0:
        testpath_1 = '/test/Abdominal_1.npz'
    else:
        testpath_1 = '/test/Abdominal_2.npz'
    testimg_1 = np.load(testpath_1)['img']
    test_lbl_anat_1 = np.load(testpath_1)['anatlbl']

    if flag == 0:
        testpath_2 = '/test/Brain_1.npz'
    else:
        testpath_2 = '/test/Brain_2.npz'
    testimg_2 = np.load(testpath_2)['img']
    test_lbl_anat_2 = np.load(testpath_2)['anatlbl']

    if flag == 0:
        testpath_3 = '/test/Femur_1.npz'
    else:
        testpath_3 = '/test/Femur_2.npz'
    testimg_3 = np.load(testpath_3)['img']
    test_lbl_anat_3 = np.load(testpath_3)['anatlbl']

    if flag == 0:
        testpath_4 = '/test/Lips_1.npz'
    else:
        testpath_4 = '/test/Lips_2.npz'
    testimg_4 = np.load(testpath_4)['img']
    test_lbl_anat_4 = np.load(testpath_4)['anatlbl']



    testimg = np.concatenate([testimg_1,testimg_2,testimg_3,testimg_4], axis=0)
    test_lbl_anat = np.concatenate([test_lbl_anat_1, test_lbl_anat_2, test_lbl_anat_3, test_lbl_anat_4], axis=0)



    return testimg, test_lbl_anat


def main():
    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/gpu:0"):
            image_orig = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 1])
            lblanat = tf.placeholder(dtype=tf.int64, shape=[None])

            # ----------------------Encoder-------------------------
            image_val = tf.expand_dims(image_orig, axis=3)

            with tf.variable_scope('Encoder_anat'):
                anat_fea_val, anat_res_scales_val, anat_saved_strides_val, anat_filters_val = residual_def.residual_encoder(
                    inputs=image_val,
                    num_res_units=1,
                    mode=tf.estimator.ModeKeys.EVAL,
                    filters=(8, 16, 32, 64, 8),
                    strides=((1, 1, 1), (2, 2, 1), (2, 2, 1), (2, 2, 1), (1, 1, 1)),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

            # ----------------------Encoder-------------------------

            with tf.variable_scope('Encoder_oths'):
                oths_fea_val, oths_res_scales_val, oths_saved_strides_val, oths_filters_val = residual_def.residual_encoder(
                    inputs=image_val,
                    num_res_units=1,
                    mode=tf.estimator.ModeKeys.EVAL,
                    filters=(8, 16, 32, 64, 8),
                    strides=((1, 1, 1), (2, 2, 1), (2, 2, 1), (2, 2, 1), (1, 1, 1)),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

            # ----------------------num_classification----------------------

            with tf.variable_scope('anat_cls'):
                anat_logits_val = residual_def.classify_dense_bn_relu(
                    anat_fea_val,
                    units=(128, 128),
                    is_train=False,
                    num_class=anat_num,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

            test_anat_softmax = tf.nn.softmax(anat_logits_val)
            test_anat_label = tf.argmax(test_anat_softmax, axis=1)
            loss_anat_softce = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=anat_logits_val,
                                                        labels=tf.one_hot(lblanat, depth=anat_num)))

        # ---------------------------------------------------
        config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

            data, lbl_anat = load_data(flag=0)
            data_new = np.reshape(data, (data.shape[0], height, width, 1))

            t_data = data_new
            t_anat_lbl = lbl_anat
            feed_dict = {image_orig: t_data, lblanat: t_anat_lbl}

            #return the last features for tsne plot
            prob_anat, pred_anat, loss_soft_anat, fea_last = sess.run(
                [test_anat_softmax, test_anat_label, loss_anat_softce, oths_fea_val],
                feed_dict=feed_dict)


            print ('lossce_anat: {}'.format(loss_soft_anat))


            # overall accuracy
            accuracy = accuracy_score(t_anat_lbl, pred_anat)
            print ("Overall Accuracy Anat= {:.4f}".format(accuracy))

            # precision, recall, f1score
            precision, recall, f1score, _ = precision_recall_fscore_support(t_anat_lbl, pred_anat, average=None,
                                                                            labels=[0, 1, 2, 3])

            # accuracy of each class and print the other measurement
            for i in range(0,4):

                right = 0
                index = np.where(t_anat_lbl == i)[0]

                y_true = t_anat_lbl[index]
                y_pred = pred_anat[index]

                for ss in range(len(index)):
                    if (y_true[ss] == i) and (y_pred[ss] == i):
                        right = right + 1

                print ("{:.4f}, {:.4f}, {:.4f}".format
                       (precision[i], recall[i], f1score[i]))


    return

main()
















