import os
import tensorflow as tf
import numpy as np
import residual_def
import pdb
import random


height = 224
width = 288
batch_size = 2
lr = 1e-4
model_dir = './model'
logs_path = './model'
max_iter_step = 30010
anat_num = 4
seed = 24
val_imgnum = 25
Beta_alpha = 0.75


def read_decode(filename_queue, minibatch):
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
        features={"img": tf.FixedLenFeature([],tf.string),
                  "anatlbl": tf.FixedLenFeature([], tf.int64),
                  "funclbl": tf.FixedLenFeature([], tf.int64)})
    image = tf.decode_raw(features["img"], tf.float32)
    Anat_label = tf.cast(features["anatlbl"], tf.int64)
    Func_label = tf.cast(features["funclbl"], tf.int64)
    image = tf.reshape(image, [height, width, 1])
    images, Anat_labels, Func_labels = tf.train.batch([image, Anat_label, Func_label], batch_size=minibatch, capacity=1000, num_threads=8)
    # images and labels are tensor object
    return images, Anat_labels, Func_labels

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise

def load():

    # load data to keep the balance in each big batch, the batch size here is the minibatch
    # load labelled data from device A

    filename_1 = '/train/Abdominal_1.tfrecords'
    filename_queue_1 = tf.train.string_input_producer([filename_1])
    image_1, anat_lbl_1, func_lbl_1 = read_decode(filename_queue_1, batch_size)

    filename_2 = '/train/Brain_1.tfrecords'
    filename_queue_2 = tf.train.string_input_producer([filename_2])
    image_2, anat_lbl_2, func_lbl_2 = read_decode(filename_queue_2, batch_size)

    filename_3 = '/train/Femur_1.tfrecords'
    filename_queue_3 = tf.train.string_input_producer([filename_3])
    image_3, anat_lbl_3, func_lbl_3 = read_decode(filename_queue_3, 2 * batch_size)

    filename_4 = '/train/Lips_1.tfrecords'
    filename_queue_4 = tf.train.string_input_producer([filename_4])
    image_4, anat_lbl_4, func_lbl_4 = read_decode(filename_queue_4, 2 * batch_size)


    # load labelled data from device B

    filename_S_1 = '/train/Abdominal_2.tfrecords'
    filename_queue_S_1 = tf.train.string_input_producer([filename_S_1])
    image_S_1, anat_lbl_S_1, func_lbl_S_1 = read_decode(filename_queue_S_1, batch_size)

    filename_S_2 = '/train/Brain_2.tfrecords'
    filename_queue_S_2 = tf.train.string_input_producer([filename_S_2])
    image_S_2, anat_lbl_S_2, func_lbl_S_2 = read_decode(filename_queue_S_2, batch_size)

    # load unlabelled data
    filename_U_0 = '/train/train_unlabelled.tfrecords'
    filename_queue_U_0 = tf.train.string_input_producer([filename_U_0])
    image_U_0, anat_lbl_U_0, func_lbl_U_0 = read_decode(filename_queue_U_0, 4*4*batch_size)

    # data
    image = tf.concat([image_1, image_S_1, image_2, image_S_2, image_3, image_4, image_U_0], 0)
    anatlbl_anat = tf.concat(
        [anat_lbl_1, anat_lbl_S_1, anat_lbl_2, anat_lbl_S_2, anat_lbl_3, anat_lbl_4, anat_lbl_U_0], 0)

    funclbl_func = tf.concat([func_lbl_1, func_lbl_S_1, func_lbl_2, func_lbl_S_2,
                                  func_lbl_3, func_lbl_4, func_lbl_U_0], 0)

    print (image.shape, anatlbl_anat.shape, funclbl_func.shape)

    return image, anatlbl_anat, funclbl_func

def MINE(x_conc, y_conc, H):

    layerx = tf.contrib.layers.linear(x_conc, H)
    layery = tf.contrib.layers.linear(y_conc, H)
    layer2 = tf.nn.leaky_relu(layerx + layery)
    output = tf.contrib.layers.linear(layer2, 1)

    return output


def mixup_data(img,lbl_label, lbl_unlabel, alpha):

    lbl_labelled = tf.one_hot(lbl_label[0: 8 * batch_size], depth=anat_num)

    lbl = tf.concat([lbl_labelled, lbl_unlabel], axis=0)

    lam = tf.distributions.Beta(alpha, alpha).sample([24 * batch_size, 1, 1, 1, 1])
    lam = tf.maximum(lam, 1 - lam)

    index = tf.random_shuffle(tf.range(24 * batch_size))
    img_shuffle = tf.gather(img, index)
    lbl_shuffle = tf.gather(lbl, index)

    mix_img = img * lam + img_shuffle * (1 - lam)
    mix_label = lbl * lam[:,:,0,0,0] + lbl_shuffle * (1 - lam[:,:,0,0,0])


    return mix_img, mix_label

def MixUp_loss(logits, label):

    mseresult = tf.square(label - tf.nn.softmax(logits))
    mseresult = tf.reduce_mean(mseresult)

    return mseresult

def invariance_loss(fea):

    # fea = tf.contrib.layers.flatten(fea)
    fea_0_mean = tf.reduce_mean(fea[0*batch_size:1*batch_size, :], axis=0, keep_dims=True)
    fea_1_mean = tf.reduce_mean(fea[2*batch_size:3 * batch_size, :], axis=0, keep_dims=True)


    fea_M_0_mean = tf.reduce_mean(fea[1 * batch_size:2 * batch_size, :], axis=0, keep_dims=True)
    fea_M_1_mean = tf.reduce_mean(fea[3 * batch_size:4 * batch_size, :], axis=0, keep_dims=True)

    # mse loss
    dis_0 = tf.reduce_mean(tf.pow(tf.subtract(fea_0_mean, fea_M_0_mean), 2))
    dis_1 = tf.reduce_mean(tf.pow(tf.subtract(fea_1_mean, fea_M_1_mean), 2))


    dist_all = dis_0 + dis_1

    return dist_all

def build_gpu():

    with tf.device("/gpu:0"):

        image_orig, anat_lbl_anat, func_lbl_func = load()

        # data augmentation: adding noise
        image_noise = gaussian_noise_layer(image_orig, 0.1)
        # data augmentation: random flip
        image_squz = tf.transpose(tf.squeeze(image_noise, axis=3), [1, 2, 0])
        image_flip = tf.image.random_flip_left_right(image_squz)
        image_flip = tf.expand_dims(tf.transpose(image_flip, [2, 0, 1]), axis=3)

        image = tf.expand_dims(image_flip, axis=3)

        w_info = tf.Variable(1e-4, dtype=tf.float32, trainable=False)
        w_reco = tf.Variable(1, dtype=tf.float32, trainable=False)
        w_cls = tf.Variable(10, dtype=tf.float32, trainable=False)
        w_internal = tf.Variable(50, dtype=tf.float32, trainable=False)
        w_invar = tf.Variable(50, dtype=tf.float32, trainable=False)
        l_r = tf.Variable(lr, dtype=tf.float32, trainable=False)

        opt_cls = tf.train.MomentumOptimizer(learning_rate=l_r, momentum=0.9)

        # ----------------------Encoder-------------------------

        with tf.variable_scope('Encoder_anat'):
            anat_fea, anat_res_scales, anat_saved_strides, anat_filters = residual_def.residual_encoder(
                inputs=image,
                num_res_units=1,
                mode=tf.estimator.ModeKeys.TRAIN,
                filters=(8, 16, 32, 64, 8),
                strides=((1, 1, 1), (2, 2, 1), (2, 2, 1), (2, 2, 1), (1, 1, 1)),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

        # ----------------------Encoder-------------------------

        with tf.variable_scope('Encoder_oths'):
            oths_fea, oths_res_scales, oths_saved_strides, oths_filters = residual_def.residual_encoder(
                inputs=image,
                num_res_units=1,
                mode=tf.estimator.ModeKeys.TRAIN,
                filters=(8, 16, 32, 64, 8),
                strides=((1, 1, 1), (2, 2, 1), (2, 2, 1), (2, 2, 1), (1, 1, 1)),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

        # ----------------------num_classification----------------------

        with tf.variable_scope('anat_cls'):
            anat_logits = residual_def.classify_dense_bn_relu(
                anat_fea,
                units=(128, 128),
                is_train=True,
                num_class=anat_num,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))


        # ----------------------reconstrunction----------------------
        combine_fea = tf.concat([anat_fea, oths_fea], -1)

        with tf.variable_scope('reconstruction'):
            net_output_ops = residual_def.residual_decoder(
                inputs=combine_fea,
                num_classes=1,
                num_res_units=1,
                mode=tf.estimator.ModeKeys.TRAIN,
                filters=oths_filters,
                res_scales=oths_res_scales,
                saved_strides=oths_saved_strides,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

            logits = net_output_ops['logits']



        # ------------------------MixUp------------------------------------

        anat_unlabeled_prob = tf.nn.softmax(anat_logits)
        mix_img, mix_label = mixup_data(image, anat_lbl_anat[0: 8 * batch_size],
                                        anat_unlabeled_prob[8 * batch_size:, :], Beta_alpha)

        with tf.variable_scope('Encoder_anat', reuse=True):
            mix_anat_fea, mix_anat_res_scales, mix_anat_saved_strides, mix_anat_filters = residual_def.residual_encoder(
                inputs=mix_img,
                num_res_units=1,
                mode=tf.estimator.ModeKeys.TRAIN,
                filters=(8, 16, 32, 64, 8),
                strides=((1, 1, 1), (2, 2, 1), (2, 2, 1), (2, 2, 1), (1, 1, 1)),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

        with tf.variable_scope('anat_cls', reuse=True):
            mix_img_logits = residual_def.classify_dense_bn_relu(
                mix_anat_fea,
                units=(128, 128),
                is_train=True,
                num_class=anat_num,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

        # ------------------------MINE-----------------------------------------------
        H = 2048
        x_in = anat_fea
        y_in = oths_fea

        y_trans = tf.transpose(y_in, perm=[4, 0, 1, 2, 3])
        y_shuffle = tf.gather(y_trans, tf.random_shuffle(tf.range(8)))
        y_shuffle_trans = tf.transpose(y_shuffle, perm=[1, 2, 3, 4, 0])

        with tf.variable_scope('mine_joint'):
            joint = MINE(x_in, y_in, H)
        with tf.variable_scope('mine_marginal'):
            marginal = MINE(x_in, y_shuffle_trans, H)

        T_xy = joint
        T_x_y = marginal


        # ----------------------classification Loss--------------------------

        labels_onehot_anat = tf.one_hot(anat_lbl_anat[0: 8* batch_size], depth=anat_num)
        anat_cls_loss_labeled = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=anat_logits[0: 8* batch_size], labels=labels_onehot_anat))
        reg_anat = tf.losses.get_regularization_loss('anat_cls')
        anat_cls_loss = anat_cls_loss_labeled + reg_anat

        #-----------------------reconstruction loss---------------------------
        lossrecon = tf.reduce_mean(tf.pow(tf.subtract(logits, image), 2))
        reg_recon = tf.losses.get_regularization_loss('reconstruction')

        recon_loss = lossrecon + reg_recon

        # ----------------------information Loss--------------------------
        info_loss = -(tf.reduce_mean(T_xy) - tf.log(tf.reduce_mean(tf.exp(T_x_y)) + 1e-1))

        t_xy = tf.reduce_mean(T_xy)
        t_x_y = tf.log(tf.reduce_mean(tf.exp(T_x_y)) + 1e-1)


        print (T_xy.shape, T_x_y.shape, info_loss.shape)

        # -------------------------internal loss-----------------------------------------
        interloss = MixUp_loss(mix_img_logits, mix_label)
        invarloss = invariance_loss(anat_logits)


        # -----------------------Total loss
        loss = w_cls * anat_cls_loss + w_reco * recon_loss + w_info * info_loss + w_internal * interloss + w_invar * invarloss

        # ------------------optimization----------------------------
        mine_joint_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'mine_joint')
        mine_marginal_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'mine_marginal')
        encoder_num_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Encoder_anat')
        encoder_domian_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Encoder_oths')
        cls_num_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'anat_cls')
        recon_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'reconstruction')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt_cls.minimize(loss, var_list=[encoder_num_var, encoder_domian_var, cls_num_var, recon_var])

        MINE_opt = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(info_loss,
                                                                       var_list=[mine_joint_var, mine_marginal_var])
        weight_decay = 5e-4
        with tf.control_dependencies([MINE_opt]):
            l2_loss_1 = weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in mine_joint_var])
            l2_loss_2 = weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in mine_marginal_var])
            sgd = tf.train.GradientDescentOptimizer(learning_rate=1.0)
            decay_op = sgd.minimize(l2_loss_1 + l2_loss_2)


    return train_op, MINE_opt, decay_op, anat_cls_loss, recon_loss, info_loss, interloss, invarloss, loss, \
           w_cls, w_reco, w_info, w_internal, w_invar, \
           image, logits, t_xy, t_x_y

def main():
    train_op, MINE_opt, decay_op, anat_cls_loss, recon_loss, info_loss, interloss, invarloss, loss, \
    w_cls, w_reco, w_info, w_internal, w_invar, \
    image, logits, t_xy, t_x_y = build_gpu()

    # ----------------validation---------------------------------
    image_orig = tf.placeholder(dtype=tf.float32, shape=[val_imgnum, height, width, 1])
    lblanat = tf.placeholder(dtype=tf.int64, shape=[val_imgnum])

    # ----------------------Encoder-------------------------
    image_val = tf.expand_dims(image_orig, 3)

    with tf.variable_scope('Encoder_anat', reuse=True):
        anat_fea_val, anat_res_scales_val, anat_saved_strides_val, anat_filters_val = residual_def.residual_encoder(
            inputs=image_val,
            num_res_units=1,
            mode=tf.estimator.ModeKeys.EVAL,
            filters=(8, 16, 32, 64, 8),
            strides=((1, 1, 1), (2, 2, 1), (2, 2, 1), (2, 2, 1), (1, 1, 1)),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

    # ----------------------Encoder-------------------------

    with tf.variable_scope('Encoder_oths', reuse=True):
        oths_fea_val, oths_res_scales_val, oths_saved_strides_val, oths_filters_val = residual_def.residual_encoder(
            inputs=image_val,
            num_res_units=1,
            mode=tf.estimator.ModeKeys.EVAL,
            filters=(8, 16, 32, 64, 8),
            strides=((1, 1, 1), (2, 2, 1), (2, 2, 1), (2, 2, 1), (1, 1, 1)),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

    # ----------------------num_classification----------------------

    with tf.variable_scope('anat_cls', reuse=True):
        anat_logits_val = residual_def.classify_dense_bn_relu(
            anat_fea_val,
            units=(128, 128),
            is_train=False,
            num_class=anat_num,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

    # ----------------------Loss--------------------------
    onehot_anat = tf.one_hot(lblanat, depth=anat_num)
    anat_cls_loss_val = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=anat_logits_val, labels=onehot_anat))


    val_anat_label = tf.argmax(tf.nn.softmax(anat_logits_val), axis=1)

    # -----------------------------------------------------------

    saver = tf.train.Saver(max_to_keep=5)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    tf.set_random_seed(seed)
    np.random.seed(seed)

    with tf.Session(config=config) as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Create a summary to monitor cost tensor
        tf.summary.scalar("anat_cls_loss", anat_cls_loss)
        tf.summary.scalar("recon_loss", recon_loss)
        tf.summary.scalar("info_loss", info_loss)
        tf.summary.scalar("interloss", interloss)
        tf.summary.scalar("invarloss", invarloss)
        tf.summary.scalar("loss", loss)

        tf.summary.scalar("anat_cls_loss_val", anat_cls_loss_val)

        tf.summary.image('image', image[:,:,:,:,0], tf.float32)
        tf.summary.image('logits', logits[:,:,:,:,0], tf.float32)

        tf.summary.image('image_val', image_val[:,:,:,:,0], tf.float32)
        tf.summary.image('logits_val', logits_val[:,:,:,:,0], tf.float32)

        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        sess.run(init_op)
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        valpath_S_0 = '/val/Femur_2.npz'
        valimg_S_0 = np.load(valpath_S_0)['img']
        val_lbl_anat_S_0 = np.load(valpath_S_0)['anatlbl']

        valpath_S_1 = '/val/Lips_2.npz'
        valimg_S_1 = np.load(valpath_S_1)['img']
        val_lbl_anat_S_1 = np.load(valpath_S_1)['anatlbl']

        valimg = np.concatenate([valimg_S_0, valimg_S_1], 0)
        val_lbl_anat = np.concatenate([val_lbl_anat_S_0, val_lbl_anat_S_1], 0)

        print ('validation number')
        print (valimg.shape)


        for i in range(max_iter_step):

            indexsel = random.sample(range(0, valimg.shape[0]), val_imgnum)
            t_data = np.reshape(valimg[indexsel, :, :], (val_imgnum, height, width, 1))
            t_anatlbl = np.reshape(val_lbl_anat[indexsel], (val_imgnum))
            feed_dict = {image_orig: t_data, lblanat: t_anatlbl}

            _, summary = sess.run([train_op, merged_summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary, i)

            for j in range(1):
                _, _, summary = sess.run([MINE_opt, decay_op, merged_summary_op], feed_dict=feed_dict)
                summary_writer.add_summary(summary, i)

            anatClsLoss, reconLoss, infoLoss, interLoss, invarLoss, Loss, \
            anatClsLoss_val, preanat = sess.run(
                [anat_cls_loss, recon_loss, info_loss, interloss, invarloss, loss,
                 anat_cls_loss_val, val_anat_label],
                feed_dict=feed_dict)

            right_anat_num = 0
            for ss in range(val_imgnum):
                if ((t_anatlbl[ss] - preanat[ss]) == 0):
                    right_anat_num = right_anat_num + 1

            acc_anat = right_anat_num / val_imgnum


            if i % 100 == 0:
                print("i = %d" % i)
                print ("Anat Cls Loss = {}".format(anatClsLoss))
                print ("Anat Cls_val = {}".format(anatClsLoss_val))
                print ("Recon Loss = {}".format(reconLoss))
                print ('Info loss = {}'.format(infoLoss))
                print ('Internal loss = {}'.format(interLoss))
                print ('Invariance loss = {}'.format(invarLoss))
                print ("Loss all = {}".format(Loss))

                print ('val_anat = {}'.format(acc_anat))

            if i % 500 == 0:
                saver.save(sess, os.path.join(model_dir, "model.val"), global_step=i)

        coord.request_stop()
        coord.join(threads)

main()
















