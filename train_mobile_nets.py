import tensorflow as tf
import tensorlayer as tl
import argparse
from  data.util import parse_function
import os
from losses.face_losses import arcface_loss
import time
from data.eval_data_reader import load_bin
from verification import ver_test
from nets.MobileFaceNet import inference

slim = tf.contrib.slim

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--epoch', default=100, help='epoch to train the network')
    parser.add_argument('--batch_size', default=256,type=int, help='batch size to train network')
    parser.add_argument('--lr_schedule', default=[4, 7, 9, 11], help='learning rate to train network')
    parser.add_argument('--weight_decay', default=5e-5, help='learning alg momentum')
    # parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_datasets', default=['asia'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./data', help='evluate datasets base path')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--num_output', default=57613, help='the image size')
    parser.add_argument('--tfrecords_file_path', default='./data', type=str,
                        help='path to the output of tfrecords file path')
    parser.add_argument('--summary_path', default='./output/summary/mobile', help='the summary file save path')
    parser.add_argument('--ckpt_path', default='./output/ckpt/mobile', help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', default=100, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--buffer_size', default=100000, help='tf dataset api buffer size')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--summary_interval', default=300, help='interval to save summary')
    parser.add_argument('--ckpt_interval', default=500, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', default=5000, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=20, help='intervals to show information')
    parser.add_argument('--num_gpus', default=1,type=int, help='the num of gpus')
    parser.add_argument('--tower_name', default='tower', help='tower name')
    args = parser.parse_args()
    return args


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # 1. define global parameters
    args = get_parser()
    global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
    inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
    images = tf.placeholder(name='img_inputs', shape=[None,112,112, 3], dtype=tf.float32)
    images_test = tf.placeholder(name='img_inputs', shape=[None,112,112, 3], dtype=tf.float32)
    labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)
    phase_train_placeholder = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=None,
                                                          name='phase_train')
    epoch = tf.Variable(name='epoch', initial_value=-1, trainable=False)
    inc_epoch_op = tf.assign_add(epoch, 1, name='increment_epoch')
    # splits input to different gpu
    images_s = tf.split(images, num_or_size_splits=args.num_gpus, axis=0)
    labels_s = tf.split(labels, num_or_size_splits=args.num_gpus, axis=0)
    # 2 prepare train datasets and test datasets by using tensorflow dataset api
    # 2.1 train datasets
    # the image is substracted 127.5 and multiplied 1/128.
    # random flip left right
    tfrecords_f = os.path.join(args.tfrecords_file_path, 'train.tfrecords')
    dataset = tf.data.TFRecordDataset(tfrecords_f)
    dataset = dataset.map(parse_function)
    dataset = dataset.shuffle(buffer_size=args.buffer_size)
    dataset = dataset.batch(args.batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    # 2.2 prepare validate datasets
    ver_list = []
    ver_name_list = []
    for db in args.eval_datasets:
        print('begin db %s convert.' % db)
        data_set = load_bin(db, args.image_size, args)
        ver_list.append(data_set)
        ver_name_list.append(db)
    # 3. define network, loss, optimize method, learning rate schedule, summary writer, saver
    # 3.1 inference phase
    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    # 3.2 define the learning rate schedule


    # lr = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=[0.0001, 0.0001, 0.0001, 0.0001],
    #                                  name='lr_schedule')

    learning_rate = tf.train.piecewise_constant(epoch, boundaries=args.lr_schedule,
                                                values=[0.001, 0.0001, 0.0001, 0.0001, 0.00001],
                                                name='lr_schedule')

    # 3.3 define the optimize method
    #opt = tf.train.MomentumOptimizer(learning_rate= learning_rate, momentum=args.momentum)
    opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)


    # Calculate the gradients for each model tower.
    tower_grads = []
    loss_dict = {}
    drop_dict = {}
    loss_keys = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(args.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (args.tower_name, i)) as scope:
            # net = get_resnet(images_s[i], args.net_depth, type='ir', w_init=w_init_method, trainable=True, keep_rate=dropout_rate)
            prelogits, net_points = inference(images_s[i], bottleneck_layer_size=128,
                                              phase_train=True, weight_decay=args.weight_decay)

            embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
            eps = 1e-5
            prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits) + eps, ord=1.0, axis=1))
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * 2e-5)
            w_init_method = slim.initializers.xavier_initializer()

            logit = arcface_loss(embedding=embeddings, labels=labels_s[i], w_init=w_init_method, out_num=args.num_output)

            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()
            # define the cross entropy
            inference_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels_s[i]))
            tf.add_to_collection('losses', inference_loss)

            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_loss = tf.add_n(regularization_losses,name='reg_loss')
            total_loss = tf.add_n([inference_loss] + regularization_losses, name='total_loss')

            # define weight deacy losses

            loss_dict[('inference_loss_%s_%d' % ('gpu', i))] = inference_loss
            loss_keys.append(('inference_loss_%s_%d' % ('gpu', i)))
            loss_dict[('wd_loss_%s_%d' % ('gpu', i))] = reg_loss
            loss_keys.append(('wd_loss_%s_%d' % ('gpu', i)))
            loss_dict[('total_loss_%s_%d' % ('gpu', i))] = total_loss
            loss_keys.append(('total_loss_%s_%d' % ('gpu', i)))
            grads = opt.compute_gradients(total_loss)
            tower_grads.append(grads)
            if i == 0:
                #test_net = get_resnet(images_test, args.net_depth, type='ir', w_init=w_init_method, trainable=False, keep_rate=dropout_rate)
                #embedding_tensor = test_net.outputs

                embedding_ten, net_points = inference(images_test, bottleneck_layer_size=128,
                                                  phase_train=False, weight_decay=args.weight_decay,reuse = True)
                embedding_tensor = tf.nn.l2_normalize(embedding_ten, 1, 1e-10, name='embedding_tensor')
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                pred = tf.nn.softmax(logit)
                acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), labels_s[i]), dtype=tf.float32))

    grads = average_gradients(tower_grads)
   
   #with tf.control_dependencies(update_ops):
        # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Track the moving averages of all trainable variables.
    moving_average_decay = 0.999
    variable_averages = tf.train.ExponentialMovingAverage( moving_average_decay, global_step) 
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies([apply_gradient_op, variables_averages_op] + update_ops):
        train_op = tf.no_op(name='train')


    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=config)
    # summary writer
    summary = tf.summary.FileWriter(args.summary_path, sess.graph)
    summaries = []
    # add grad histogram op
    for grad, var in grads:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
    # add trainabel variable gradients
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))
    # add loss summary
    for keys, val in loss_dict.items():
        summaries.append(tf.summary.scalar(keys, val))
    # add learning rate
    #summaries.append(tf.summary.scalar('leraning_rate', lr_schedule))
    summary_op = tf.summary.merge(summaries)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())
    # init all variables
    sess.run(tf.global_variables_initializer())

    #restore_saver = tf.train.Saver()
    #restore_saver.restore(sess, '/data/zzivenzhu/colabin/tf_facerec/output/ckpt/multi/InsightFace_iter_170000.ckpt')

    # begin iteration
    count = 0
    for i in range(args.epoch):
        sess.run(iterator.initializer)
        _ = sess.run(inc_epoch_op)
        while True:
            try:
                images_train, labels_train = sess.run(next_element)
                feed_dict = {images: images_train, labels: labels_train, dropout_rate: 0.6}
                start = time.time()
                _, _, inference_loss_val_gpu_1, wd_loss_val_gpu_1, total_loss_gpu_1,acc_val = sess.run([train_op, inc_op, loss_dict[loss_keys[0]],
                                                                         loss_dict[loss_keys[1]],
                                                                         loss_dict[loss_keys[2]],
                                                                             acc],
                                                                         feed_dict=feed_dict)
                end = time.time()
                pre_sec = args.batch_size/(end - start)
                # print training information
                if count > 0 and count % args.show_info_interval == 0:
                    # print('epoch %d, total_step %d, total loss gpu 1 is %.2f , inference loss gpu 1 is %.2f, weight deacy '
                    #       'loss gpu 1 is %.2f, total loss gpu 2 is %.2f , inference loss gpu 2 is %.2f, weight deacy '
                    #       'loss gpu 2 is %.2f, training accuracy is %.6f, time %.3f samples/sec' %
                    #       (i, count, total_loss_gpu_1, inference_loss_val_gpu_1, wd_loss_val_gpu_1, total_loss_gpu_2,
                    #        inference_loss_val_gpu_2, wd_loss_val_gpu_2, acc_val, pre_sec))

                    print('epoch %d, total_step %d, total loss: [ %.2f], inference loss: [%.2f], weight deacy loss: [%.2f], training accuracy is %.6f, time %.3f samples/sec' %
                          (i, count, total_loss_gpu_1,  inference_loss_val_gpu_1,wd_loss_val_gpu_1, acc_val, pre_sec))
                count += 1

                # save summary
                if count > 0 and count % args.summary_interval == 0:
                    feed_dict = {images: images_train, labels: labels_train, dropout_rate: 0.4}
                    summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                    summary.add_summary(summary_op_val, count)

                # save ckpt files
                if count > 0 and count % args.ckpt_interval == 0:
                    filename = 'InsightFace_iter_{:d}'.format(count) + '.ckpt'
                    filename = os.path.join(args.ckpt_path, filename)
                    saver.save(sess, filename)
                # # validate
                if count >= 0 and count % args.validate_interval == 0:
                    feed_dict_test ={dropout_rate: 1.0}
                    results = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=count, sess=sess,
                             embedding_tensor=embedding_tensor, batch_size=args.batch_size//args.num_gpus, feed_dict=feed_dict_test,
                             input_placeholder=images_test)
                    print('test accuracy is: ', str(results[0]))
                    if max(results) > 0.998:
                        print('best accuracy is %.5f' % max(results))
                        filename = 'InsightFace_iter_best_{:d}'.format(count) + '.ckpt'
                        filename = os.path.join(args.ckpt_path, filename)
                        saver.save(sess, filename)
            except tf.errors.OutOfRangeError:
                print("End of epoch %d" % i)
                break
            except tf.errors.InvalidArgumentError:
                print("InvalidArgumentError")
                break
