import os
import pickle
import glob
import time
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import config as cfg


class SolverWrapper(object):
    """
    A Wrapper Class for the training Process
    """

    def __init__(self, sess, network, output_dir, tb_dir):
        self.net = network
        self.sess = sess
        self.output_dir = output_dir
        self.tb_dir = tb_dir
        self.tb_dir_val = tb_dir + '_val'
        self.state = {}

    def snapshot(self, iter):
        """

        :param iter:
        :return:
        """
        net = self.net
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # Store model snapshot
        filename = cfg.TRAIN_SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(self.sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

        nfilename = cfg.TRAIN_SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pkl'
        nfilename = os.path.join(self.output_dir, nfilename)

        with open(nfilename, 'wb') as fid:
            pickle.dump(self.state, fid, pickle.HIGHEST_PROTOCOL)

        return filename, nfilename

    def from_snapshot(self, sfile, nfile):
        print('Restoring model snapshots from {:s}'.format(sfile))
        self.saver.restore(self.sess, sfile)
        print('Restored')
        with open(nfile, 'rb') as fid:
            self.state = pickle.load(fid)

        return self.state['last_snapshot_iter'], self.state['last_snapshot_rate']

    def find_snapshot(self):
        sfiles = os.path.join(self.output_dir, cfg.TRAIN_SNAPSHOT_PREFIX +
                              '_iter_*.ckpt.meta')
        sfiles = glob.glob(sfiles)
        sfiles.sort(key=os.path.getmtime)

        nfiles = [ss.replace('.ckpt.meta', '.pkl') for ss in sfiles]
        sfiles = [ss.replace('.meta', '') for ss in sfiles]

        lsf = len(sfiles)

        assert lsf == len(nfiles)

        return lsf, nfiles, sfiles

    def initialize(self):
        np_paths = []
        ss_paths = []
        last_snapshot_iter = 0
        print('Loading initial model weights from {:s}'.format(self.pretrained_model))
        variables = tf.global_variables()
        self.sess.run(tf.variables_initializer(variables, name='init'))
        var_keep_dic = self.get_variables_in_checkpoint_file(self.pretrained_model)
        variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)

        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(self.sess, self.pretrained_model)
        print('Loaded')

        rate = cfg.TRAIN_LEARNING_RATE

        return rate, last_snapshot_iter, np_paths, ss_paths

    def get_variables_in_checkpoint_file(self, file_name):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been"
                      "compressed with SNAPPY")

    def restore(self, sfile, nfile):
        np_paths = [nfile]
        ss_paths = [sfile]
        last_snapshot_iter, last_snapshot_rate = self.from_snapshot(sfile, nfile)

        return last_snapshot_iter, last_snapshot_rate, np_paths, ss_paths

    def remove_snapshot(self, np_paths, ss_paths):
        to_remove = len(np_paths) - cfg.TRAIN_SNAPSHOT_KEPT
        for c in range(to_remove):
            nfile = np_paths[0]
            os.remove(str(nfile))
            np_paths.remove(nfile)

        to_remove = len(ss_paths) - cfg.TRAIN_SNAPSHOT_KEPT
        for c in range(to_remove):
            sfile = ss_paths[0]
            if os.path.exists(sfile):
                os.remove(str(sfile))
            else:
                os.remove(str(sfile + '.data-00000-of-00001'))
                os.remove(str(sfile + '.index'))
            sfile_meta = sfile + '.meta'
            os.remove(str(sfile_meta))
            ss_paths.remove(sfile)

    def train_model(self):
        with self.sess.graph.as_default():
            tf.set_random_seed(cfg.RND_SEED)
            layers = self.net.create_architecture()
            loss = layers['total_loss']
            lr = tf.Variable(cfg.TRAIN_LEARNING_RATE, trainable=False)
            self.optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN_MOMENTUM)
            gvs = self.optimizer.compute_gradients(loss)
            train_op = self.optimizer.apply_gradients(gvs)

            self.saver = tf.train.Saver(max_to_keep=100000)
            self.writer = tf.summary.FileWriter(self.tb_dir, self.sess.graph)
            self.valwriter = tf.summary.FileWriter(self.tb_dir_val)

        lsf, nfiles, sfiles = self.find_snapshot()

        if lsf == 0:
            rate, last_snapshot_iter, np_paths, ss_paths = self.initialize()
        else:
            rate, last_snapshot_iter, np_paths, ss_paths = \
            self.restore(str(sfiles[-1]), str(nfiles[-1]))

        iter = last_snapshot_iter + 1
        last_summary_time = time.time()
        while iter < max_iters + 1:
            if iter % (cfg.TRAIN_LR_REDUCTION) == 0:
                self.snapshot(iter)
                rate *= cfg.TRAIN_LR_GAMMA
                self.sess.run(tf.assign(lr, rate))

            data = self._get_train_data()

            now = time.time()
            if iter == 1 or iter % (cfg.TRAIM_SUMMARY_INTERVAL) == 0:







