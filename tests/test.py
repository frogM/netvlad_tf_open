import sys
sys.path.append('/home/frog/Desktop/netvlad_tf_open/python')
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import netvlad_tf.nets as nets

import cv2
import numpy as np
import scipy.io as scio
import tensorflow as tf
import time
import unittest

import netvlad_tf.net_from_mat as nfm

class TestNets(unittest.TestCase):
    def testVgg16NetvladPca(self):
       
        tf.reset_default_graph()

        image_batch = tf.placeholder(
                dtype=tf.float32, shape=[None, None, None, 3])

        net_out = nets.vgg16NetvladPca(image_batch)
        saver = tf.train.Saver()

        sess = tf.Session()
        saver.restore(sess, nets.defaultCheckpoint())

        rimg = cv2.imread('data/images/office_1.png')
        rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)

        rbatch = np.expand_dims(rimg, axis=0)

        #%% Generate TF results
        for _ in range(2):
            sess.run(net_out, feed_dict={image_batch: rbatch})
        rt = time.time()
        rresult = sess.run(net_out, feed_dict={image_batch: rbatch})
        print('Took %f seconds' % (time.time() - rt))


        qimg = cv2.imread('data/images/office_3.png')
        qimg = cv2.cvtColor(qimg, cv2.COLOR_BGR2RGB)

        qbatch = np.expand_dims(qimg, axis=0)

        #%% Generate TF results
        for _ in range(2):
            sess.run(net_out, feed_dict={image_batch: qbatch})
        qt = time.time()
        qresult = sess.run(net_out, feed_dict={image_batch: qbatch})
        print('Took %f seconds' % (time.time() - qt))

        #%% Compare final output
        out_diff = np.abs(qresult - rresult)
        print('Image presentation distance: %f', np.linalg.norm(out_diff))
        #self.assertLess(np.linalg.norm(out_diff), 0.0053)
        #print('Error of final vector is %f' % np.linalg.norm(out_diff))

if __name__ == '__main__':
    unittest.main()
