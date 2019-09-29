import sys, os
sys.path.append('/home/wangrong/netvlad_tf_open-master/python')
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

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

        print('Initialization Finished!')
    
        id = 0

        while True:
            im2dl = "/home/wangrong/orbslam2_modified/ORB_SLAM2_modified-master/Examples/RGB-D/img/img_%d.png" % id
            #im2dl = "/home/wangrong/netvlad_tf_open-master/example.jpg"
            if( os.path.isfile(im2dl) and np.all(cv2.imread(im2dl)!=None)):
              
                start = time.time()

                img = cv2.imread(im2dl)
                #if img == None:
                #    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                batch = np.expand_dims(img, axis=0)

                #%% Generate TF results
                for _ in range(2):
                    sess.run(net_out, feed_dict={image_batch: batch})
                result = sess.run(net_out, feed_dict={image_batch: batch})
                
                end = time.time()
                print('Took %f seconds' % (end - start))

                dl2vlad = "/home/wangrong/orbslam2_modified/ORB_SLAM2_modified-master/Examples/RGB-D/vlad/vlad_%d.txt" % id
                #dl2vlad = "/home/wangrong/netvlad_tf_open-master/result.txt"
                #4096
                np.savetxt(dl2vlad, result, fmt='%f')

                flag = "/home/wangrong/orbslam2_modified/ORB_SLAM2_modified-master/Examples/RGB-D/flag/flag_%d.txt" % id
                with open(flag,"w") as f:
                    f.write("1")
                
                id = id + 1

if __name__ == '__main__':
    unittest.main()
