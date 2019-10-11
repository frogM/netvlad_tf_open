import sys, os
sys.path.append('/home/frog/Desktop/netvlad_tf_open/python')
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import netvlad_tf.nets as nets

import cv2
import numpy as np
import scipy.io as scio
import tensorflow as tf
import time

import netvlad_tf.net_from_mat as nfm

def testVgg16NetvladPca():   
    tf.reset_default_graph()
    image_batch = tf.placeholder(
            dtype=tf.float32, shape=[None, None, None, 3])
    net_out = nets.vgg16NetvladPca(image_batch)
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    saver.restore(sess, nets.defaultCheckpoint())

    print('Initialization Finished!')

    overwrite = False
    data_name =  "corridor"
    dataset_id = 2
    
    data_path = f"/home/frog/Desktop/lifelongSLAM/Examples/RGB-D/data/{data_name}-1-package/{data_name}-1-{dataset_id}"

    out_put_path = data_path + "/vlad"
    if  not os.path.exists(out_put_path):
        os.mkdir(out_put_path)
    
    data_names = open(os.path.join(data_path, "color.txt"))

    for i_data_name in data_names:
        i_time = i_data_name.strip('\n').split(' ')[0]
        i_name = i_data_name.strip('\n').split(' ')[-1]
        im2dl = os.path.join(data_path, i_name)
        if( os.path.isfile(im2dl) and np.all(cv2.imread(im2dl)!=None)):
            
            start = time.time()
            dl2vlad = os.path.join(out_put_path, f"{i_time}.txt")
            if os.path.exists(dl2vlad) and not overwrite:
                continue
            img = cv2.imread(im2dl)
            if isinstance(img, type(np.nan)):
                continue
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

            #dl2vlad = "/home/wangrong/netvlad_tf_open-master/result.txt"
            #4096
            np.savetxt(dl2vlad, result, fmt='%f')

            # flag = "/home/frog/Desktop/lifelongSLAM/Examples/RGB-D/result/flag/flag_%d.txt" % id
            # with open(flag,"w") as f:
            #     f.write("1")
            
            # id = id + 1

if __name__ == '__main__':
    testVgg16NetvladPca()
