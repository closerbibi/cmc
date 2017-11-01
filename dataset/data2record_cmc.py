## create script that download datasets and transform into tf-record
## Assume the datasets is downloaded into following folders
## mjsyth datasets(41G)
## data/sythtext/*

import numpy as np 
import pdb
import os, os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
sys.path.append('/home/closerbibi/workspace/data/scannet-segmentation')
import visualize_cube as vscube
import tensorflow as tf
import re
from dataset.dataset_utils import int64_feature, float_feature, bytes_feature ,ImageCoder, norm
import glob
import SimpleITK as sitk
from random import shuffle
from dataset.utils import writeImage, writeMedicalImage, fast_hist
import scipy.ndimage
import h5py
import matplotlib.pyplot as plt

from PIL import Image

tf.app.flags.DEFINE_string(
 'train_data', '/media/disk3/data/scannet_task/voxel_labeling/train-samples',
 'Directory of the datasets')

tf.app.flags.DEFINE_string(
 'val_data', '/media/disk3/data/scannet_task/voxel_labeling/train-samples/',
 'Directory of the datasets')

tf.app.flags.DEFINE_string(
 'path_save', '/media/disk3/data/scannet_task/voxel_labeling/tfrecord/',
 'Dictionary to save sythtext record')

tf.app.flags.DEFINE_boolean(
 'is_training', True,
 'save train data or val data')

FLAGS = tf.app.flags.FLAGS

## SythText datasets is too big to store in a record. 
## So Transform tfrecord according to dir name

def label_mapping(label):
    nonuse = [13,15,17,18,19,20,21,22,23,25,26,27,29,30,31,32,35,37,38,39,255]
    if label in nonuse:
        new_label = 40
    else:
        new_label = label
    return new_label

def _convert_to_example(image_data, label):
    #print 'shape: {}, height:{}, width:{}'.format(shape,shape[0],shape[1])
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': bytes_feature(image_data),
            'label/encoded': bytes_feature(label)
            }))
    return example


def _processing_image(seq, label, depth):
    seqs = []
    labs = []
    for d in [depth-2, depth-1, depth]:
        label_data = label[d]
        labs.append(np.array(label_data))
        mod = []
        for im in seq:
            image_data = im[d]
            #image_data = scipy.ndimage.interpolation.zoom(image_data, 2, order=1, mode='nearest')
            # upsample 
            mod.append(image_data)
        seqs.append(np.array(mod))
    seqs = np.array(seqs)
    labs = np.array(labs)
    return seqs.tobytes(), labs.tobytes()

def _processing_image_single(seq, label, depth):
    mod = []
    for im in seq:
        image_data = im[depth]
        #image_data = scipy.ndimage.interpolation.zoom(image_data, 2, order=1, mode='nearest')
        # upsample 
        mod.append(image_data)
    mod = np.array(mod)
    label_data = label[depth]
    # upsample
    #label_data = scipy.ndimage.interpolation.zoom(label_data, 2, order=1, mode='nearest')
    shape = list(mod.shape)
    #if (mod.shape[0] != 4 or mod.shape[1] != 240 or mod.shape[2] != 240):
    #	print(shape)
    #	print("SHIT")
    #	exit()
    # 4,240,240
    return mod.tobytes(), label_data.tobytes(), shape

def norm_image_by_cube(im):
    #im = sitk.GetArrayFromImage(sitk.ReadImage(imname)).astype(np.float32)
    return (im - im.mean()) / im.std()
    roi_index = im > 0
    mean = im[roi_index].mean()
    std = im[roi_index].std()
    im[roi_index] -= mean
    im[roi_index] /= std
    return im

def get_image_label(h5name):
    data = h5py.File(h5name)
    im = np.transpose(data['data'], (1,0,2,3,4))
    imoc = im[0]
    imvi = im [1]
    label = data['label']
    return imoc, imvi, label

def count_class_freq(label_batch):
  cls_num = 22
  imagesize = 32
  hist = np.zeros(cls_num)
  imagesPresent = [0] * cls_num
  for i in range(len(label_batch)):
    new_hist = np.bincount(label_batch[i], minlength=cls_num)
    hist += new_hist
    for ii in range(cls_num):
        if (new_hist[ii] != 0):
            imagesPresent[ii] += 1
  print(hist)
  freqs = [hist[v]/float((imagesPresent[v]+1e-5)*32*32) for v in range(cls_num)]
  median = np.median(freqs)
  o = []
  for i in range(cls_num):
      if (freqs[i] <= 1e-5):
          o.append(0.0)
      else:
          o.append(float(median)/(freqs[i]))
  print(o)
  return o

def checkLabel(label, d):
    if np.count_nonzero(label[d]) > 0:
        return True, 1
    else:
        return False, 0

def count_freq(labels):
    #freq = np.array([0.0,0.0,0.0,0.0,0.0])
    freq = np.zeros((22,))
    for la in labels:
        freq += np.bincount(la, minlength=22).astype(np.float32)
    print(freq)
    print(freq/freq.sum())
    count_class_freq(labels)

def run():
    folderTrain = glob.glob(FLAGS.train_data + '/scene*.h5')
    #folderTest = glob.glob( + '/test*.h5')
    folder_train = folderTrain[:300] ############### test
    folder_val = folderTrain[-50:] ######## test
    tf_filename = FLAGS.path_save+'scannet_train32.tfrecord'
    all_example = []
    print("Saving training record....")
    all_label_data = []
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for index, i in enumerate(folder_train):
            ### test
            #if index==1:
            #    break
            print("%d, %s"%(index,i))

            ##### change dataset
            imoc, imvi, labels = get_image_label(i)
            
            #### test
            for im_ind in range(imoc.shape[0]):
            #for im_ind in range(100):
                im0 = imoc[im_ind].astype('float32')
                im1 = imvi[im_ind].astype('float32')
                label = labels[im_ind].astype('float32')
                seq = [im0, im1]

                for depth in range(2,62):
                    is_valid, sample_num = checkLabel(label, depth)
                    if ( not is_valid):
                        continue
                    for i in range(sample_num):
                        # image_data is the array containing three image(slices)
                        image_data, label_data = _processing_image(seq, label, depth)
                        # for freq. count
                        all_label_data.append(label[depth].flatten().astype(np.int64))
                        example = _convert_to_example(image_data, label_data)
                        all_example.append(example)
                        assert(len(image_data)==len(label_data)*2)
                #tfrecord_writer.write(example.SerializeToString()) 
        count_freq(all_label_data)
        print("slices:", len(all_example))
        shuffle(all_example)
        for ex in all_example:
            tfrecord_writer.write(ex.SerializeToString()) 
        # [0.011868184281122324, 1.0859737711507338, 0.80660914716121235, 0.0, 1.0]
    print 'Transform to tfrecord finished'
    print("Saving validation record....")
    for index, i in enumerate(folder_val):
        print("%d, %s"%(index,i))
        ##### change dataset
        imoc, imvi, labels = get_image_label(i)
        imname = i.split("/")[-1].split('.')[0]

        ### test
        #if index==1:
        #    break
        #for im_ind in range(1):
        for im_ind in range(imoc.shape[0]):
            tf_filename = FLAGS.path_save+'val/'+imname+'.tfrecord'
            with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
                ##### change dataset
                im0 = imoc[im_ind].astype('float32')
                im1 = imvi[im_ind].astype('float32')
                label = labels[im_ind].astype('float32')
                seq = [im0, im1]

                ind = 0
                for depth in range(62):
                    ind += 1
                    image_data, label_data, shape = _processing_image_single(seq, label, depth)
                    example = _convert_to_example(image_data, label_data)
                    tfrecord_writer.write(example.SerializeToString()) 

    print 'Transform to tfrecord finished'

if __name__ == '__main__':
    run()




