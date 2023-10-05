import os
import tensorflow as tf
import scipy.io as sio
import numpy as np
import time
from datetime import datetime
import sys
import random
import math
from sklearn.metrics import roc_auc_score
import numpy.linalg
pi = math.pi
import argparse
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--train_size',type=int)
parser.add_argument('--val_size', type=int, default=5000)
parser.add_argument('--image_size',type = int)
parser.add_argument('--num_channels',type = int,  default=4)
parser.add_argument('--batch_size',type = int, default=250)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--signal_path',type=str)
parser.add_argument('--signal_name',type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument('--tied_toggle',type=int, default=1)
parser.add_argument('--validation',type=int, default=0)
parser.add_argument('--train_offset',type=int, default=30000)
parser.add_argument('--val_offset',type=int, default=5000)
parser.add_argument('--save_path', type=str)

args = parser.parse_args()

BATCH_SIZE = args.batch_size #in images, not image pairs
IMAGE_SIZE = args.image_size
NUM_CHANNELS = args.num_channels
TIED = True

NUM_EPOCHS = args.num_epochs
#NUM_EPOCHS = max(4000000.0 / args.train_size, 500)
TRAIN_SIZE = 2*args.train_size#2*250 #in images, not image pairs
VAL_SIZE = 2*args.val_size#2*5000 #in images, not image pairs

if (args.tied_toggle==1):
  TIED_TOGGLE = "T"
else:
  TIED_TOGGLE = "UT"

if (args.validation==1):
  STOP_METHOD = "Val"
else:
  STOP_METHOD = "Train"

ID = "TMI_%sLS_%s_%i_%sStop" % (NUM_CHANNELS, TIED_TOGGLE, int(TRAIN_SIZE/2), STOP_METHOD)

CONTINUE = False
BASE_DIR = args.save_path + "/%s/" % ID
CKPT_NAME = BASE_DIR + ("checkpoints/AETSI_%iLS_%s.ckpt" % (NUM_CHANNELS, TIED_TOGGLE))
CKPT_DIR = BASE_DIR + "checkpoints"

print(CKPT_DIR)

if not os.path.exists(CKPT_DIR):
    #os.makedirs(BASE_DIR)
    os.makedirs(CKPT_DIR)
    os.makedirs(BASE_DIR + 'output')

PREFIX_IMAGE = BASE_DIR + 'output/images.dat'
PREFIX_DATAGEN = BASE_DIR + 'output/channels.dat'

#SIGNAL = sio.loadmat('../../Data/elliptical_signal.mat')
#SIGNAL = SIGNAL['elliptical_signal']

TRAIN_DATA = sio.loadmat(args.data_path + '/train.mat')
TRAIN_DATA = TRAIN_DATA['train']
SIGNAL = np.reshape(np.mean(TRAIN_DATA[args.train_offset:args.train_offset + int(TRAIN_SIZE/2)] - TRAIN_DATA[0:int(TRAIN_SIZE/2)], axis=0), [IMAGE_SIZE, IMAGE_SIZE], order='F')
TRAIN_DATA = np.concatenate((TRAIN_DATA[0:int(TRAIN_SIZE/2)],  TRAIN_DATA[args.train_offset:args.train_offset + int(TRAIN_SIZE/2)]), axis=0)
fname_signal = BASE_DIR + 'output/ae_signal.mat'
save_mat = {}
save_mat["ae_signal"] = SIGNAL
sio.savemat(fname_signal, save_mat)
print("Signal Size")
print(SIGNAL.shape)

VAL_DATA = sio.loadmat(args.data_path + '/val.mat')
VAL_DATA = VAL_DATA['val']

def _variable_on_cpu(name, shape, initializer, trainable):
  with tf.device('/gpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable, dtype=dtype)
  return var

def next_batch(batch_size, count, permutation, offset, data):
  batch = []
  input_data = []
  output_data = []
  label = []
  #print(len(data))
  #print(len(data[0]))
  for i in range(int(batch_size / 2)):
    if count >= (int(len(data)/2)):
      permutation = np.random.permutation(int(len(data)/2))
      count = 0
      #print('Data Shuffled')
      #print(len(data))
    image = data[permutation[count]]
    #noise = np.random.normal(loc=0, scale=20, size=len(image))
    input_data.append(image)
    output_data.append(np.zeros(IMAGE_SIZE*IMAGE_SIZE))
    label.append([0])
    #noise = np.random.normal(loc=0, scale=20, size=len(image))
    #signal = SIGNAL
    signal = np.reshape(SIGNAL, -1, order='F')
    #print(offset)
    image2 = data[permutation[count]+offset]
    input_data.append(image2)
    output_data.append(signal)
    label.append([1])
    count = count + 1

  batch.append(np.asarray(input_data).astype(np.float32))
  batch.append(np.asarray(output_data).astype(np.float32))
  batch.append(np.asarray(label).astype(np.float32))
  return batch, count, permutation

def test_signal():
  signal = SIGNAL
  imgplot = plt.imshow(signal)
  plt.show()

def test_image():
  images = next_batch(2)
  nosigplt = plt.imshow(np.reshape(images[1][0], (IMAGE_SIZE, IMAGE_SIZE), order='F'))
  plt.show()
  sigplt = plt.imshow(np.reshape(images[1][1], (IMAGE_SIZE, IMAGE_SIZE), order='F'))
  plt.show()

def ske_latent_hotelling_observer(latent, images, labels, W1, signal):
  signal = np.reshape(signal, [-1, 1], order='F')
  cov = np.matrix(np.cov(latent, rowvar=False))
  temp1 = W1.T @ signal
  lho = np.linalg.solve(cov, (W1.T @ signal)).T @ W1.T
  thresh = images @ lho.T
  auc = roc_auc_score(labels, thresh)
  return auc

def train():
  np.random.seed(1) #for reproducability
  #tf.random.set_random_seed(1)
  sampled_image = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE])
  if TIED:
    reconstructed_image, W, z = inference(sampled_image)
  else:
    reconstructed_image, W1, W2, z = inference(sampled_image)
  signal_image = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE])

  global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int32)
  loss = tf.losses.mean_squared_error(signal_image, reconstructed_image)
  train_step = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss)#, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N, colocate_gradients_with_ops=True)

  increment_global_step_op = tf.assign_add(global_step, 1)

  saver = tf.train.Saver(tf.global_variables())

  print("Beginning Training:")
  config = tf.ConfigProto(allow_soft_placement=True)
  #config.gpu_options.allow_growth = True

  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    if CONTINUE:
      ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
      saver.restore(sess, ckpt.model_checkpoint_path)
      print("Model Restored")

    current_step = tf.train.global_step(sess, global_step)

    best_AUC = -np.inf
    
    train_auc = []
    train_permutation = np.random.permutation(int(len(TRAIN_DATA)/2))
    val_permutation = np.random.permutation(int(len(VAL_DATA)/2))
    train_count = np.inf
    val_count = np.inf

    while current_step < NUM_EPOCHS:
      start_train = time.time()
      batch_times = []
      train_loss = []
      latent_state_list = []
      for j in range(int(TRAIN_SIZE / BATCH_SIZE)): #num batches per epoch, should probably add a check or some logic here
        start = time.time()
        batch, train_count, train_permutation = next_batch(BATCH_SIZE, train_count, train_permutation, int(TRAIN_SIZE/2), TRAIN_DATA)
        if TIED:
          _, temp_loss, W_out, latent_train = sess.run([train_step, loss, W, z], feed_dict={sampled_image: batch[0], signal_image: batch[1]})
        else:
          _, temp_loss, W1_out, W2_out, latent_train = sess.run([train_step, loss, W1, W2, z], feed_dict={sampled_image: batch[0], signal_image: batch[1]})
        train_loss.append(temp_loss)
        end = time.time()
        batch_times.append(end-start)
        latent_state_list.append(latent_train)
        if TIED:
          t_auc = ske_latent_hotelling_observer(latent_train, batch[0], batch[2], W_out, SIGNAL)
        else:
          t_auc = ske_latent_hotelling_observer(latent_train, batch[0], batch[2], W1_out, SIGNAL)
        train_auc.append(t_auc)

      end_train = time.time()
      start_val = time.time()
      input_images = []
      output_images = []
      val_loss = []
      labels = []

      sess.run(increment_global_step_op)
      current_step = tf.train.global_step(sess, global_step)
      
      if current_step % 10 == 0:
        #np.random.seed(1999999999) # validation seed
        for j in range(int(VAL_SIZE / BATCH_SIZE)):
          start = time.time()
          #print(val_count)
          batch, val_count, val_permutation = next_batch(BATCH_SIZE, val_count, val_permutation, args.val_offset, VAL_DATA)
          if TIED:
            output, temp_loss, W_out = sess.run([z, loss, W], feed_dict={sampled_image: batch[0], signal_image: batch[1]})
          else:
            output, temp_loss, W1_out, W2_out = sess.run([z, loss, W1, W2], feed_dict={sampled_image: batch[0], signal_image: batch[1]})
          end = time.time()
          val_loss.append(temp_loss)
          batch_times.append(end-start)
          input_images.append(batch[0])
          output_images.append(output)
          labels.append(batch[2])

        end_val = time.time()
        labels = np.reshape(labels, [VAL_SIZE, 1])
        input_images = np.reshape(input_images, [VAL_SIZE, IMAGE_SIZE*IMAGE_SIZE])
        output_images = np.reshape(output_images, [VAL_SIZE, NUM_CHANNELS])
        latent_state_list = np.reshape(latent_state_list, [TRAIN_SIZE, NUM_CHANNELS])
        latent_state_concat = np.concatenate((latent_state_list, output_images))
        train_auc = np.mean(train_auc)
        if TIED:
          auc = ske_latent_hotelling_observer(latent_state_concat, input_images, labels, W_out, SIGNAL)
        else:
          auc = ske_latent_hotelling_observer(latent_state_concat, input_images, labels, W1_out, SIGNAL)
        format_str = ('%s: step %d, Train AUC = %.4f, Val AUC = %.4f, train loss = %.4f, validation loss = %.4f, training time = %.3f, validation time = %.3f (%.3f sec/batch)')
        train_loss = sum(train_loss) / float(len(train_loss))
        val_loss = sum(val_loss) / float(len(val_loss))
        sec_per_batch = sum(batch_times) / float(len(batch_times))
        print (format_str % (datetime.now(), current_step, train_auc, auc, train_loss, val_loss, end_train - start_train, end_val - start_val, sec_per_batch))
        sys.stdout.flush()

        if args.validation == 1:
          target_val = auc
        else:
          target_val = train_auc

        if best_AUC < target_val:
          best_AUC = target_val
          saver.save(sess, CKPT_NAME, global_step=global_step)
          if TIED:
            fname_W = BASE_DIR + 'output/W1.dat'
            fid_w = open(fname_W, 'w')
            W_out.tofile(fid_w)
          else:
            fname_W1 = BASE_DIR + 'output/W1.dat'
            fid_w1 = open(fname_W1, 'w')
            W1_out.tofile(fid_w1)
            fname_W2 = BASE_DIR + 'output/W2.dat'
            fid_w2 = open(fname_W2, 'w')
            W2_out.tofile(fid_w2)
          fname_recon = BASE_DIR + 'output/channels.dat'
          fid_recon = open(fname_recon, 'w')
          np.concatenate(output_images).tofile(fid_recon)
          fname_labels = BASE_DIR + 'output/labels.dat'
          fid_labels = open(fname_labels, 'w')
          np.concatenate(labels).tofile(fid_labels)

        train_auc = []  

      end_epoch = time.time()

def inference(sampled_image):
    if TIED:
      W =_variable_on_cpu(shape=[IMAGE_SIZE * IMAGE_SIZE, NUM_CHANNELS], initializer=tf.contrib.layers.xavier_initializer(), name=('W'), trainable=True)
      z = tf.matmul(sampled_image, W)
      current_reconstruction = tf.matmul(z, tf.transpose(W))

      return current_reconstruction, W, z

    else:
      W1 =_variable_on_cpu(shape=[IMAGE_SIZE * IMAGE_SIZE, NUM_CHANNELS], initializer=tf.contrib.layers.xavier_initializer(), name=('W1'), trainable=True)
      W2 =_variable_on_cpu(shape=[IMAGE_SIZE * IMAGE_SIZE, NUM_CHANNELS], initializer=tf.contrib.layers.xavier_initializer(), name=('W2'), trainable=True)
      z = tf.matmul(sampled_image, W1)
      current_reconstruction = tf.matmul(z, tf.transpose(W2))

      return current_reconstruction, W1, W2, z


def main():
  train()

if __name__ == "__main__":
  main()
