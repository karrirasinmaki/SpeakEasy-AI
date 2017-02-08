"""Train the SpeakEasy model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import os
path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(path)
import multiprocessing
from threading import Thread

import numpy as np
import time
import math
from six.moves import xrange

import tensorflow as tf
# import tensorflow.python.platform
from tensorflow.python.platform import gfile

from model import model_utils
from data import data_utils
from slack import slack
from runtime_variables import params, buckets

def read_data(data_path, max_size=None):
  """Read data from data files to compile data set.
  Args:
    data_path: Path to the file with token-ids
    max_size: Maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).
  Returns:
    If using a model with buckets:
      data_set: A list of length len(_buckets); data_set[n] contains a list of
        (prompt, response) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(prompt) < _buckets[n][0] and
        len(response) < _buckets[n][1]; prompt and response are lists of token-ids.
    Else:
      data_set: A list of (prompt, response) pairs read from the provided data files;
        prompt and response are lists of token-ids.
  """
  data_set = [[] for _ in buckets] if params.buckets else []
  print("Opening data set %s" % (data_path))
  sys.stdout.flush()
  with gfile.GFile(data_path, mode="r") as data_file:
    # Response line is line after prompt line
    prompt, response, _ = data_file.readline(), data_file.readline(), data_file.readline()
    counter = 0
    while prompt and response and (not max_size or counter < max_size):
      # Skip empty lines
      if (len(prompt.strip().split()) > 1) and (len(response.strip().split()) > 1):
        counter += 1
        if counter % 10000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        prompt_ids = [int(x) for x in prompt.split()]
        response_ids = [int(x) for x in response.split()]
        response_ids.append(data_utils.EOS_ID)
        if params.buckets:
          for bucket_id, (prompt_size, response_size) in enumerate(buckets):
            if len(prompt_ids) < prompt_size and len(response_ids) < response_size:
              data_set[bucket_id].append([prompt_ids, response_ids])
              break
        else:
          if len(prompt_ids) <= params.max_sentence_length and len(response_ids) <= params.max_sentence_length:
            data_set.append([prompt_ids, response_ids])
      prompt, response = data_file.readline(), data_file.readline()
  return data_set

def train():
  slack.connection.notify(
    text='Training SpeakEasy!',
  )
  # Prepare reddit data.
  print("Preparing data in %s" % params.data_dir)
  sys.stdout.flush()
  data_train, data_dev, _ = data_utils.prepare_data(params.data_dir, params.vocab_size)

  # gpu_options = tf.GPUOptions(allocator_type = 'BFC', per_process_gpu_memory_fraction = 0.9)
  # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
  config = tf.ConfigProto(device_count={"CPU": multiprocessing.cpu_count()})
  #config.gpu_options.allow_growth = True
  #config.gpu_options.per_process_gpu_memory_fraction = 0.3
  # CPU multithreading
  config.inter_op_parallelism_threads=2 # size of Eigen threadpool
  config.intra_op_parallelism_threads=1 # how many ops can be launched in parallel
  with tf.Session(config=config) as sess:
    # Create model.
    print("Creating %s model with %d layers of %d units." % (params.model_type, params.num_layers, params.size))
    if params.buckets: print("Using bucketed model.")
    sys.stdout.flush()
    model = model_utils.create_model(sess, False)
    print('Model ready')
    sys.stdout.flush()

    # Set up event logging. NOTE: added this, this is not finished
    merged_summaries = tf.summary.merge_all()
    # writer = tf.summary.FileWriter(params.train_dir, sess.graph)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)." % params.max_train_data_size)
    sys.stdout.flush()
    dev_set = read_data(data_dev)
    train_set = read_data(data_train, params.max_train_data_size)

    if params.buckets:
      train_bucket_sizes = [len(train_set[b]) for b in xrange(len(buckets))]
      train_total_size = float(sum(train_bucket_sizes))
      # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
      # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
      # the size if i-th training bucket, as used later.
      train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    keep_training = True
    # while True:
    while keep_training:
      if (current_step % 100 == 0):
        print("Training, step %d and counting..." % (current_step,))
        sys.stdout.flush()
      # Get a batch and make a step.
      start_time = time.time()

      if params.buckets:
        # Choose a bucket according to data distribution. We pick a random number
        # in [0, 1] and use the corresponding interval in train_buckets_scale.
        random_number_01 = np.random.random_sample()
        bucket_id = min([i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
        _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
      else:
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id=None)
        _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, 0, False)
      step_time += (time.time() - start_time) / params.steps_per_checkpoint
      loss += step_loss / params.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % params.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        log_line = ("global step %d learning rate %.4f step-time %.2f perplexity %.2f" %
            (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))
        print(log_line)
        sys.stdout.flush()
        slack.connection.notify(
          text=log_line,
        )
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(params.train_dir, "speakEasy_vocab%d_size%d.ckpt" % (params.vocab_size, params.size))
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0

        print("saved")
        sys.stdout.flush()

        if params.buckets:
          thread = Thread(target = log_buckets, args = (sess, model, dev_set))
          thread.start()
          if perplexity < 100:
            thread.join()
            break
        else:
          if perplexity < 100:
            break

def log_buckets(sess, model, dev_set):
  for bucket_id in xrange(len(buckets)):
    if len(dev_set[bucket_id]) == 0:
      print("  eval: empty bucket %d" % (bucket_id))
      continue
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
	dev_set, bucket_id)
    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
				 target_weights, bucket_id, True)
    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
	"inf")
    print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
  sys.stdout.flush()

