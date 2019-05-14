"""Training script."""
import time
import os
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
import model
import dataset

FLAGS = flags.FLAGS

def get_optimizer():
  lr = tf.train.piecewise_constant(tf.get_global_step(), FLAGS.lr_boundaries, FLAGS.lr_values)
  opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=FLAGS.momentum, epsilon=FLAGS.adam_epsilon)
  return opt

def model_fn(features, labels, mode):
  training = mode == tf.estimator.ModeKeys.TRAIN
  modelfun = model.sph_embedding_residual
  postproc_fun = tf.nn.sigmoid
  prediction, target, inp = modelfun(features, training=training, postproc_fun=postproc_fun)
  outpred = prediction[-1] if isinstance(prediction, list) else prediction
  pred0 = prediction[0] if isinstance(prediction, list) else tf.zeros_like(prediction)
  loss, rel_err = model.embedding_loss(prediction, target, inp)
  eval_ops = {'mean_loss': tf.metrics.mean(loss), 'relative_err': tf.metrics.mean(rel_err)}
  # mode==Train
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = get_optimizer()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  # mode==TEST
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_ops)

def main(unused_argv):
  config = tf.estimator.RunConfig(log_step_count_steps=1, save_checkpoints_secs=5*60, keep_checkpoint_max=2)
  train(config)

def train(config):
  """Training loop."""
  classifier = tf.estimator.Estimator(model_fn=model_fn,
                                      model_dir=FLAGS.model_dir,
                                      config=config,
                                      warm_start_from=None)

  if not FLAGS.eval_only:
    logging.info('Start training...')
    # do not count epochs when # of steps is given
    classifier.train(
        input_fn=lambda: dataset.make_next_batch_op(FLAGS.batch_size, None, FLAGS.dataset, 'train'),
        max_steps=FLAGS.max_steps)
    logging.info('Finished training.')

  #Eval mode
  last_mtime, i = -1, 0
  logging.info('Starting evaluation.')
  while i < FLAGS.eval_n_times:
    ckpt = classifier.latest_checkpoint()
    run_eval = False
    if ckpt is None:
      logging.info('No checkpoints found.')
      run_eval = False
    elif (ckpt is not None and os.path.isfile(ckpt + '.meta')):
      mtime = os.stat(ckpt + '.meta').st_mtime
      if mtime != last_mtime:
        logging.info('New checkpoint found! evaluating...')
        last_mtime = mtime
        i += 1
        run_eval = True
    if run_eval:
      out = classifier.evaluate(
          input_fn=lambda: dataset.make_next_batch_op(FLAGS.batch_size, 1, get_test_datafolder(), 'test'),
          steps=FLAGS.max_steps)
      logging.info(out)
    time.sleep(FLAGS.eval_n_times_wait)

if __name__ == '__main__':
  tf.app.run()
