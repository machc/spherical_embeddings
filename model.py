## Code snippet demonstrating Spherical AutoEncoder Architecture.
import tensorflow as tf
import numpy as np
from absl import flags
from absl import logging
from tensorflow_models.official.resnet import resnet_model
import tensorflow_hub as hub

FLAGS = flags.FLAGS
na = tf.newaxis
slim = tf.contrib.slim

def residual_encoder(curr, training, n_latent=1024, initial_channels=64, max_channels=256, layers_per_block=2):
  dim = curr.shape[1].value
  # we want last layer to be 4x4, layer 1 already has subsampling.
  nblocks = int(np.log2(dim))
  assert np.allclose(nblocks, np.log2(dim))
  for i in range(nblocks):
    if i == 0:
      curr = resnet_model.conv2d_fixed_padding(inputs=curr, filters=initial_channels, kernel_size=7, strides=2, data_format='channels_last')
    else:
      strides = 1 if i == 1 else 2
      filters = min(initial_channels * 2**i, max_channels)
      curr = resnet_model.block_layer(curr, filters, bottleneck=False, block_fn=resnet_model._building_block_v2, blocks=layers_per_block, strides=strides, training=training, name='res_block{:02d}'.format(i), data_format='channels_last')
  # linear layer to latent space
  curr = tf.layers.conv2d(curr, filters=n_latent, kernel_size=2, activation=None, padding='valid')
  return curr

def residual_decoder(curr, training, target, initial_channels=64, max_channels=256, layers_per_block=2):
  dim = target.shape[-2].value
  nch = target.shape[-1].value
  normalizer_fn = slim.batch_norm
  normalizer_fn_args = {
        'is_training': training,
        'zero_debias_moving_mean': True,
  }
  nblocks = int(np.log2(dim)-2)
  assert np.allclose(nblocks, np.log2(dim)-2)

  with slim.arg_scope([normalizer_fn], **normalizer_fn_args):
    with slim.arg_scope([slim.conv2d_transpose], normalizer_fn=normalizer_fn, activation_fn=tf.nn.relu, stride=2, kernel_size=3):
      for i in range(nblocks):
        n_filters = min(initial_channels * 2**(nblocks - i  - 1), max_channels)
        if i == 0:
          curr = slim.conv2d_transpose(curr, n_filters, kernel_size=4, stride=1, padding='VALID')
        else:
          curr = resnet_model.block_layer(curr, n_filters, bottleneck=False, block_fn=resnet_model._building_block_v2, blocks=layers_per_block, strides=1, training=training, name='res_block{:02d}'.format(i), data_format='channels_last')
          curr = slim.conv2d_transpose(curr, n_filters)
        print(curr)
  # last 7x7 linear layer back to original res.
  curr = slim.conv2d_transpose(curr, num_outputs=nch, kernel_size=7, stride=2, activation_fn=tf.identity)
  return curr

def sph_embedding_encdec(features, training=False, fun=None):
  """Maps 2D view to spherical embedding via encoder-decoder architecture"""
  sph, im, R = features
  # view2fmaps; targets come from running spherical CNNs on mesh2sph rep.
  target = hub.Module(FLAGS.tgt_embed_dir)(sph[..., na])
  if FLAGS.augment and training:
    im = util.augment_batch(im[..., na])[..., 0]
  if len(im.shape) == 3:
    im = im[..., na]
  if len(target.shape) == 3:
    target = target[..., na]
  pred = fun(im, target, training)
  return pred, target, im

def sph_embedding_residual(features, training, postproc_fun=tf.identity):
  """Maps 2D view to spherical embedding using residual encoder-decoder"""
  def fun(im, target, training):
    padmode = 'REFLECT'
    def circpad(*a, **kwa):
      return util.padding(*a, **kwa, mode=padmode, circular=True)
    def pad(*a, **kwa):
      return util.padding(*a, **kwa, mode=padmode, circular=False)
    encpad = pad
    with util.monkeypatched(resnet_model, 'fixed_padding', encpad):
      latent = residual_encoder(im, training, layers_per_block=FLAGS.layers_per_block, max_channels=FLAGS.max_channels)
    decpad = circpad   # output is spherical - use circular padding.
    with util.monkeypatched(resnet_model, 'fixed_padding', decpad):
      pred = residual_decoder(latent, training, target, layers_per_block=FLAGS.layers_per_block, max_channels=FLAGS.max_channels)
    pred = postproc_fun(pred)
    return pred
  return sph_embedding_encdec(features, training=training, fun=fun)

def embedding_loss(prediction, target, inp):
  def huber(*args, **kwargs):
    return tf.losses.huber_loss(*args, **kwargs, delta=FLAGS.huber_delta)
  # loss is on the sph; take area_weights into account
  target = sph_layers.area_weights(target)
  losses = {'l1': tf.losses.absolute_difference, 'l2': tf.losses.mean_squared_error, 'huber': huber}
  loss = 0
  for p in prediction:
    p = sph_layers.area_weights(p)
    loss += losses[huber](target, p, loss_collection=None)
  loss /= len(prediction)
  # measure relative error.
  rel_err = tf.reduce_mean(tf.abs(prediction[-1] - target) / (tf.abs(prediction[-1]) + tf.abs(target) + 1e-8))
  tf.summary.scalar('relative_err', rel_err)
  return loss, rel_err
