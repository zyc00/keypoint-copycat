"""Hyperparameters of the structured video prediction models."""

import os


class ConfigDict(dict):
    """A dictionary whose keys can be accessed as attributes."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, key, value):
        self[key] = value

    def get(self, key, default=None):
        """Allows to specify defaults when accessing the config."""
        if key not in self:
            return default
        return self[key]


def get_config(FLAGS):
    """Default values for all hyperparameters."""

    cfg = ConfigDict()

    # Directories:
    cfg.exp_name = FLAGS.exp_name
    cfg.base_dir = os.path.join(FLAGS.base_dir, cfg.exp_name)

    cfg.data_dir = FLAGS.data_dir
    cfg.train_dir = FLAGS.train_dir
    cfg.test_dir = FLAGS.test_dir

    cfg.checkpoint_dir = FLAGS.checkpoint_dir
    cfg.logs_dir = FLAGS.logs_dir
    cfg.pretrained_path = FLAGS.pretrained_path

    # Architecture:
    cfg.layers_per_scale = 2
    cfg.conv_layer_kwargs = _conv_layer_kwargs()
    cfg.dense_layer_kwargs = _dense_layer_kwargs()

    # Optimization:
    cfg.batch_size = FLAGS.batch_size
    cfg.steps_per_epoch = FLAGS.steps_per_epoch
    cfg.num_epochs = FLAGS.num_epochs
    cfg.learning_rate = FLAGS.learning_rate
    cfg.clipnorm = FLAGS.clipnorm

    # Image sequence parameters:
    cfg.observed_steps = FLAGS.timesteps
    cfg.predicted_steps = FLAGS.timesteps

    # Keypoint encoding settings:
    cfg.num_keypoints = FLAGS.num_keypoints
    cfg.heatmap_width = 16
    cfg.hearmap_regularization = FLAGS.heatmap_reg
    cfg.keypoint_width = 1.5
    cfg.num_encoder_filters = 32
    cfg.separation_loss_scale = FLAGS.temp_reg
    cfg.separation_loss_sigma = FLAGS.temp_width

    return cfg


def _conv_layer_kwargs():
    """Returns a configDict with default conv layer hyperparameters."""

    cfg = ConfigDict()

    cfg.kernel_size = 1
    cfg.padding = 1

    return cfg


def _dense_layer_kwargs():
    """Returns a configDict with default dense layer hyperparameters."""

    cfg = ConfigDict()

    return cfg
