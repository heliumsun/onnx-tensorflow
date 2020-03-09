import tensorflow as tf

from onnx_tf.common import get_data_format
from onnx_tf.common import get_perm_from_formats
from onnx_tf.common import supports_device
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("BatchNormalization")
class BatchNormalization(BackendHandler):
  @classmethod
  def get_attrs_processor_param(cls):
    return {
        "default": {
            "epsilon": 1e-5
        },
        # "rename": {
        #     "epsilon": "variance_epsilon"
        # }
    }

  @classmethod
  def _common(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]
    x_shape = x.get_shape().as_list()
    x_rank = len(x_shape)

    params_shape_broadcast = list([1, x_shape[1]] + [1 for _ in range(2, x_rank)])

    total_num_dim = len(x.get_shape())
    scale = tensor_dict[node.inputs[1]]
    bias = tensor_dict[node.inputs[2]]
    running_mean = tensor_dict[node.inputs[3]]
    running_variance = tensor_dict[node.inputs[4]]

    support_cuda = supports_device("CUDA")
    storage_format, compute_format = get_data_format(x_rank)

    # from version 7, force to use test mode
    if cls.SINCE_VERSION >= 7 or node.attrs.get("is_test", 0):
      inputs = [x, scale, bias, running_mean, running_variance]
    else:
      spatial = node.attrs.get("spatial", 1) == 1
      momentum = node.attrs.get("momentum", 0.9)
      axis = [0] if spatial else [0] + list(range(2, total_num_dim))
      mean, variance = tf.nn.moments(x, axis)
      running_mean = running_mean * momentum + mean * (1 - momentum)
      running_variance = running_variance * momentum + variance * (1 - momentum)
      # TODO: need to conform to the documentation here
      inputs = [x, scale, bias, running_mean, running_variance]

    if not support_cuda:
      x = tf.transpose(x, perm=get_perm_from_formats(storage_format, compute_format))
    output, _, _ = tf.nn.fused_batch_norm(x,
                                          scale,
                                          bias,
                                          running_mean,
                                          running_variance,
                                          node.attrs.get('epsilon'),
                                          data_format=compute_format,
                                          is_training=False)
    if not support_cuda:
      output = tf.transpose(output, perm=get_perm_from_formats(compute_format, storage_format))
    return [output]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_9(cls, node, **kwargs):
    return cls._common(node, **kwargs)
