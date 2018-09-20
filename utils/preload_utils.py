#查看checkpoint 节点信息： 代码如下
from tensorflow.python import pywrap_tensorflow
import os

checkpoint_path = os.path.join( "checkpoint-00454721")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
        print("tensor_name: ", key)
        

#初始化相关变量
uninitialized_vars = []
for var in tf.global_variables():
    try:
        sess.run(var)
    except tf.errors.FailedPreconditionError:
        uninitialized_vars.append(var)
initialize_op = tf.variables_initializer(uninitialized_vars)
sess.run(initialize_op)
