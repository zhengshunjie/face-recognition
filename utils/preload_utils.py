#查看checkpoint 节点信息： 代码如下
from tensorflow.python import pywrap_tensorflow
import os

#打印变量
tf.contrib.framework.get_variables_to_restore()
tf.all_variables()
tf.trainable_varialbes()

#得到该网络中，所有可以加载的参数
variables = tf.contrib.framework.get_variables_to_restore()
#删除output层中的参数
variables_to_resotre = [v for v in varialbes if v.name.split('/')[0]!='output']
#构建这部分参数的saver
saver = tf.train.Saver(variables_to_restore)
saver.restore(sess,'model.ckpt')



#这种办法无法打印出img_placeholder这种tensor
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
