# face-recognition
人脸识别项目ing<br/>

# 项目介绍：<br/>
目前在做人脸图像相关项目，开个repository记录一下开发心得<br/>

# 参考论文:<br/>
## 人脸检测，关键点定位，追踪：<br/>
【1】Deep Convolutional Network Cascade for Facial Point Detection<br/>
【2】A Convolutional Neural Network Cascade for Face Detection<br/>
【3】Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks<br/>
【4】Robust facial landmark tracking via cascade regression<br/>
【5】Convolutional Experts Network for Facial Landmark Detection<br/>
【6】A Functional Regression approach to Facial Landmark Tracking<br/>

## 人脸识别：<br/>
【1】FaceNet:A Unified Embedding for Face Recognition and Clustering<br/>
【2】Additive Margin Softmax for Face Verification<br/>
【3】ArcFace-Additive Angular Margin Loss for Deep Face Recognition<br/>
## 活体检测：<br/>
【1】Learning Deep Models for Face Anti-Spoofing Binary or Auxiliary Supervision<br/>
## 人脸去网纹：<br/>
【1】Multi-task ConvNet for Blind Face Inpainting with Application to Face Verification<br/>
【2】DeMeshNet: Blind Face Inpainting for Deep MeshFace Verification<br/>


# 数据集<br/>
人脸识别：欧美：vggFace，lfw，WebFace，MSIM   亚洲：msra亚洲人脸数据集<br/>
人脸检测对齐：FDDB，widerface，ALFW<br/>

# loss函数实测：（仅作参考）<br/>
softmax_loss：目前证明可以很好地拟合训练集，loss函数可以降到0.0x级别，但是在lfw测试集上面性能和主流loss差距较大<br/>
center_loss：强调最小化类内距离，单独不太好训练，和softmax结合着用，实测效果提升不是很明显<br/>
triplet_loss：谷歌论文提出的损失函数，强调类间类内距离，实测不太好训练，所以打算在后期尝试用来finetune，也有一种想法是将triplet_loss的欧式距离改成余弦距离测试一下效果<br/>
arc_loss：理论上来说应该是精度最高的，基于L_Resnet_E_IR网络进行训练，实测网络训练时占用显存比较多，要复现论文的batch_size和迭代步骤数需要一定的硬件支撑和时间(论文中作者用了4个tesla-p40,实测确实要4个才跑的起512的batch_size，一个batch迭代1.5-2秒)<br/>
additive_margin_loss:一种基于angular的损失函数，采用20层或者36层的resnet网络，收敛速度比较快，lfw数据集上准确率明显优于softmax,但是loss相对于softmax不太容易降下去，对于训练集的拟合程度不是很好，目前打算采用GPU并行增大batch_size数量进行迭代，同时调整学习率，观察loss下降趋势进一步优化<br/>

# 网络架构实测：<br/>
resnet(20,36):实测收敛速度快，用相同的loss函数进行迭代对比，lfw准确率介于inception和LR之间，后期可以试试更深的层数迭代<br/>
inception_resnet_v1：按理说效果应该是要比resnet好，但是实测性能也没有太明显的提升<br/>
L_Resnet_E_IR：同样的loss函数下，最终迭代精确度最高的网络，迭代速度较于resnet和inception要慢一倍(L_Resnet_50网络 :32bacth下单GPU显存8g，teslap40共22g显存，单gpu可跑64batch)<br/>

# 相关学习资料
李飞飞计算机视觉cs231n课程<br/>
fast.ai课程<br/>
补充知识点：<br/>
关于CNN对图像特征的位移、尺度、形变不变性的理解<br/>
卷积神经网络显存,内存,使用量估计<br/>
深度学习优化器解析，模型参数调优，正则化机制<br/>
人脸识别算法评价指标——TAR，FAR，FRR，ERR<br/>


# 工程相关：
opencv,dlib
pip wheel --wheel-dir=下载目录 tensorlayer==x.x.x

# 更新ing
9.16<br/>
1.增加bin文件生成脚本<br/>
2.增加tf_record文件生成脚本<br/>
3.增加pair.txt生成脚本<br/>
4.增加视频提取人脸帧，并且进行方向调整脚本（左旋90，右旋90，倒立均无法检测出人脸）<br/>
5.增加模型预加载测试脚本<br/>

9.20<br/>
增加网纹生成脚本<br/>

9.25<br/>
增加预加载多gpu训练模型脚本<br/>

10.1<br/>
记录一些docker常用命令：<br/>
sudo docker import docker_name.tar docker_name:1.0<br/>
sudo docker images<br/>
sudo docker run -idt -v /data:/data dce2d4ae5307 /bin/bash<br/>
sudo docker run -it docker_name:1.0 /bin/bash<br/>
sudo docker ps<br/>
sudo docker attach p_id<br/>
docker commit -m="has update" -a="runoob" 47090a533b51 user_name/docker_name:3.0<br/>
sudo docker stop<br/>
exit<br/>
