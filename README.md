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
1:1 verification <br/>
【1】FaceNet:A Unified Embedding for Face Recognition and Clustering<br/>
【2】Additive Margin Softmax for Face Verification<br/>
【3】ArcFace-Additive Angular Margin Loss for Deep Face Recognition<br/>
【4】MobileFaceNets Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices <br/>

1:n identification <br/>
【1】A Fast and Accurate System for Face Detection, Identification, and Verification<br/>
【2】MegaFace A Million Faces for Recognition at Scale<br/>
【3】Unconstrained Face Recognition Identifying a Person of Interest from a Media Collection<br/>
【4】Web-Scale Training for Face Identification<br/>

## 活体检测：<br/>
【1】Face Flashing: a Secure Liveness Detection Protocol based on Light Reflections<br/>

## 人脸去网纹：<br/>
【1】Multi-task ConvNet for Blind Face Inpainting with Application to Face Verification<br/>
【2】DeMeshNet: Blind Face Inpainting for Deep MeshFace Verification<br/>

## 防翻拍：<br/>
【1】Learning Deep Models for Face Anti-Spoofing Binary or Auxiliary Supervision<br/>
【2】CCoLBP: Chromatic Co-Occurrence of Local Binary Pattern for Face Presentation Attack Detection<br/>

## 遮挡检测：<br/>
【1】An Occluded Stacked Hourglass Approach to Facial Landmark Localization and Occlusion Estimation<br/>
【2】Occlusion Coherence: Detecting and Localizing Occluded Faces<br/>

# 数据集<br/>
人脸识别：欧美：vggFace，lfw，WebFace，MSIM   亚洲：msra亚洲人脸数据集<br/>
人脸检测对齐：FDDB，widerface，ALFW<br/>

开发环境：
tesla p40，4gpu，单卡显存22g<br/>
python3 + tensorflow_gpu 1.11.0 + cuda9.2 + cudnn7200<br/>
python3 + tensorflow_gpu 1.11.0 + cuda9.0 + cudnn7300<br/>


# loss函数实测：（仅作参考）<br/>
softmax_loss：目前证明可以很好地拟合训练集，loss函数可以降到0.0x级别，但是在lfw测试集上面性能和主流loss差距较大<br/>
center_loss：强调最小化类内距离，单独不太好训练，和softmax结合着用，实测效果提升不是很明显<br/>
triplet_loss：谷歌论文提出的损失函数，强调类间类内距离，实测不太好训练，所以打算在后期尝试用来finetune，也有一种想法是将triplet_loss的欧式距离改成余弦距离测试一下效果<br/>
arc_loss：理论上来说应该是精度最高的，基于L_Resnet_E_IR网络进行训练，实测网络训练时占用显存比较多，要复现论文的batch_size和迭代步骤数需要一定的硬件支撑和时间(论文中作者用了4个tesla-p40,实测确实要4个才跑的起512的batch_size，一个batch迭代1.5-2秒)<br/>
additive_margin_loss:一种基于angular的损失函数，采用20层或者36层的resnet网络，收敛速度比较快，lfw数据集上准确率明显优于softmax,但是loss相对于softmax不太容易降下去，对于训练集的拟合程度不是很好，目前打算采用GPU并行增大batch_size数量进行迭代，同时调整学习率，观察loss下降趋势进一步优化<br/>

# 网络架构实测：<br/>
resnet(20,36):实测收敛速度快，用相同的loss函数进行迭代对比，lfw准确率介于inception和LR之间，后期可以试试更深的层数迭代<br/>
inception_resnet_v1：按理说效果应该是要比resnet好，但是实测性能也没有太明显的提升<br/>
L_Resnet_E_IR：同样的loss函数下，最终迭代精确度最高的网络，迭代速度较于resnet和inception-resnet要慢一倍,采用了L_Resnet_50网络:bacth_size=32时GPU显存占用8g，tensorflow模型迭代速度约30-40 sample/s，识别一张图片300ms左右<br/>
Mobilefacenet:单gpu batch_size=256,GPU显存占用16g，模型迭代速度260sample/s，识别一张图片60ms左右<br/>

# 一些调优经验：<br/>
关于神经网络的调优，一般可以考虑以下一些方面：<br/>
embedding_size:在mobilefacenet中，128调整为512，更加稳定<br/>
adam参数调整，图像分类时部分超参数需要手动设置<br/>
梯度爆炸，learning rate调小<br/>
结合tensorboard观察loss下降趋势，相应地调节epoch迭代步数<br/>
weight_decay罚项减小，增加模型复杂度<br/>
图像归一化处理，训练和测试集要统一<br/>
batch_size不是越大越好，要和模型相适应<br/>
模型对于训练集的拟合度和类别的数量有直接关系，复杂的网络对于类别多的数据集拟合能力更好<br/>


# 相关学习资料
李飞飞计算机视觉cs231n课程<br/>
fast.ai课程<br/>
补充知识点：<br/>
关于CNN对图像特征的位移、尺度、形变不变性的理解<br/>
卷积神经网络显存,内存,使用量估计<br/>
深度学习优化器解析，模型参数调优，正则化机制<br/>
人脸识别算法评价指标——TAR，FAR，FRR，ERR<br/>


# 命令行记录：<br/>
1.pip wheel --wheel-dir=下载目录 tensorlayer==x.x.x<br/>
2.https://pypi.tuna.tsinghua.edu.cn/simple/<br/>
3.awk 'BEGIN{srand();} {printf "%s %s\n", rand(), $0}' t | sort -k1n | awk '{gsub($1FS,""); print $0}' | tee shuffle.txt<br/>
4.cat /usr/local/cuda/version.txt<br/>
5.cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2<br/>

# 更新ing
9.16<br/>
1.增加bin文件生成脚本<br/>
2.增加tf_record文件生成脚本<br/>
3.增加pair.txt生成脚本<br/>
4.增加视频提取人脸帧，并且进行方向调整脚本（左旋90，右旋90，倒立均无法检测出人脸）<br/>
5.增加模型预加载测试脚本<br/>

9.20<br/>
增加网纹生成脚本<br/>

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
sudo docker rm  <br/>
sudo docker rmi <br/>
ctrl+p, ctrl+q <br/>

10.20<br/>
1.添加L_Resnet_E_IR多GPU训练脚本,可以进行单GPU或者CPU finetune<br/>
2.添加MoblienFacenet多GPU训练脚本,可以进行单GPU或者CPU finetune<br/>


