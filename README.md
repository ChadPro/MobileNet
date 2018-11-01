# Mobile Net: Efficient Convolutional Neural Networks for Mobile Vision Applications
Mobile Net is a small net, use for mobile.And we change the net construct for create SSD net.We retain the seperable net.Use input_size 300.And make the net more useful for Space Message.
# Dataset
We use image net data,for 1000 class.  
And we have two types dataset, one is 300 size image dataset, and the other is 224.The names are imageNetTrain_300.tfrecord and imageNetTrain_224.tfrecord.  
Download in.
# Net Construct
1. Primary MobileNet-224 original  
![image](http://owv7la1di.bkt.clouddn.com/blog/180129/eE4BIFGIIe.png=200*250)

2. MobileNet-300 original

3. MobileNet-224 dense

4. MobileNet-300 dense

# Use this Repository
### Run

### Training
##### 1. Train param
We use script **mobile_net_train.py** to train model.  
```python
python mobile_net_train.py
--learning_rate_base=0.01  
--train_data_path='../../Datasets/ImageNet_224/imageNetTrain.tfrecord'
--val_data_path='../../Datasets/ImageNet_224/imageNetVal.tfrecord' 
--dataset='imagenet_224'
--batch_size=32
--net_chose='mobile_net_224_original'
--image_size=224
```
If you want to chose another net, you can change the 'net_chose', 'image_size',for example:  
```python
python mobile_net_train.py
--learning_rate_base=0.01  
--train_data_path='../../Datasets/ImageNet_224/imageNetTrain.tfrecord'
--val_data_path='../../Datasets/ImageNet_224/imageNetVal.tfrecord' 
--dataset='imagenet_224'
--batch_size=32
--net_chose='mobile_net_224_v1'
--image_size=224
```
or
```python
python mobile_net_train.py
--learning_rate_base=0.01  
--train_data_path='../../Datasets/ImageNet_224/imageNetTrain.tfrecord'
--val_data_path='../../Datasets/ImageNet_224/imageNetVal.tfrecord' 
--dataset='imagenet_300'
--batch_size=32
--net_chose='mobile_net_300_v1'
--image_size=300
```

##### 2. Train Details


### Fine tune
```python
python mobile_net_train.py
--learning_rate_base=0.01  
--train_data_path='../../Datasets/ImageNet_224/imageNetTrain.tfrecord'
--val_data_path='../../Datasets/ImageNet_224/imageNetVal.tfrecord' 
--dataset='imagenet_224'
--batch_size=32
--net_chose='mobile_net_224_original'
--image_size=224
--restore_model_dir='./model/model.ckpt'
```