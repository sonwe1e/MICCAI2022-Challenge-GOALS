# MICCAI2022-Challenge-GOALS
BaiDu AI Studio 环扫OCT图像的层分割；

# 文件作用
pl_train.py 利用pytorch-lightning对数据进行分割训练  

model.py 对UNet进行修改  

utils.py 关于分割的损失函数 包括GMS-Loss Dice-Loss Focal-Loss  

pl_predict.py 利用pytorch-lightning进行分割预测  

