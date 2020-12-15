# zlx_NAM_github
 
这个是深度可加模型的个人复现实验，论文来自《Neural Additive Models:Interpretable Machine Learning with Neural Nets》，但是没有用它的代码，代码全部由zlx编写，其中加入了生成可以排列组合的内容。  
 😊
 
 # 代码环境
 
 tensorflow_gpu >= 2.0  
 python 2.7  
 keras >= 2.3  
 
 # 模型架构
 
 基于是线性可加模型,然后每个feature是放入一个神经网络中，之后将两个排列组合的features放入神经网络中，最后一个线性可加层。  
 
 
 # 实验结果
 
 用模拟实验  
 数据集是Y = I((X1>0.5) and (X2*X3>0))。  
 结果表明发现了X1的是直接影响结果的关系。单变量X2，X3是没有这种效果的。  
