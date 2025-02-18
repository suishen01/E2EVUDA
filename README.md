# [NeurIPS 2024] ["End-to-End Video Semantic Segmentation in Adverse Weather using Fusion Blocks and Temporal-Spatial Teacher-Student Learning"]([https://openaccess.thecvf.com/content/ACCV2022/papers/Yang_Object_Detection_in_Foggy_Scenes_by_Embedding_Depth_and_Reconstruction_ACCV_2022_paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/fed6d142d12b2f8c031615cc8fd50893-Paper-Conference.pdf))

### Abstract
Adverse weather conditions can significantly degrade video frames, leading to erroneous predictions by current video semantic segmentation methods. Furthermore, these methods rely on accurate optical flows, which become unreliable under adverse weather. To address this issue, we introduce the novelty of our approach: the first end-to-end, optical-flow-free, domain-adaptive video semantic segmentation method. This is accomplished by enforcing the model to actively exploit the temporal information from adjacent frames through a fusion block and temporal-spatial teachers. The key idea of our fusion block is to offer the model a way to merge
information from consecutive frames by matching and merging relevant pixels from those frames. The basic idea of our temporal-spatial teachers involves two teachers: one dedicated to exploring temporal information from adjacent frames,
the other harnesses spatial information from the current frame and assists the temporal teacher. Finally, we apply temporal weather degradation augmentation to consecutive frames to more accurately represent adverse weather degradations.
Our method achieves a performance of 25.4% and 33.0% mIoU on the adaptation from VIPER and Synthia to MVSS, respectively, representing an improvement of 4.3% and 5.8% mIoU over the existing state-of-the-art method.

### Preparation
This paper follows the environment of [TPS](https://github.com/xing0047/TPS/tree/main)


### Train and Test
 ```Shell
 # train
 CUDA_VISIBLE_DEVICES=GPU_ID python train.py --cfg configs/{your config}.yml
 
 # test
 CUDA_VISIBLE_DEVICES=GPU_ID python test.py --cfg configs/{your config}.yml

### Acknowledgement
This codebase is heavily borrowed from [TPS](https://github.com/xing0047/TPS/tree/main)
