# Fast RCNN
Re-implement of Fast RCNN Without Python Codes

#Contents
1. Requirements
2. Changes
3. Installation
4. Usage
5. Experiment Logs

#Requirements
External dependencies  

1. Pugixml - a light-weight C++ XML processing library. See http://pugixml.org/  
2. Matio - matlab mat file I/O library. See https://sourceforge.net/projects/matio/  
3. Cfgparser - configuration reader C++ library. See http://cfgparser.sourceforge.net/  

  Header files of these external libraries has been added in the directory - "include/caffe/3rdparty". And source files added
in "src/caffe/3rdparty".

#Changes
Main changes compared to Caffe

1. roi_data_extractor.cpp in "src/caffe/util"  
  1.1 Load bounding boxes from XML file in the PASCAL VOC format  
  1.2 Load selective search regions of interest in the Matlab format  
  1.3 Compute regression target of bounding boxes  

2. parse_config.cpp in "src/caffe/util"  
   Load config options for fast rcnn    
   
3. roi_data_extractor.cpp in "src/caffe/layers"  
   The data layer used during training to train a fast rcnn network.  
   
4. detection.cpp in "tools"  
   Show detections in sample images. All detected images are defaultly saved in the directory "data/results".  

#Installation
  git clone https://github.com/gobigrassland/fast-rcnn.git

#Usage
 Organize the data directory. See https://github.com/rbgirshick/fast-rcnn  
 labels.txt containing all categories is located in "data/VOCdevkit/VOC2007".  
 
 Train a Fast R-CNN detector. For example, train a VGG16 network on PASCAL VOC 2007 trainval:   
 caffe/build/tools/caffe.bin --solver=models/VGG16/solver.prototxt --weights=data/imagenet_models/VGG16.v2.caffemodel --gpu=0  

 Deploy a Fast R-CNN detector. For example, deploy the VGG16 network on PASCAL VOC 2007 test:  
 Firstly, you need change the values of [COMMON]-[IMGS_LIST] and [COMMON]-[SS_MAT].  
 caffe/build/tools/detection.bin --solver=models/VGG16/test.prototxt --weights=data/imagenet_models/VGG16_fast_rcnn_iter_40000.caffemodel  
 --config=cfg/config.cfg --gpu=0
#Experiment logs
 Experiment logs are located in "logs".