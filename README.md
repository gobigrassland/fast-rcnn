# Fast RCNN
Re-implement of Fast Rcnn Without Python Codes

#Contents
1. Requirements
2. Changes
3. Installation
4. Usage
5. Experiment Logs

#Requirements
External dependencies  

1. Pugixml - a light-weight C++ XML processing library. see http://pugixml.org/  
2. Matio - matlab mat file I/O library. see https://sourceforge.net/projects/matio/  
3. Cfgparser - configuration reader C++ library. see http://cfgparser.sourceforge.net/  

  Header files of these external libraries has been added in the directory - "include/caffe/3rdparty". And source files added
in "src/caffe/3rdparty".

#Changes
1. roi_data_extractor.cpp in "src/caffe/util"  
  1.1 Load bounding boxes from XML file in the PASCAL VOC format  
  1.2 Load selective search regions of interest in the Matlab format  
  1.3 Compute regression target of bounding boxes  
   
2. parse_config.cpp in "src/caffe/util"  
   Load config options for fast rcnn    
   
3. roi_data_extractor.cpp in "src/caffe/layers"  
   The data layer used during training to train a fast rcnn network  
   
4. detection.cpp in "tools"  
   Show detections in sample images. All detected images are defaultly saved in the directory "data/results"  

#Installation
  git clone https://github.com/gobigrassland/fast-rcnn.git


#Experiment logs
