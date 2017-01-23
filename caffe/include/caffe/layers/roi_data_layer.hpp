#ifndef CAFFE_ROI_DATA_LAYER_HPP
#define CAFFE_ROI_DATA_LAYER_HPP

#include <opencv2/opencv.hpp>

#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/roi_data_extractor.hpp"
#include "caffe/util/parse_config.hpp"

namespace caffe
{

    template <typename Dtype>
    class ROIDataLayer : public BaseROIPrefetchingDataLayer<Dtype>
    { 
        public:
            explicit ROIDataLayer(const LayerParameter& param):BaseROIPrefetchingDataLayer<Dtype>(param){}
            
            virtual ~ROIDataLayer();

            virtual void DataLayerSetUp(const vector<Blob<Dtype>*> &bottom,
                    const vector<Blob<Dtype>*> &top);
            
            virtual inline const char* type() const {return "ROIData";}
            
            virtual inline int ExactNumTops() const {return -1;}
            
            virtual inline int MinTopBlobs() const {return 3;}
            
            virtual inline int MaxTopBlobs() const {return 5;}
            
        protected:
            
            void ReadCFGParameter();

            void ShuffleROIdbIndex();

            void PrepImForBlob(const cv::Mat& im, cv::Mat& dst, int target_size, float &im_scale);

            //build an input blob from the images in the roidb_ at the specified scales
            void GetImageBlob(const vector<int>& images_ind,
            		const vector<int>& scales,
                        vector<float>& vec_im_scales,
                        BatchROI<Dtype>* batch);
            
            void SampleROIs(const vector<int>& images_ind,
                    const vector<float>& random_scales,
                    int fg_rois_per_image,
                    int rois_per_image,
                    int num_classes,
                    BatchROI<Dtype>* batch);

            void GetNextBatchIndex(vector<int>& next_batch_inds);

            void GetNextBatch(const vector<int>& next_batch_inds,
                              BatchROI<Dtype>* batch);

            virtual void load_batch(BatchROI<Dtype>* batch);

        private:
            struct TRAIN train_cfg_;
            struct COMMON common_cfg_;
            vector<ROI> roidb_;
            vector<string> img_name_list_;
            vector<string> classes_list_;
            //number of roidbs
            int num_roidb_;
            shared_ptr<Caffe::RNG> rng_;
            // current index of roidb
            int cur_ind_;
            vector<int> perm_;
                
        
    };
}

#endif /* CAFFE_ROI_DATA_LAYER_HPP */

