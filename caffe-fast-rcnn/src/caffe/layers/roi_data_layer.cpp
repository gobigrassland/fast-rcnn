#include "caffe/layers/roi_data_layer.hpp"
#include <fstream>
#include <iostream>

#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include <sys/stat.h>
#include <time.h>
#include "opencv2/opencv.hpp"
#include "caffe/util/roi_data_extractor.hpp"
#include "caffe/util/parse_config.hpp"
#include <vector>
#include <string>


namespace caffe
{

    template<typename Dtype>
    ROIDataLayer<Dtype>::~ROIDataLayer()
    {
    	this->StopInternalThread();
    }
    
    template<typename Dtype>
    void ROIDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*> &bottom,
                    const vector<Blob<Dtype>*> &top)
    {
        string config_file = this->layer_param_.roi_data_param().config_file();
        ParseConfig config(config_file);
        config.ParseTrainConfig();
        config.ParseCommonConfig();
        train_cfg_ = config.GetTrainConfig();
        common_cfg_ = config.GetCommonConfig();
        std::cout << std::endl;

        std::ifstream infile(common_cfg_.IMGS_LIST.c_str());
        string img_name;
        while(infile >> img_name)
        	img_name_list_.push_back(img_name);
        CHECK(img_name_list_.size() > 0) << "Image list is empty";

        std::ifstream classes_file(common_cfg_.CLASSES_LIST.c_str());
        string class_name;
        while(classes_file >> class_name)
        {
             classes_list_.push_back(class_name);
        }
        int num_classes = classes_list_.size();
        CHECK(num_classes > 0) << "Classes list is empty";

        //Read an image to initialize the top blob
        cv::Mat cv_img = ReadImageToCVMat(common_cfg_.DIR_IMGS + "/" + img_name_list_[0] + ".jpg", true);
        CHECK(cv_img.data) << "Could not load " << img_name_list_[0];

        int height = cv_img.rows;
        int width = cv_img.cols;
        int channels = cv_img.channels();

        top[0]->Reshape(train_cfg_.IMS_PER_BATCH, channels, height, width);
        top[1]->Reshape(train_cfg_.BATCH_SIZE, 5, 1, 1);
        top[2]->Reshape(train_cfg_.BATCH_SIZE, 1, 1, 1);
        top[3]->Reshape(train_cfg_.BATCH_SIZE, 4*num_classes, 1, 1);
        top[4]->Reshape(train_cfg_.BATCH_SIZE, 4*num_classes, 1, 1);
        for(int i = 0; i < this->PREFETCH_COUNT; i ++)
        {
            this->prefetch_roi_[i].data_.Reshape(train_cfg_.IMS_PER_BATCH, channels, height, width);
            this->prefetch_roi_[i].rois_.Reshape(train_cfg_.BATCH_SIZE, 5, 1, 1);
            this->prefetch_roi_[i].label_.Reshape(train_cfg_.BATCH_SIZE, 1, 1, 1);
            this->prefetch_roi_[i].bboxes_target_.Reshape(train_cfg_.BATCH_SIZE, 4*num_classes, 1, 1);
            this->prefetch_roi_[i].bboxes_weight_.Reshape(train_cfg_.BATCH_SIZE, 4*num_classes, 1, 1);
                   
        }
        DLOG(INFO) << "Input img size: " << top[0]->num() << ", " << top[0]->channels() << ", "
                << top[0]->height() << ", " << top[0]->width();
        DLOG(INFO) << "Input roi size: " << top[1]->num() << ", " << top[1]->channels() << ", "
                << top[1]->height() << ", " << top[1]->width();
        DLOG(INFO) << "Input label size: " << top[2]->num() << ", " << top[2]->channels() << ", "
                << top[2]->height() << ", " << top[2]->width();
        DLOG(INFO) << "Input target size: " << top[3]->num() << ", " << top[3]->channels() << ", "
                << top[3]->height() << ", " << top[3]->width();
        DLOG(INFO) << "Input target weight size: " << top[4]->num() << ", " << top[4]->channels() << ", "
                << top[4]->height() << ", " << top[4]->width();

        ROIDataExtractor roi_data_extractor(common_cfg_.DIR_IMGS,
        		common_cfg_.IMGS_LIST,
                common_cfg_.CLASSES_LIST,
                common_cfg_.DIR_ANNOTATIONS,
                common_cfg_.SS_MAT,
                train_cfg_.USE_FLIPPED);
        roi_data_extractor.roi_data_extract(roidb_);
        num_roidb_ = roidb_.size();
        cur_ind_ = 0;
        perm_.resize(num_roidb_);
        for(int i = 0; i < num_roidb_; i ++)
        	perm_[i] = i;
        const unsigned int rng_seed = caffe_rng_rand();
        rng_.reset(new Caffe::RNG(rng_seed));
        ShuffleROIdbIndex();
    }
    

    template<typename Dtype>
    void ROIDataLayer<Dtype>::ShuffleROIdbIndex()
    {
    	CHECK(rng_);
    	caffe::rng_t* random_generator = static_cast<caffe::rng_t*>(rng_->generator());
    	shuffle(perm_.begin(), perm_.end(), random_generator);
    	cur_ind_ = 0;
    }
    
    template<typename Dtype>
    void ROIDataLayer<Dtype>::PrepImForBlob(const cv::Mat& im, cv::Mat& dst, int target_size, float& im_scale)
    {
    	int height = im.rows;
    	int width = im.cols;

    	im.convertTo(dst, CV_32FC3);
    	for(int i = 0; i < height; i ++)
    	{
    		float *ptr = (float *)dst.row(i).data;
    		for(int j = 0; j < width; j ++)
    		{
    			ptr[3*j] -= common_cfg_.PIXEL_MEANS[0];
    			ptr[3*j+1] -= common_cfg_.PIXEL_MEANS[1];
    			ptr[3*j+2] -= common_cfg_.PIXEL_MEANS[2];
    		}
    	}
    	int im_size_min = std::min(height, width);
    	int im_size_max = std::max(height, width);
        
    	im_scale = float(target_size)/float(im_size_min);
    	if (round(im_scale * im_size_max) > train_cfg_.MAX_SIZE)
    		im_scale = float(train_cfg_.MAX_SIZE) / float(im_size_max);
    	cv::resize(dst, dst, cv::Size(round(im_scale*width), round(im_scale*height)));

    }
    
    
    template<typename Dtype>
    void ROIDataLayer<Dtype>::GetImageBlob(const vector<int>& images_ind,
    		const vector<int>& scales,
                vector<float>& vec_im_scales,
            BatchROI<Dtype>* batch)
    {
    	CHECK(images_ind.size() == scales.size());
    	int num_images = images_ind.size();
        vector<cv::Mat> vec_ims(num_images);
        vec_im_scales.resize(num_images);
    	for(int i = 0; i < num_images; i ++)
    	{
    		int ind = images_ind[i];
                if (ind >= img_name_list_.size())
                    ind -= img_name_list_.size();
    		cv::Mat img = ReadImageToCVMat(common_cfg_.DIR_IMGS + "/" + img_name_list_[ind] + ".jpg", true);
    		if(roidb_[i].flipped == true)
    			cv::flip(img, img, 1);
    		int target_size = scales[i];
    		PrepImForBlob(img, vec_ims[i], target_size, vec_im_scales[i]);
    	}
        int max_height = vec_ims[0].rows;
        int max_width = vec_ims[0].cols;
        for(int i = 1; i < num_images; i ++)
        {
            max_height = std::max(max_height, vec_ims[i].rows);
            max_width = std::max(max_width, vec_ims[i].cols);
        }

        batch->data_.Reshape(num_images, 3, max_height, max_width);
        int num_pixels_each_plane = batch->data_.count(2);


        for(int i = 0; i < num_images; i ++)
        {
            cv::Mat mat_temp(max_height, max_width, CV_32FC3, cv::Scalar::all(0));
            // ROI copy of cv::Mat failed in this manner
            //mat_temp(cv::Range(0,max_height), cv::Range(0,max_width)) = vec_ims[i];
            //cv::Rect(, , width, height))
            vec_ims[i].copyTo(mat_temp(cv::Rect(0,0,vec_ims[i].cols,vec_ims[i].rows)));
            vector<cv::Mat> vec_mat_temp;
            cv::split(mat_temp, vec_mat_temp);
            for(int c = 0; c < vec_mat_temp.size(); c ++)
            {
                int offset_cur = batch->data_.offset(i, c);
                caffe_copy(num_pixels_each_plane, (Dtype*)vec_mat_temp[c].data, batch->data_.mutable_cpu_data()+offset_cur);
            }
        }
    }
    
    template<typename Dtype>
    void ROIDataLayer<Dtype>::SampleROIs(const vector<int>& images_ind,
            const vector<float>& random_scales,
            int fg_rois_per_image,
            int rois_per_image,
            int num_classes,
            BatchROI<Dtype>* batch)
    {
    	int num_images = images_ind.size();
    	Dtype *labels_array = new Dtype[rois_per_image * num_images];
    	Dtype *boxes_array = new Dtype[5 * rois_per_image * num_images];
    	Dtype *targets_array = new Dtype[4*num_classes*rois_per_image*num_images];
    	Dtype *targets_loss_weight_array = new Dtype[4*num_classes*rois_per_image*num_images];
        memset(labels_array, 0, rois_per_image * num_images * sizeof(Dtype));
        memset(boxes_array, 0, 5 * rois_per_image * num_images * sizeof(Dtype));
    	memset(targets_array, 0, 4*num_classes*rois_per_image*num_images * sizeof(Dtype));
    	memset(targets_loss_weight_array, 0, 4*num_classes*rois_per_image*num_images * sizeof(Dtype));
    	for(int k = 0; k < num_images; k ++)
    	{
        	int ind = images_ind[k];
            vector<vector<double> > *overlaps = &roidb_[ind].gt_overlaps;
            vector<int> fg_inds, bg_inds;
            for(int i = 0; i < (*overlaps).size(); i ++)
            {
            	if ((*overlaps)[i][2] >= train_cfg_.FG_THRESH)
            		fg_inds.push_back(i);
            	if ((*overlaps)[i][2] >= train_cfg_.BG_THRESH_LO && (*overlaps)[i][2] < train_cfg_.BG_THRESH_HI)
            		bg_inds.push_back(i);
            }

            int fg_rois_per_this_image = fg_inds.size() > fg_rois_per_image ?
            		fg_rois_per_image : fg_inds.size();

            int bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image;
            bg_rois_per_this_image = bg_inds.size() > bg_rois_per_this_image ?
            		bg_rois_per_this_image : bg_inds.size();

           
            //Sample foreground/background regions
            CHECK(rng_);
            caffe::rng_t* random_generator = static_cast<caffe::rng_t*>(rng_->generator());
            if (fg_inds.size() > 0)
        	shuffle(fg_inds.begin(), fg_inds.end(), random_generator);
            if (bg_inds.size() > 0)
        	shuffle(bg_inds.begin(), bg_inds.end(), random_generator);
            
        	for(int i = 0; i < fg_rois_per_this_image; i ++)
        	{
        		int ind_fg_roi_chosed = fg_inds[i];
        		int ind_class = roidb_[ind].gt_overlaps[ind_fg_roi_chosed][1];
        		labels_array[k*rois_per_image+i] = ind_class;
        		boxes_array[5*k*rois_per_image+5*i] = k;
        		for(int j = 0; j < 4; j ++)
        			boxes_array[5*(k*rois_per_image+i)+j+1] = roidb_[ind].boxes[ind_fg_roi_chosed][j] * random_scales[k];

        		for(int j = 4*ind_class, j_ = 0; j < 4*(ind_class+1), j_ < 4; j ++, j_ ++)
        		{
        			targets_array[4*num_classes*(k*rois_per_image+i) + j] = roidb_[ind].targets[ind_fg_roi_chosed][j_+1];
        			targets_loss_weight_array[4*num_classes*(k*rois_per_image+i) + j] = (Dtype)1;
        		}

        	}
        	for(int i = fg_rois_per_this_image; i < fg_rois_per_this_image + bg_rois_per_this_image; i ++)
        	{
        		int ind_bg_roi_chosed = bg_inds[i-fg_rois_per_this_image];
        		labels_array[k*rois_per_image+i] = 0;
        		boxes_array[5*k*rois_per_image+5*i] = k;
        		for(int j = 0; j < 4; j ++)
        			boxes_array[5*k*rois_per_image+5*i+j+1] = roidb_[ind].boxes[ind_bg_roi_chosed][j] * random_scales[k];
        	}
    	}

    	caffe_copy(rois_per_image * num_images, labels_array, batch->label_.mutable_cpu_data());
    	caffe_copy(5 * rois_per_image * num_images, boxes_array, batch->rois_.mutable_cpu_data());
    	caffe_copy(4*num_classes*rois_per_image*num_images, targets_array, batch->bboxes_target_.mutable_cpu_data());
    	caffe_copy(4*num_classes*rois_per_image*num_images, targets_loss_weight_array, batch->bboxes_weight_.mutable_cpu_data());
        
    	delete [] labels_array;
    	delete [] boxes_array;
    	delete [] targets_array;
    	delete [] targets_loss_weight_array;
    }
    
        template<typename Dtype>
    void ROIDataLayer<Dtype>::GetNextBatchIndex(vector<int>& next_batch_inds)
    {
    	next_batch_inds.clear();
    	if (cur_ind_ + train_cfg_.IMS_PER_BATCH >= num_roidb_)
    		ShuffleROIdbIndex();
    	for(int i = 0; i < train_cfg_.IMS_PER_BATCH; i ++)
    		next_batch_inds.push_back(perm_[cur_ind_ + i]);
    	cur_ind_ += train_cfg_.IMS_PER_BATCH;
    }
    

    template<typename Dtype>
    void ROIDataLayer<Dtype>::GetNextBatch(const vector<int>& next_batch_inds,
            BatchROI<Dtype>* batch)
    {
    	int num_scales = train_cfg_.SCALES.size();
    	int num_images = next_batch_inds.size();
    	CHECK(rng_);
    	caffe::rng_t *rng = static_cast<caffe::rng_t*>(rng_->generator());
    	//sample random scales to use for each image in this batch
    	vector<int> random_scales(num_images);
    	for(int i = 0; i < num_images; i ++)
    		random_scales[i] = train_cfg_.SCALES[(*rng)() % num_scales];

    	CHECK(train_cfg_.BATCH_SIZE % num_images == 0);
    	int rois_per_image = train_cfg_.BATCH_SIZE/ num_images;
    	int fg_rois_per_image = round(train_cfg_.FG_FRACTION * rois_per_image);

        std::vector<float> scale_ratios;
    	GetImageBlob(next_batch_inds, random_scales, scale_ratios, batch);
    	SampleROIs(next_batch_inds, scale_ratios, fg_rois_per_image, rois_per_image, classes_list_.size(), batch);
    }
    
    template<typename Dtype>
    void ROIDataLayer<Dtype>::load_batch(BatchROI<Dtype>* batch)
    {  
        CHECK(batch->data_.count());
        CHECK(batch->label_.count());
        CHECK(batch->rois_.count());
        CHECK(batch->bboxes_target_.count());
        CHECK(batch->bboxes_weight_.count());
    	vector<int> next_batch_inds;
    	GetNextBatchIndex(next_batch_inds);
    	GetNextBatch(next_batch_inds, batch);
    }

#ifdef CPU_ONLY
STUB_GPU(ROIDataLayer);
#endif

INSTANTIATE_CLASS(ROIDataLayer);
REGISTER_LAYER_CLASS(ROIData);
    
}
