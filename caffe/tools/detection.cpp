#include "caffe/caffe.hpp"
#include "glog/logging.h"
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/3rdparty/matio.h"
#include "caffe/util/bboxproc.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/parse_config.hpp"
#include <sys/stat.h>

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;

DEFINE_int32(gpu, -1,
    "Run in GPU mode on given device ID.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning. "
    "Cannot be set simultaneously with snapshot.");
DEFINE_string(config, "",
    "Config options");

class Detection
{
public:
    Detection(const std::string& model_file,
            const std::string& weights_file,
            int gpu_id);
    ~Detection(){}
    
    void Initialize();
    
    void setMeans(const std::vector<float>& means);
    
    void setScales(const std::vector<int>& scales);
    
    void subMeans(const cv::Mat& im, cv::Mat& dst);
    
    void getImgBlob(const cv::Mat& im, std::vector<float>& scales_factor);
    
    void getROIBlob(float* rois_ptr, const int num, const std::vector<float> scales_factor);
    
    void detect(const float* rois_ptr,
                int rows, int cols,
                std::vector<std::vector<float> >& pred_bboxes,
                std::vector<std::vector<float> >& pred_probs);
    void clipBBox(int rows, int cols,
    		std::vector<std::vector<float> >& pred_bboxes,
    		std::vector<std::vector<float> >& pred_probs);
private:
    shared_ptr<Net<float> > _dete_net;
    std::string _model_file;
    std::string _weights_file;
    int _gpu_id;
    std::vector<float> _means;
    std::vector<int> _scales;
    Blob<float>* input_img;
    Blob<float>* input_rois;
    Blob<float>* output_probs;
    Blob<float>* output_bboxes;
    int num_classes;
};

Detection::Detection(const std::string& model_file,
        const std::string& weights_file,
        int gpu_id)
{
    _model_file = model_file;
    _weights_file = weights_file;
    _gpu_id = gpu_id;
}

void Detection::Initialize()
{
   // Set device id and mode
   if (_gpu_id >= 0)
   {
     LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
     Caffe::SetDevice(FLAGS_gpu);
     Caffe::set_mode(Caffe::GPU);
   } 
   else 
   {
     LOG(INFO) << "Use CPU.";
     Caffe::set_mode(Caffe::CPU);
   }
    _dete_net.reset(new Net<float>(_model_file, caffe::TEST));
    _dete_net->CopyTrainedLayersFrom(_weights_file);
    
    //check test prototxt
    CHECK_EQ(_dete_net->num_inputs(), 2) << "Network should have exactly two inputs.";
    CHECK_EQ(_dete_net->num_outputs(), 2) << "Network should have exactly two outpus.";
    
    Blob<float>* img_blob = _dete_net->input_blobs()[0];
    int num_channels = img_blob->channels();
    CHECK(num_channels == 1 || num_channels == 3) << "Input img should have one or three channels.";
    
    Blob<float>* rois_blob = _dete_net->input_blobs()[1];
    std::vector<int> test = rois_blob->shape();
    
    input_img = _dete_net->input_blobs()[0];
    input_rois = _dete_net->input_blobs()[1];
    output_probs = _dete_net->output_blobs()[1];
    output_bboxes = _dete_net->output_blobs()[0];
    num_classes = output_probs->shape(1);
}

void Detection::setMeans(const std::vector<float>& means)
{
    _means = means;
}
    
void Detection::setScales(const std::vector<int>& scales)
{
    _scales = scales;
}

void Detection::subMeans(const cv::Mat& im, cv::Mat& dst)
{
    int height = im.rows;
    int width = im.cols;

    im.convertTo(dst, CV_32FC3);
    for(int i = 0; i < height; i ++)
    {
            float *ptr = (float *)dst.row(i).data;
            for(int j = 0; j < width; j ++)
            {
                    ptr[3*j] -= _means[0];
                    ptr[3*j+1] -= _means[1];
                    ptr[3*j+2] -= _means[2];
            }
    }
}

void Detection::getImgBlob(const cv::Mat& im, std::vector<float>& scales_factor)
{
    cv::Mat im_sub;
    subMeans(im, im_sub);
    CHECK(im_sub.channels() == 3) << "Image blob must be three channels.";
    int size_min = std::min(im.rows, im.cols);
    int size_max = std::max(im.rows, im.cols);
    
    int width_max = 0;
    int height_max = 0;
    for(int i = 0; i < _scales.size(); i ++)
    {
        float im_scale = float(_scales[i])/size_min;
        if (im_scale * size_max > 1000)
            im_scale = float(1000) / size_max;
        scales_factor.push_back(im_scale);
        int height = round(im_scale*im_sub.rows);
        int width = round(im_scale*im_sub.cols);
        if (width_max < width)
            width_max = width;
        if (height_max < height)
            height_max = height;
    }
    input_img->Reshape(_scales.size(), 3, height_max, width_max);
    
    //_dete_net->Reshape();
    int pixels_channel = height_max * width_max;
    for(int i = 0; i < _scales.size(); i ++)
    {
        int height = round(scales_factor[i]*im_sub.rows);
        int width = round(scales_factor[i]*im_sub.cols);
        cv::Mat im_resize;
        cv::resize(im_sub, im_resize, cv::Size(width, height));
        cv::Mat temp(height_max, width_max, CV_32FC3, cv::Scalar::all(0.0f));
        im_resize.copyTo(temp(cv::Rect(0, 0, im_resize.cols, im_resize.rows)));
        std::vector<cv::Mat> vec_mat;
        cv::split(temp, vec_mat);
        for(int c = 0; c < vec_mat.size(); c ++)
        {
            int offset = input_img->offset(i,c);
            caffe::caffe_copy(pixels_channel, (float*)vec_mat[c].data, input_img->mutable_cpu_data() + offset);
        }
    }
}

void Detection::getROIBlob(float* rois_ptr, const int num, const std::vector<float> scales_factor)
{
    CHECK(num%5 == 0);
    int num_rois = num / 5;
    input_rois->Reshape(num_rois, 5, 1, 1);
    //_dete_net->Reshape();
    int num_scales = scales_factor.size();
    if(num_scales == 1)
    {
        for(int i = 0; i < num_rois; i ++)
        {
            for(int j = 1; j <= 4; j ++)
                rois_ptr[5*i+j] *= scales_factor[0];
        }
        caffe::caffe_copy(num, rois_ptr, input_rois->mutable_cpu_data());
    }
        
    else
    {
        std::vector<float> areas(num_rois), scaled_areas(num_rois), levels(num_rois);
        int level_id = 0;
        float level_area = 0.0f;
        float min_area = FLT_MAX;
        int area_ref = 224 * 224;
        for(int i = 0; i < num_rois; i ++)
        {
            areas[i] = (rois_ptr[5*i+3] - rois_ptr[5*i+2] + 1.0) * 
                    (rois_ptr[5*i+5] - rois_ptr[5*i+4] + 1.0);
            
            for(int j = 0; j < num_scales; j ++)
            {
                level_area = abs(areas[i]*_scales[j] - area_ref);
                if(level_area < min_area)
                {
                    level_id = j;
                    min_area = level_area;
                }
            }
            rois_ptr[5*i+1] = level_id;
            for(int k = 2; k <= 5; k ++)
                rois_ptr[5*i+k] *= _scales[level_id];
        }
        caffe::caffe_copy(num, rois_ptr, input_rois->mutable_cpu_data());
    }
}

void Detection::detect(const float* rois_ptr,
                       int rows, int cols,
                       std::vector<std::vector<float> >& pred_bboxes,
                       std::vector<std::vector<float> >& pred_probs)
{
    _dete_net->Reshape();
    _dete_net->ForwardPrefilled();
    
    std::vector<int> probs_shape = output_probs->shape();
    std::vector<int> bboxes_shape = output_bboxes->shape();
    int num_rois = probs_shape[0];
    int num_classes = probs_shape[1];

    const float* pred_delta = output_bboxes->cpu_data();
    const float* pred_score = output_probs->cpu_data();
    pred_bboxes.resize(num_rois);
    pred_probs.resize(num_rois);
    for(int i = 0; i < num_rois; i ++)
    {
        float center_x = (rois_ptr[5*i+1] + rois_ptr[5*i+3]) / 2;
        float center_y = (rois_ptr[5*i+4] + rois_ptr[5*i+2]) / 2;
        float width = rois_ptr[5*i+3] - rois_ptr[5*i+1] + 1.0;
        float height = rois_ptr[5*i+4] - rois_ptr[5*i+2] + 1.0;
        pred_bboxes[i].resize(num_classes*4);
        pred_probs[i].resize(num_classes);
        for(int j = 0; j < num_classes; j ++)
        {
            pred_probs[i][j] = pred_score[i*num_classes+j];
            float pred_center_x = pred_delta[i*num_classes*4+4*j] * width + center_x;
            float pred_center_y = pred_delta[i*num_classes*4+4*j+1] * height + center_y;
            float pred_width = exp(pred_delta[i*num_classes*4+4*j+2]) * width;
            float pred_height = exp(pred_delta[i*num_classes*4+4*j+3]) * height;
            int pred_left = int(pred_center_x - 0.5 * pred_width);
            int pred_right = int(pred_center_x + 0.5 * pred_width);
            int pred_top = int(pred_center_y - 0.5 * pred_height);
            int pred_bottom = int(pred_center_y + 0.5 * pred_height);

            pred_left = pred_left > 0 ? pred_left : 0;
            pred_right = pred_right < cols ? pred_right : cols - 1;
            pred_top = pred_top > 0 ? pred_top : 0;
            pred_bottom = pred_bottom < rows ? pred_bottom : rows - 1;

            if(pred_right - pred_left < 32 || pred_bottom - pred_top < 32)
            	pred_probs[i][j] = 0.0;

            pred_bboxes[i][4*j] = pred_left;
            pred_bboxes[i][4*j+1] = pred_top;
            pred_bboxes[i][4*j+2] = pred_right;
            pred_bboxes[i][4*j+3] = pred_bottom;
        }
    }
}


void readSSMat(const std::string& mat_file, std::vector<std::vector<float> > &ss_rois)
{
    mat_t* matfp = Mat_Open(mat_file.c_str(), MAT_ACC_RDONLY);
    CHECK(matfp) << "Error opening the mat file.";
    matvar_t *mat_boxes = Mat_VarRead(matfp, (char*)"boxes");
    CHECK(mat_boxes) << "Error reading boxes.";
    
    unsigned num_cell = 1;
    for(int i = 0; i < mat_boxes->rank; i ++)
        num_cell *= mat_boxes->dims[i];
    ss_rois.resize(num_cell);
    LOG(INFO) << "Parse object proposals";
    for(int i = 0; i < num_cell; i ++)
    {
        matvar_t* cell = Mat_VarGetCell(mat_boxes, i);
        unsigned cell_shape[2] = {0, 0};
        for(int j = 0; j < cell->rank; j++)
            cell_shape[j] = cell->dims[j];
        
        const double *cell_data = static_cast<const double*>(cell->data);
        ss_rois[i].resize(5*cell_shape[0] + 1);
        ss_rois[i][0] = 5*cell_shape[0];
        int num_bboxes = cell_shape[0];
        for(int j = 0; j < cell_shape[0]; j ++)
        {
            ss_rois[i][5*j+1] = 0;
            ss_rois[i][5*j+2] = (float)cell_data[j+num_bboxes] - 1.0f;
            ss_rois[i][5*j+3] = (float)cell_data[j] - 1.0f;
            ss_rois[i][5*j+4] = (float)cell_data[j+3*num_bboxes] - 1.0f;
            ss_rois[i][5*j+5] = (float)cell_data[j+2*num_bboxes] - 1.0f;
        }
    }
}


void mergebbox(const std::vector<std::vector<float> >& pred_bboxes,
               const std::vector<std::vector<float> >& pred_probs,
               int cls_id,
               float conf_thresh,
               float nms_thresh,
               std::vector<int>& ind_selected)
{
    int num_rois = pred_bboxes.size();
    std::vector<std::pair<float, int> > probs;
    probs.reserve(num_rois);
    for(int i = 0; i < num_rois; i ++)
    {
        if(pred_probs[i][cls_id] < conf_thresh)
            continue;
        probs.push_back(std::make_pair(pred_probs[i][cls_id], i));
    }
    if (probs.size() == 0)
        return;
    std::sort(probs.rbegin(), probs.rend());
    std::vector<std::vector<float> > bboxes_sorted(probs.size());
    std::vector<float> probs_sorted(probs.size());
    for(int i = 0; i < probs.size(); i ++)
    {
    	probs_sorted[i] = probs[i].first;
    	int ind = probs[i].second;
    	std::vector<float> temp(4);
    	for(int j = 0; j < 4; j ++)
    		temp[j] = pred_bboxes[ind][4*cls_id+j];
    	bboxes_sorted[i] = temp;
    }
    nms_cpu<float>(bboxes_sorted, probs_sorted, ind_selected, nms_thresh);
    for(int i = 0; i < ind_selected.size(); i ++)
        ind_selected[i] = probs[ind_selected[i]].second;
}



int main(int argc, char** argv)
{
    // Print output to stderr (while still logging).
    FLAGS_alsologtostderr = 1;
    // Usage message.
    gflags::SetUsageMessage(
      "usage: detection <command> <args>\n\n"
      "commands:\n"
      "  model           the model definition protocol buffer text file\n"
      "  weights         the trained weights\n"
      "  gpu             run in GPU mode on given device ids");
    // Run tool or show usage.
    caffe::GlobalInit(&argc, &argv);
  
    CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
    CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
    CHECK_GT(FLAGS_config.size(), 0) << "Need a config file.";
    
    ParseConfig config(FLAGS_config);
    config.ParseCommonConfig();
    struct COMMON common_cfg = config.GetCommonConfig();
    config.ParseDeployConfig();
    struct DEPLOY deploy_cfg = config.GetDeployConfig();
    
    //read images list
    std::vector<std::string> imgs_list, classes_list;
    std::ifstream infile(common_cfg.IMGS_LIST.c_str());
    std::string img_name;
    while(infile >> img_name)
            imgs_list.push_back(img_name);
    CHECK(imgs_list.size() > 0) << "Image list is empty";

    //read classes list
    std::ifstream classes_file(common_cfg.CLASSES_LIST.c_str());
    std::string class_name;
    while(classes_file >> class_name)
    {
         classes_list.push_back(class_name);
    }
    CHECK(classes_list.size() > 0) << "Classes list is empty";
    
    //Create a directory to save images
    struct stat sb;
    if(stat("data/results", &sb) == 0 && S_ISDIR(sb.st_mode))
        DLOG(INFO) << "Directory of data/results exists";
    else
    {
        int mkflag = mkdir("data/results", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        CHECK(mkflag != -1) << "Error creating directory";
    }
    
    //read selective search bboxes for every image
    //std::string mat_file = "data/selective_search_data/voc_2007_test.mat";
    std::vector<std::vector<float> > ss_rois;
    readSSMat(common_cfg.SS_MAT, ss_rois);
    Detection dete(FLAGS_model, FLAGS_weights, FLAGS_gpu);
    dete.Initialize();
    dete.setMeans(common_cfg.PIXEL_MEANS);
    dete.setScales(deploy_cfg.SCALES);
    for(int i = 0; i < imgs_list.size(); i ++)
    {
        LOG(INFO) << imgs_list[i];
        cv::Mat img = cv::imread(common_cfg.DIR_IMGS + "/" + imgs_list[i] + ".jpg", true);
        CHECK(img.data) << "Cannot find or open the image: " << imgs_list[i];
        std::vector<float> img_scales_factor;
        dete.getImgBlob(img, img_scales_factor);
        int len_rois = ss_rois[i][0];
        float *rois = new float[len_rois];
        memcpy(rois, &(ss_rois[i][1]), sizeof(float)*len_rois);
        dete.getROIBlob(rois, len_rois, img_scales_factor);
        delete [] rois;
        std::vector<std::vector<float> > pred_bboxes;
        std::vector<std::vector<float> > pred_probs;
        dete.detect(&(ss_rois[i][1]), img.rows, img.cols, pred_bboxes, pred_probs);
        
        int font_face = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 0.5;
        int thickness = 2;
        int line_type = 1;
        for(int j = 1; j < 21; j ++)
        {
            std::vector<int> index_selected;
            mergebbox(pred_bboxes, pred_probs, j, deploy_cfg.CONF_THRESH, deploy_cfg.NMS, index_selected);
            if (index_selected.size() == 0)
                continue;
            cv::Mat img_saved;
            img.copyTo(img_saved);
            for(int k = 0; k < index_selected.size(); k ++)
            {
                int ind = index_selected[k];
                char chs[128];
                sprintf(chs, "%.3f",pred_probs[ind][j]);
                std::string str_score(chs);
        
                int left = pred_bboxes[ind][4*j];
                int right = pred_bboxes[ind][4*j+2];
                int top = pred_bboxes[ind][4*j+1];
                int bottom = pred_bboxes[ind][4*j+3];
                cv::rectangle(img_saved, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255));
                cv::putText(img_saved, str_score, cv::Point(left, top), font_face, font_scale, cv::Scalar(0, 0, 255), thickness);
            }
            std::string saved_name = "data/results/" + imgs_list[i] + "_" + classes_list[j] + ".jpg";
            cv::imwrite(saved_name, img_saved);
        }
        
    }
    
    return 0;
}
