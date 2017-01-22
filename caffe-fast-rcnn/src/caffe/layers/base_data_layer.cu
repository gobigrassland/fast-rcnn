#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
        top[1]->mutable_gpu_data());
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

template <typename Dtype>
void BaseROIPrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    BatchROI<Dtype>* batch = prefetch_roi_full_.pop("Data layer prefetch queue empty");
    //Reshape to loaded data
    top[0]->ReshapeLike(batch->data_);
    caffe_copy(batch->data_.count(), batch->data_.gpu_data(), top[0]->mutable_gpu_data());
    if(this->output_labels_)
    {
        top[2]->ReshapeLike(batch->label_);
        caffe_copy(batch->label_.count(), batch->label_.gpu_data(), top[2]->mutable_gpu_data());
    }
    top[1]->ReshapeLike(batch->rois_);
    caffe_copy(batch->rois_.count(), batch->rois_.gpu_data(), top[1]->mutable_gpu_data());
    
    top[3]->ReshapeLike(batch->bboxes_target_);
    caffe_copy(batch->bboxes_target_.count(), batch->bboxes_target_.gpu_data(), top[3]->mutable_gpu_data());
    
    top[4]->ReshapeLike(batch->bboxes_weight_);
    caffe_copy(batch->bboxes_weight_.count(), batch->bboxes_weight_.gpu_data(), top[4]->mutable_gpu_data());
    
    // Ensure the copy is synchronous wrt the host, so that the next batch isn't
    // copied in meanwhile.
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
    prefetch_roi_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);
INSTANTIATE_LAYER_GPU_FORWARD(BaseROIPrefetchingDataLayer);

}  // namespace caffe
