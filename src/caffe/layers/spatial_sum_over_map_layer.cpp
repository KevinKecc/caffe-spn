#include <cstdio>  
#include <vector>  

#include "caffe/util/im2col.hpp"   
#include "caffe/util/math_functions.hpp"  
#include "caffe/layers/spatial_sum_over_map_layer.hpp"  

namespace caffe {   
template <typename Dtype>  
void SpatialSumOverMapLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  // bottom 
  num_ = bottom[0]->num();  
  channels_ = bottom[0]->channels();  
  height_ = bottom[0]->height();    
  width_ = bottom[0]->width();     

  
  top[0]->Reshape(num_, channels_, 1, 1); 
}


template <typename Dtype>   
void SpatialSumOverMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->mutable_cpu_data();  
    Dtype* top_data = top[0]->mutable_cpu_data();    
    for (int n = 0; n < num_; n ++) {  
        for (int c = 0; c < channels_; c ++) {
            long bottom_index = (c+ channels_*n)*height_*width_;
            Dtype maxval = 0.0;
            for (int h = 0; h < height_; h ++) { 
                for (int w = 0; w < width_; w ++) {   
                    long index = bottom_index + w + h * width_;
                    if(bottom_data[index] > maxval)
                      maxval = bottom_data[index];
                }
                 top_data[c+ channels_*n] = maxval;
             }
        }  
    }     
}  



template <typename Dtype>  
void SpatialSumOverMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {  
    if (!propagate_down[0]) {  
        return;  
    }  
    // const Dtype* top_diff = top[0]->cpu_diff();  
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();  
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);  
    const Dtype* top_diff = top[0]->cpu_diff();  
    for (int n = 0; n < num_; n ++) { 
        for (int c = 0; c < channels_; c ++) {
            long bottom_index = (c+ channels_*n)*height_*width_;  
            for (int h = 0; h < height_; h ++) { 
                for (int w = 0; w < width_; w ++) {   
                    long index = bottom_index + w + h * width_;
                    bottom_diff[index] += top_diff[c+ channels_*n];  
                }  
            }   
        }  
    }   
}  

#ifdef CPU_ONLY
STUB_GPU(SpatialSumOverMapLayer);
#endif
INSTANTIATE_CLASS(SpatialSumOverMapLayer);
REGISTER_LAYER_CLASS(SpatialSumOverMap); 
}  // namespace caffe  