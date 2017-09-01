#ifndef CAFFE_SPATIAL_SUM_OVER_MAP_LAYER_HPP_
#define CAFFE_SPATIAL_SUM_OVER_MAP_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {  
template <typename Dtype>  
class SpatialSumOverMapLayer : public Layer<Dtype> {  
 public:  
  explicit SpatialSumOverMapLayer(const LayerParameter& param)  
      : Layer<Dtype>(param) {}  
  virtual inline const char* type() const { return "SpatialSumOverMap"; }
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

 protected:  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);  
  //virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,  
  //    vector<Blob<Dtype>*>* top);  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,  
  //    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);  

  int num_;  
  int channels_;  
  int height_;  
  int width_;  
};  

}
#endif