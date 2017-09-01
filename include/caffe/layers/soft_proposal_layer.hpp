#ifndef CAFFE_SOFT_PROPOSAL_LAYER_HPP_
#define CAFFE_SOFT_PROPOSAL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {  
template <typename Dtype>  
class SoftProposalLayer : public NeuronLayer<Dtype> {  
 public:  
  explicit SoftProposalLayer(const LayerParameter& param)  
      : NeuronLayer<Dtype>(param) {}  
  virtual inline const char* type() const { return "SoftProposalLayer"; }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  /**
   * @brief Compute the soft proposal mask.
   */ 
  //step1: initialize the distance matrix
  void InitDistanceMetricKernel();
  //step2: initialize the transfer matrix with the distance matrix
  void InitTransferMatrixKernel(const Dtype* input_data);
  // step3: normalize transfer matrix
  void NormTransferMatrixKernel();
  // Step4: iteratively updata soft proposal mask
  void UpdateProposalKernel(const int nthreads, 
    const Dtype* src_data,
    const Dtype* diff_data,
    Dtype* dst_data,
    const float scale);
  // generate soft proposal mask using step1-step4
  void Generate(const Dtype* src_data);
  // compute the top blob data with soft proposal mask  
  void Couple_cpu(const Dtype* src_data, Dtype* dst_data);
  //void Couple_gpu(const Dtype* src_data, Dtype* dst_data);
  
  // math functions
  Dtype sumall(const int nthreads, Dtype* src_data);
  
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
  int N_;
  
  float factor_;
  float tolerance_;
  int maxIteration_;
  string dm_folder_;
  Blob<Dtype> proposal_; 
  Blob<Dtype> distanceMetric_; 
  Blob<Dtype> transferMatrix_;
};  

}
#endif