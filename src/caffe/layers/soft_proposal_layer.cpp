#include <cstdio>  
#include <vector>
#include <sstream>
#include <boost/filesystem.hpp>

#include "caffe/util/im2col.hpp"   
#include "caffe/util/math_functions.hpp"  
#include "caffe/layers/soft_proposal_layer.hpp" 
#include "caffe/util/io.hpp"  

namespace caffe {  

using ::boost::filesystem::path;

// Layer setup 
template <typename Dtype>  
void SoftProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const SoftProposalParameter& soft_proposal_param = this->layer_param_.soft_proposal_param(); 
    
  dm_folder_ = soft_proposal_param.dm_folder();
  if(soft_proposal_param.has_tolerance()) 
    tolerance_ = soft_proposal_param.tolerance();
  else
    tolerance_ = 0.0001;
  if(soft_proposal_param.has_maxiteration()) 
    maxIteration_ = soft_proposal_param.maxiteration();
  else
    maxIteration_ = 20;
  LOG(INFO) << "tolerance: " << tolerance_ << ", maxIteration: " << maxIteration_;
    
  // bottom 
  num_ = bottom[0]->num();  
  channels_ = bottom[0]->channels();  
  height_ = bottom[0]->height();    
  width_ = bottom[0]->width(); 
  N_ =  height_* width_; 
  
  if(soft_proposal_param.has_factor())
    factor_ = soft_proposal_param.factor();
  else
    factor_ = width_*0.15;
  LOG(INFO) << "factor_:" << factor_; 
  
  
  proposal_.Reshape(num_, 1, height_, width_);
  vector<int> mult_dims(2, N_);
  distanceMetric_.Reshape(mult_dims);
  transferMatrix_.Reshape(mult_dims);
}  

// sum function
template <typename Dtype>
Dtype SoftProposalLayer<Dtype>::sumall(const int nthreads, Dtype* src_data){
  Dtype sumval=0.0;
  for (long i=0;i<nthreads;i++)
  {
    sumval += src_data[i];
  }
  return sumval;
}

// for page rank
template <typename Dtype>  
void SoftProposalLayer<Dtype>::InitDistanceMetricKernel(){
  Dtype* distanceMetric_data = distanceMetric_.mutable_cpu_data();  
  
  
  if(!boost::filesystem::exists(dm_folder_))
  {
    LOG(INFO) << dm_folder_.c_str();
    boost::filesystem::create_directory(dm_folder_);
  }
    
  stringstream s;
  s<< dm_folder_ << "/dm_" << height_ << "_" << width_ << ".blob";;
  string dm_filename=s.str();
  if(boost::filesystem::exists(dm_filename.c_str()))
  {
    BlobProto dm;
    ReadProtoFromBinaryFile(dm_filename, &dm);
    distanceMetric_.FromProto(dm);
    return;
  }
  
  long nthreads = N_ * N_;
  for(long n=0;n<nthreads;n++)
  {
    const long q = n % width_; 
    const long p = (n / width_) % height_;
    const long j = (n / N_) % width_;
    const long i = n / N_ / width_;
    const long u = i * width_ + j;
    const long v = p * width_ + q;
    
    if (u >= v) {
        *(distanceMetric_data + n) = expf(((i - p) * (i - p) + (j - q) * (j - q)) / (-2 * factor_ * factor_));
        *(distanceMetric_data + v*N_ + u) = *(distanceMetric_data + n);
    }
  } 
  
  BlobProto dm;
  distanceMetric_.ToProto(&dm, false);
  WriteProtoToBinaryFile(dm, dm_filename.c_str());   
}

template <typename Dtype> 
void SoftProposalLayer<Dtype>::InitTransferMatrixKernel(const Dtype* input_data){
  InitDistanceMetricKernel();
  
  Dtype* distanceMetric_data = distanceMetric_.mutable_cpu_data(); 
  Dtype* transferMatrix_data = transferMatrix_.mutable_cpu_data();   

  long nthreads = N_ * N_;
  for(long n=0; n<nthreads;n++)
  {
    const long q = n % width_; 
    const long p = (n / width_) % height_;
    const long j = (n / N_) % width_;
    const long i = n / N_ / width_;
    const long u = i * width_ + j;
    const long v = p * width_ + q;
    
    if (i*j >= p*q) {
      long c;
      float sum = 0.0f;
      for (c = 0; c < channels_; c++) {
        const float pntA = *(input_data + c * N_ + i * width_ + j);
        const float pntB = *(input_data + c * N_ + p * width_ + q);
        sum += (pntA - pntB) * (pntA - pntB);
      }
    
    *(transferMatrix_data + n) = sqrt(sum) + *(distanceMetric_data + n);
    *(transferMatrix_data + v*N_ + u) = *(transferMatrix_data + n);
    }
  }
}

template <typename Dtype>
void SoftProposalLayer<Dtype>::NormTransferMatrixKernel(){
  Dtype* transferMatrix_data = transferMatrix_.mutable_cpu_data(); 
  
  for(long n=0;n<N_;n++)
  {
    long c;
    float sum = 0.0f;
    for (c = 0; c < N_; c++) {
        sum += *(transferMatrix_data + c * N_ + n);
    }
    for (c = 0; c < N_; c++) {
        *(transferMatrix_data + c * N_ + n) /= sum;
    }
  }
}

template <typename Dtype>
void SoftProposalLayer<Dtype>::UpdateProposalKernel(const int nthreads, 
    const Dtype* src_data,
    const Dtype* diff_data,
    Dtype* dst_data,
    const float scale){
  for(long n=0;n<nthreads;n++)
  {
    if (scale < 0) 
        *(dst_data + n) = *(src_data + n) + *(diff_data + n);
    else
        *(dst_data + n) = *(src_data + n) * scale;
  }
}
 
template <typename Dtype>
void SoftProposalLayer<Dtype>::Generate(const Dtype* src_data){ 
  Dtype* proposal_data = proposal_.mutable_cpu_data();
  Dtype* transferMatrix_data = transferMatrix_.mutable_cpu_data(); 
   
  Blob<Dtype> proposalBuffer_; 
  proposalBuffer_.Reshape(1, 1, height_, width_);
  Dtype* proposalBuffer_data = proposalBuffer_.mutable_cpu_data(); 
    
  
  const float avg = 1.0f / N_;
  float sumOver;
  long i, j;
  
  caffe_set(proposal_.count(), Dtype(avg), proposal_data);
  for (i = 0; i < num_; i++)
  {
    InitTransferMatrixKernel(src_data+ i*channels_*N_);
    NormTransferMatrixKernel();
    
    caffe_set(proposalBuffer_.count(), Dtype(avg), proposalBuffer_data);
    for (j = 0; j < maxIteration_; j++) {
        // calculate diffs
        //caffe_copy(proposalBuffer_.count(), proposalBuffer_data, proposal_data);
        caffe_cpu_gemv<Dtype>(CblasNoTrans, N_, N_, 1.,
          transferMatrix_data, proposal_data, -1., proposalBuffer_data);
        
        float normDiff = sqrt(proposalBuffer_.sumsq_data());
        // LOG(INFO) << "normDiff(" << j << "): " << normDiff;
        if (normDiff < tolerance_) break;
        // add diffs
        UpdateProposalKernel(N_, proposal_data, proposalBuffer_data, proposalBuffer_data, -1.0f);

        sumOver = sumall(N_, proposalBuffer_data);
        if (sumOver < 0) break;
        // update proposal
        UpdateProposalKernel(N_, proposalBuffer_data, proposal_data, proposal_data, 1.0f / sumOver);
      }
      proposal_data += N_;
  }
}

template <typename Dtype>
void SoftProposalLayer<Dtype>::Couple_cpu(const Dtype* input_data, Dtype* output_data){
  Dtype* proposal_data = proposal_.mutable_cpu_data();  
  for (int n = 0; n<num_; n++)
  {
    for (int c=0; c<channels_; c++)
    { 
      long loc = (n*channels_+c)*N_;
      long locp = n*N_;
      for(long l=0;l<N_;l++)
      {
        *(output_data + loc + l) = *(input_data + loc+ l) * *(proposal_data + locp + l);
      }
    }
  }
}

template <typename Dtype>   
void SoftProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data(); 
  Dtype* top_data = top[0]->mutable_cpu_data();
  
  InitTransferMatrixKernel(bottom_data);  
  NormTransferMatrixKernel(); 
  
  Generate(bottom_data);
  
  Couple_cpu(bottom_data, top_data);  
}  

template <typename Dtype>  
void SoftProposalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down,  const vector<Blob<Dtype>*>& bottom) {  
    const Dtype* top_diff = top[0]->cpu_diff();  
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();  
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);  
     
    Couple_cpu(top_diff, bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(SoftProposalLayer);
#endif
INSTANTIATE_CLASS(SoftProposalLayer);
REGISTER_LAYER_CLASS(SoftProposal); 
}  // namespace caffe  