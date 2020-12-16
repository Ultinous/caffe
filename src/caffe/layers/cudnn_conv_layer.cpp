#ifdef USE_CUDNN
#include <algorithm>
#include <vector>

#include "caffe/layers/cudnn_conv_layer.hpp"

namespace caffe {

// Set to three for the benefit of the backward pass, which
// can use separate streams for calculating the gradient w.r.t.
// bias, filter weights, and bottom data for each group independently
#define CUDNN_STREAMS_PER_GROUP 3

/**
 * TODO(dox) explain cuDNN interface
 */
template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
// Initialize CUDA streams and cuDNN.
#if !CUDNN_VERSION_MIN(7,0,0)
  stream_         = new cudaStream_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
  handle_         = new cudnnHandle_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
  end_event_      = new cudaEvent_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
#endif

  // Initialize algorithm arrays
  fwd_algo_       = new cudnnConvolutionFwdAlgo_t[bottom.size()];
  bwd_filter_algo_= new cudnnConvolutionBwdFilterAlgo_t[bottom.size()];
  bwd_data_algo_  = new cudnnConvolutionBwdDataAlgo_t[bottom.size()];

  // initialize size arrays
  workspace_fwd_sizes_ = new size_t[bottom.size()];
  workspace_bwd_filter_sizes_ = new size_t[bottom.size()];
  workspace_bwd_data_sizes_ = new size_t[bottom.size()];

  // workspace data
  workspaceSizeInBytes = 0;
  workspaceData = NULL;
  workspace = new void*[this->group_ * CUDNN_STREAMS_PER_GROUP];

  for (size_t i = 0; i < bottom.size(); ++i) {
    // initialize all to default algorithms
    fwd_algo_[i] = (cudnnConvolutionFwdAlgo_t)0;
    bwd_filter_algo_[i] = (cudnnConvolutionBwdFilterAlgo_t)0;
    bwd_data_algo_[i] = (cudnnConvolutionBwdDataAlgo_t)0;
    // default algorithms don't require workspace
    workspace_fwd_sizes_[i] = 0;
    workspace_bwd_data_sizes_[i] = 0;
    workspace_bwd_filter_sizes_[i] = 0;
  }
#if !CUDNN_VERSION_MIN(7,0,0)
  CUDA_CHECK(cudaEventCreateWithFlags(&start_event_,cudaEventDisableTiming));
  for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
    CUDA_CHECK(cudaStreamCreate(&stream_[g]));
    CUDNN_CHECK(cudnnCreate(&handle_[g]));
    CUDNN_CHECK(cudnnSetStream(handle_[g], stream_[g]));
    CUDA_CHECK(cudaEventCreateWithFlags(&end_event_[g],cudaEventDisableTiming));
    workspace[g] = NULL;
  }
#endif
  // Set the indexing parameters.
#if CUDNN_VERSION_MIN(7,0,0)
  bias_offset_ = this->num_output_;
#else
  bias_offset_ = (this->num_output_ / this->group_);
#endif

  // Create filter descriptor.
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int kernel_h = kernel_shape_data[0];
  const int kernel_w = kernel_shape_data[1];
#if CUDNN_VERSION_MIN(7,0,0)
  cudnn::createFilterDesc<Dtype>(&filter_desc_,
      this->num_output_, this->channels_ / this->group_,
      kernel_h, kernel_w);
#else
  cudnn::createFilterDesc<Dtype>(&filter_desc_,
      this->num_output_ / this->group_, this->channels_ / this->group_,
      kernel_h, kernel_w);
#endif

  // Create tensor descriptor(s) for data and corresponding convolution(s).
  for (int i = 0; i < bottom.size(); i++) {
    cudnnTensorDescriptor_t bottom_desc;
    cudnn::createTensor4dDesc<Dtype>(&bottom_desc);
    bottom_descs_.push_back(bottom_desc);
    cudnnTensorDescriptor_t top_desc;
    cudnn::createTensor4dDesc<Dtype>(&top_desc);
    top_descs_.push_back(top_desc);
    cudnnConvolutionDescriptor_t conv_desc;
    cudnn::createConvolutionDesc<Dtype>(&conv_desc);
#if CUDNN_VERSION_MIN(7,0,0)
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc,this->group_));
#endif
    conv_descs_.push_back(conv_desc);
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::createTensor4dDesc<Dtype>(&bias_desc_);
  }

  handles_setup_ = true;
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(2, this->num_spatial_axes_)
      << "CuDNNConvolution input must have 2 spatial axes "
      << "(e.g., height and width). "
      << "Use 'engine: CAFFE' for general ND convolution.";
#if CUDNN_VERSION_MIN(7,0,0)
  bottom_offset_ = this->bottom_dim_;
  top_offset_ = this->top_dim_;
#else
  bottom_offset_ = this->bottom_dim_ / this->group_;
  top_offset_ = this->top_dim_ / this->group_;
#endif
  const int height = bottom[0]->shape(this->channel_axis_ + 1);
  const int width = bottom[0]->shape(this->channel_axis_ + 2);
  const int height_out = top[0]->shape(this->channel_axis_ + 1);
  const int width_out = top[0]->shape(this->channel_axis_ + 2);
  const int* pad_data = this->pad_.cpu_data();
  const int pad_h = pad_data[0];
  const int pad_w = pad_data[1];
  const int* stride_data = this->stride_.cpu_data();
  const int stride_h = stride_data[0];
  const int stride_w = stride_data[1];
#if CUDNN_VERSION_MIN(8, 0, 0)
  int ret_count;
  bool found_conv_algorithm;
  std::vector<cudnnConvolutionFwdAlgoPerf_t> fwd_algo_pref_;
  std::vector<cudnnConvolutionBwdDataAlgoPerf_t> bwd_data_algo_pref_;
#endif
  // Specify workspace limit for kernels directly until we have a
  // planning strategy and a rewrite of Caffe's GPU memory mangagement
  size_t workspace_limit_bytes = 8*1024*1024;

  for (int i = 0; i < bottom.size(); i++) {
#if CUDNN_VERSION_MIN(7,0,0)
    cudnn::setTensor4dDesc<Dtype>(&bottom_descs_[i],
        this->num_,
        this->channels_, height, width,
        this->channels_ * height * width,
        height * width, width, 1);
    cudnn::setTensor4dDesc<Dtype>(&top_descs_[i],
        this->num_,
        this->num_output_, height_out, width_out,
        this->num_output_ * this->out_spatial_dim_,
        this->out_spatial_dim_, width_out, 1);
#else
    cudnn::setTensor4dDesc<Dtype>(&bottom_descs_[i],
        this->num_,
        this->channels_ / this->group_, height, width,
        this->channels_ * height * width,
        height * width, width, 1);
    cudnn::setTensor4dDesc<Dtype>(&top_descs_[i],
        this->num_,
        this->num_output_ / this->group_, height_out, width_out,
        this->num_output_ * this->out_spatial_dim_,
        this->out_spatial_dim_, width_out, 1);
#endif
    cudnn::setConvolutionDesc<Dtype>(&conv_descs_[i], bottom_descs_[i],
        filter_desc_, pad_h, pad_w,
        stride_h, stride_w);
#if CUDNN_VERSION_MIN(8,0,0)
    int max_alg_count = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithmMaxCount(Caffe::cudnn_handle(), &max_alg_count));
    fwd_algo_pref_.resize(max_alg_count);

    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(Caffe::cudnn_handle(),
      bottom_descs_[i],
      filter_desc_,
      conv_descs_[i],
      top_descs_[i],
      max_alg_count,
      &ret_count,
      fwd_algo_pref_.data()));

    found_conv_algorithm = false;
    for(int n=0;n<ret_count; n++){
      if (fwd_algo_pref_[n].status == CUDNN_STATUS_SUCCESS &&
          fwd_algo_pref_[n].algo != CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED &&
          fwd_algo_pref_[n].memory < workspace_limit_bytes){
        found_conv_algorithm = true;
        fwd_algo_[i]                   = fwd_algo_pref_[n].algo;
        workspace_fwd_sizes_[i]        = fwd_algo_pref_[n].memory;
        break;
      }
    }
    if(!found_conv_algorithm) CUDNN_CHECK(cudnnStatus_t::CUDNN_STATUS_BAD_PARAM);

    // choose backward algorithm for filter
    // for better or worse, just a fixed constant due to the missing
    // cudnnGetConvolutionBackwardFilterAlgorithm in cuDNN version 8.0
    bwd_filter_algo_[i] = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    //twice the amount of the forward search to be save
    workspace_bwd_filter_sizes_[i] = 2 * workspace_fwd_sizes_[i];

    cudnnGetConvolutionBackwardDataAlgorithmMaxCount(Caffe::cudnn_handle(), &max_alg_count);
    bwd_data_algo_pref_.resize(max_alg_count);

    // cudnnGetConvolutionBackwardFilterAlgorithm is not available in cuDNN 8 anymore.
    // choose backward algo for data
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(Caffe::cudnn_handle(),
      filter_desc_,
      top_descs_[i],
      conv_descs_[i],
      bottom_descs_[i],
      max_alg_count,
      &ret_count,
      bwd_data_algo_pref_.data()));

    found_conv_algorithm = false;
    for(int n=0;n<ret_count;n++){
      if (bwd_data_algo_pref_[n].status == CUDNN_STATUS_SUCCESS &&
          bwd_data_algo_pref_[n].algo != CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD &&
          bwd_data_algo_pref_[n].algo != CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED &&
          bwd_data_algo_pref_[n].memory < workspace_limit_bytes){
        found_conv_algorithm = true;
        bwd_data_algo_[i]              = bwd_data_algo_pref_[n].algo;
        workspace_bwd_data_sizes_[i]   = bwd_data_algo_pref_[n].memory;
        break;
      }
    }
    if(!found_conv_algorithm) CUDNN_CHECK(cudnnStatus_t::CUDNN_STATUS_BAD_PARAM);

#else
    // choose forward and backward algorithms + workspace(s)
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(Caffe::cudnn_handle(),
      bottom_descs_[i],
      filter_desc_,
      conv_descs_[i],
      top_descs_[i],
      CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
      workspace_limit_bytes,
      &fwd_algo_[i]));

    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(Caffe::cudnn_handle(),
      bottom_descs_[i],
      filter_desc_,
      conv_descs_[i],
      top_descs_[i],
      fwd_algo_[i],
      &(workspace_fwd_sizes_[i])));

    // choose backward algorithm for filter
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(Caffe::cudnn_handle(),
          bottom_descs_[i], top_descs_[i], conv_descs_[i], filter_desc_,
          CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
          workspace_limit_bytes, &bwd_filter_algo_[i]) );

    // get workspace for backwards filter algorithm
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(Caffe::cudnn_handle(),
          bottom_descs_[i], top_descs_[i], conv_descs_[i], filter_desc_,
          bwd_filter_algo_[i], &workspace_bwd_filter_sizes_[i]));

    // choose backward algo for data
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(Caffe::cudnn_handle(),
          filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
          CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
        workspace_limit_bytes, &bwd_data_algo_[i]));

    // get workspace size
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(Caffe::cudnn_handle(),
          filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
          bwd_data_algo_[i], &workspace_bwd_data_sizes_[i]) );
#endif
  }

  // reduce over all workspace sizes to get a maximum to allocate / reallocate
  size_t total_workspace_fwd = 0;
  size_t total_workspace_bwd_data = 0;
  size_t total_workspace_bwd_filter = 0;

  for (size_t i = 0; i < bottom.size(); i++) {
    total_workspace_fwd        = std::max(total_workspace_fwd,
                                     workspace_fwd_sizes_[i]);
    total_workspace_bwd_data   = std::max(total_workspace_bwd_data,
                                     workspace_bwd_data_sizes_[i]);
    total_workspace_bwd_filter = std::max(total_workspace_bwd_filter,
                                     workspace_bwd_filter_sizes_[i]);
  }
  // get max over all operations
  size_t max_workspace = std::max(total_workspace_fwd,
                             total_workspace_bwd_data);
  max_workspace = std::max(max_workspace, total_workspace_bwd_filter);
  // ensure all groups have enough workspace
#if CUDNN_VERSION_MIN(7,0,0)
  size_t total_max_workspace = max_workspace;
#else
  size_t total_max_workspace = max_workspace *
                               (this->group_ * CUDNN_STREAMS_PER_GROUP);
#endif


  // this is the total amount of storage needed over all groups + streams
  if (total_max_workspace > workspaceSizeInBytes) {
    DLOG(INFO) << "Reallocating workspace storage: " << total_max_workspace;
    workspaceSizeInBytes = total_max_workspace;

    // free the existing workspace and allocate a new (larger) one
    cudaFree(this->workspaceData);

    cudaError_t err = cudaMalloc(&(this->workspaceData), workspaceSizeInBytes);
    if (err != cudaSuccess) {
      // force zero memory path
      for (int i = 0; i < bottom.size(); i++) {
        workspace_fwd_sizes_[i] = 0;
        workspace_bwd_filter_sizes_[i] = 0;
        workspace_bwd_data_sizes_[i] = 0;
        fwd_algo_[i] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        bwd_filter_algo_[i] = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        bwd_data_algo_[i] = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
      }

      // NULL out all workspace pointers
#if CUDNN_VERSION_MIN(7,0,0)
      workspace[0] = NULL;
#else
      for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
        workspace[g] = NULL;
      }
#endif
      // NULL out underlying data
      workspaceData = NULL;
      workspaceSizeInBytes = 0;
    }

#if CUDNN_VERSION_MIN(7,0,0)
    workspace[0] = reinterpret_cast<char *>(workspaceData);
#else
    // if we succeed in the allocation, set pointer aliases for workspaces
    for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
      workspace[g] = reinterpret_cast<char *>(workspaceData) + g*max_workspace;
    }
#endif
  }


  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::setTensor4dDesc<Dtype>(&bias_desc_,
        1, this->num_output_, 1, 1);
  }
}

template <typename Dtype>
CuDNNConvolutionLayer<Dtype>::~CuDNNConvolutionLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  for (int i = 0; i < bottom_descs_.size(); i++) {
    cudnnDestroyTensorDescriptor(bottom_descs_[i]);
    cudnnDestroyTensorDescriptor(top_descs_[i]);
    cudnnDestroyConvolutionDescriptor(conv_descs_[i]);
  }
  if (this->bias_term_) {
    cudnnDestroyTensorDescriptor(bias_desc_);
  }
  cudnnDestroyFilterDescriptor(filter_desc_);

#if !CUDNN_VERSION_MIN(7,0,0)
  cudaEventDestroy(start_event_); 
  for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
    cudaEventDestroy(end_event_[g]);
    cudaStreamDestroy(stream_[g]);
    cudnnDestroy(handle_[g]);
  }
#endif

  cudaFree(workspaceData);
  delete [] workspace;
#if !CUDNN_VERSION_MIN(7,0,0)
  delete [] end_event_;
  delete [] stream_;
  delete [] handle_;
#endif
  delete [] fwd_algo_;
  delete [] bwd_filter_algo_;
  delete [] bwd_data_algo_;
  delete [] workspace_fwd_sizes_;
  delete [] workspace_bwd_data_sizes_;
  delete [] workspace_bwd_filter_sizes_;
}

INSTANTIATE_CLASS(CuDNNConvolutionLayer);

}   // namespace caffe
#endif
