#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_conv_layer.hpp"

namespace caffe {

//__global__ void sync_conv_groups() { }

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();

#if CUDNN_VERSION_MIN(7,0,0)
    CUDNN_CHECK(cudnnConvolutionForward(Caffe::cudnn_handle(),
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], bottom_data,
            filter_desc_, weight,
            conv_descs_[i],
            fwd_algo_[i], workspace[0], workspace_fwd_sizes_[i],
            cudnn::dataType<Dtype>::zero,
            top_descs_[i], top_data));
    // Bias.
    if (this->bias_term_) {
      const Dtype* bias_data = this->blobs_[1]->gpu_data();
      CUDNN_CHECK(cudnnAddTensor(Caffe::cudnn_handle(),
            cudnn::dataType<Dtype>::one,
            bias_desc_, bias_data,
            cudnn::dataType<Dtype>::one,
            top_descs_[i], top_data));
    }
#else
    cudaEventRecord(start_event_, Caffe::cuda_stream());

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      cudaStream_t stream;
      cudnnGetStream(handle_[g], &stream);
      cudaStreamWaitEvent(stream,start_event_,0);
      // Filters.
      CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], bottom_data + bottom_offset_ * g,
            filter_desc_, weight + this->weight_offset_ * g,
            conv_descs_[i],
            fwd_algo_[i], workspace[g], workspace_fwd_sizes_[i],
            cudnn::dataType<Dtype>::zero,
            top_descs_[i], top_data + top_offset_ * g));

      // Bias.
      if (this->bias_term_) {
        const Dtype* bias_data = this->blobs_[1]->gpu_data();
        CUDNN_CHECK(cudnnAddTensor(handle_[g],
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_data + top_offset_ * g));
      }
      cudaEventRecord(end_event_[g], stream);
      cudaStreamWaitEvent(Caffe::cuda_stream(),end_event_[g],0);
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    //BAD HABIT
    //sync_conv_groups<<<1, 1>>>();
#endif
  }
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();

#if CUDNN_VERSION_MIN(7,0,0)
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      CUDNN_CHECK(cudnnConvolutionBackwardBias(Caffe::cudnn_handle(),
            cudnn::dataType<Dtype>::one,
            top_descs_[i],  top_diff,
            cudnn::dataType<Dtype>::one,
            bias_desc_, bias_diff));
    }

    // Gradient w.r.t. weights.
    if (this->param_propagate_down_[0]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      CUDNN_CHECK(cudnnConvolutionBackwardFilter(
            Caffe::cudnn_handle(),
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], bottom_data,
            top_descs_[i],    top_diff,
            conv_descs_[i],
            bwd_filter_algo_[i], workspace[0],
            workspace_bwd_filter_sizes_[i],
            cudnn::dataType<Dtype>::one,
            filter_desc_, weight_diff ));
    }

    // Gradient w.r.t. bottom data.
    if (propagate_down[i]) {
      if (weight == NULL) {
        weight = this->blobs_[0]->gpu_data();
      }
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      CUDNN_CHECK(cudnnConvolutionBackwardData(
            Caffe::cudnn_handle(),
            cudnn::dataType<Dtype>::one,
            filter_desc_, weight,
            top_descs_[i], top_diff,
            conv_descs_[i],
            bwd_data_algo_[i], workspace[0],
            workspace_bwd_data_sizes_[i],
            cudnn::dataType<Dtype>::zero,
            bottom_descs_[i], bottom_diff));
    }
#else
    cudaEventRecord(start_event_,Caffe::cuda_stream());
    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        const int& handle_id = 0*this->group_ + g;
        cudaStream_t stream;
        cudnnGetStream(handle_[handle_id], &stream);
        cudaStreamWaitEvent(stream,start_event_,0);
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[handle_id],
              cudnn::dataType<Dtype>::one,
              top_descs_[i],  top_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_diff + bias_offset_ * g));
        cudaEventRecord(end_event_[handle_id],stream);
        cudaStreamWaitEvent(Caffe::cuda_stream(),end_event_[handle_id],0);
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        const int& handle_id = 1*this->group_ + g;
        cudaStream_t stream;
        cudnnGetStream(handle_[handle_id], &stream);
        cudaStreamWaitEvent(stream,start_event_,0);
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(
              handle_[handle_id],
              cudnn::dataType<Dtype>::one,
              bottom_descs_[i], bottom_data + bottom_offset_ * g,
              top_descs_[i],    top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_filter_algo_[i], workspace[handle_id],
              workspace_bwd_filter_sizes_[i],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight_diff + this->weight_offset_ * g));
        cudaEventRecord(end_event_[handle_id],stream);
        cudaStreamWaitEvent(Caffe::cuda_stream(),end_event_[handle_id],0);
      }

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (weight == NULL) {
          weight = this->blobs_[0]->gpu_data();
        }
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        const int& handle_id = 2*this->group_ + g;
        cudaStream_t stream;
        cudnnGetStream(handle_[handle_id], &stream);
        cudaStreamWaitEvent(stream,start_event_,0);
        CUDNN_CHECK(cudnnConvolutionBackwardData(
              handle_[handle_id],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight + this->weight_offset_ * g,
              top_descs_[i], top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_data_algo_[i], workspace[handle_id],
              workspace_bwd_data_sizes_[i],
              cudnn::dataType<Dtype>::zero,
              bottom_descs_[i], bottom_diff + bottom_offset_ * g));
        cudaEventRecord(end_event_[handle_id],stream);
        cudaStreamWaitEvent(Caffe::cuda_stream(),end_event_[handle_id],0);
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    //BAD HABIT
    //sync_conv_groups<<<1, 1>>>();
#endif
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNConvolutionLayer);

}  // namespace caffe
#endif
