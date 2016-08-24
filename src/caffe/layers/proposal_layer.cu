
#include "caffe/layers/proposal_layer.hpp"
#include <stdio.h>
#include <boost/graph/graph_concepts.hpp>
namespace caffe
{ 

__device__ int bottom_offset(const int n, const int c,  const int h, const int w,
                                   const int channels, const int height, const int width)
{
  return ((n * channels + c) * height + h) * width + w;
}

template <typename Dtype>
__device__ Dtype d_min(const Dtype a, const Dtype b)
{
  return a < b ? a : b; 
}

template <typename Dtype>
__device__ Dtype d_max(const Dtype a, const Dtype b)
{
  return a > b ? a : b; 
}

template <typename Dtype>
__global__ void create_full_anchors(const int nthreads,
                                    const Dtype* base_anchors, const int num_anchors, const int width, const int feat_stride,
                                    Dtype* anchors)
{
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    int j = index%num_anchors;
    int i = index/num_anchors;
    
    anchors[index*4] = base_anchors[j*4+0]+(i%width)*feat_stride;
    anchors[index*4+1] = base_anchors[j*4+1]+(i/width)*feat_stride;
    anchors[index*4+2] = base_anchors[j*4+2]+(i%width)*feat_stride;
    anchors[index*4+3] = base_anchors[j*4+3]+(i/width)*feat_stride;
  }
}



template <typename Dtype>
__global__ void create_full_proposals(const int nthreads, 
                                      const Dtype* const bottom_data_1 ,const int channels_1, const int height_1, const int width_1 ,
                                      const Dtype* const bottom_data_0, const int channels_0, const int height_0, const int width_0 ,
                                      Dtype* proposals, Dtype* anchors, Dtype* scores, int* indexes,
                                      const Dtype* const im_info, const int num_anchors , const int min_size)
{
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    int img_w = im_info[1];
    int img_h = im_info[0];
    int minimal_size = min_size * im_info[2];
    int ch_1 = index % (channels_1/4);
    int w_1 = (index / (channels_1/4)) % width_1;
    int h_1 = (index / (channels_1/4)) / width_1;
    int ch_0 = index % (channels_0-num_anchors) ;
    int w_0 = (index / (channels_0-num_anchors)) % width_0;
    int h_0 = (index / (channels_0-num_anchors)) / width_0;
    
    double width = anchors[index*4+2] - anchors[index*4] + 1;
    double height = anchors[index*4+3] - anchors[index*4+1] + 1;;
    double ctr_x = anchors[index*4] + 0.5 * width;
    double ctr_y = anchors[index*4+1] + 0.5 * height;
            
    double pred_ctr_x = bottom_data_1[bottom_offset(0,ch_1*4, h_1, w_1, channels_1, height_1, width_1)] * width + ctr_x;
    double pred_ctr_y = bottom_data_1[bottom_offset(0,ch_1*4+1, h_1, w_1, channels_1, height_1, width_1)]* height + ctr_y;
    double pred_w = exp( bottom_data_1[bottom_offset(0,ch_1*4+2, h_1, w_1, channels_1, height_1, width_1)] ) * width;
    double pred_h = exp( bottom_data_1[bottom_offset(0,ch_1*4+3, h_1, w_1, channels_1, height_1, width_1)] ) * height;
    
    proposals[index*4] = d_max<Dtype>(d_min<Dtype>(pred_ctr_x - 0.5 * pred_w, img_w-1),0); 
    proposals[index*4+1] = d_max<Dtype>(d_min<Dtype>(pred_ctr_y - 0.5 * pred_h, img_h-1),0); 
    proposals[index*4+2] = d_max<Dtype>(d_min<Dtype>(pred_ctr_x + 0.5 * pred_w,img_w-1),0);
    proposals[index*4+3] = d_max<Dtype>(d_min<Dtype>( pred_ctr_y + 0.5 * pred_h,img_h-1),0);
    
    indexes[index] = index;
    scores[index] = 
    minimal_size <= (proposals[index*4+2]-proposals[index*4]+1) && minimal_size <= (proposals[index*4+3]-proposals[index*4+1]+1) ?
    bottom_data_0[bottom_offset(0,ch_0+num_anchors, h_0, w_0, channels_0, height_0, width_0 )] : -1;
    
  }
}

template <typename Dtype>
__global__ void bitonic_sort_base(const int nthreads, const int j, const int k,  int* indexes, const Dtype* scores)
{
  unsigned int i, ixj;
  int index_cache_1;
  int index_cache_2;
  CUDA_KERNEL_LOOP(index, nthreads)
  { 
    i = index;
    ixj = i^j;
    
    if ((ixj)>i) {
      index_cache_1 = indexes[i];
      index_cache_2 = indexes[ixj]; 
      
      if ((i&k)==0) {
        /* Sort ascending */        
        if (scores[index_cache_1]<scores[index_cache_2]) {
          /* exchange(i,ixj); */
          indexes[i] = index_cache_2;
          indexes[ixj] = index_cache_1;
        }
      }
      if ((i&k)!=0) {
        /* Sort descending */
        if (scores[index_cache_1]>scores[index_cache_2]) {
          /* exchange(i,ixj); */
          indexes[i] = index_cache_2;
          indexes[ixj] = index_cache_1;
        }
      }
    }
  }
}

template <typename Dtype>
__global__ void base_gpu_nms(const int nthreads, int* indexes, Dtype* scores, const Dtype* proposals, const Dtype threshold)
{
  int index_j;
  for(int j = 0; j < nthreads; ++j)
  {
    index_j = indexes[j];
    
    if(scores[index_j]>0)
    {
      Dtype x1_j = proposals[index_j*4];
      Dtype x2_j = proposals[index_j*4+2];
      Dtype y1_j = proposals[index_j*4+1];
      Dtype y2_j = proposals[index_j*4+3];
      
      Dtype area_j = x2_j-x1_j+1 * y2_j-y1_j+1;
      CUDA_KERNEL_LOOP(index, nthreads)
      {
        int index_i = indexes[index];
        if(index>j && scores[index_i]!=0)
        {
          Dtype x1_i = proposals[index_i*4];
          Dtype y1_i = proposals[index_i*4+1];
          Dtype x2_i = proposals[index_i*4+2];
          Dtype y2_i = proposals[index_i*4+3];
          Dtype area_i = x2_i-x1_i+1 * y2_i-y1_i+1;
          
          Dtype  x1_inter = d_max<Dtype>(x1_j, x1_i);
          Dtype  y1_inter = d_max<Dtype>(y1_j, y1_i);
          Dtype  x2_inter = d_min<Dtype>(x2_j, x2_i);
          Dtype  y2_inter = d_min<Dtype>(y2_j, y2_i);
          
          Dtype inter = d_max<Dtype>(0.0, x2_inter - y1_inter + 1) * d_max<Dtype>(0.0, y2_inter - y1_inter + 1);
          if(threshold >= inter / (area_i + area_j - inter))
            scores[index_i]=-1;
        }
      }         
    }
    __syncthreads();
  }
}

template <typename Dtype>
__global__ void data_to_top(const int nthreads, const int* indexes, const Dtype* proposals, Dtype* top_data)
{
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    int ind = indexes[index];
    top_data[index*5]=0;
    top_data[index*5+1] = proposals[ind*4];
    top_data[index*5+2] = proposals[ind*4+1];
    top_data[index*5+3] = proposals[ind*4+2];
    top_data[index*5+4] = proposals[ind*4+3];
  }
}

template <typename Dtype>
__device__ Dtype IOU(const Dtype x1_i, const Dtype y1_i, const Dtype x2_i, const Dtype y2_i, 
                     const Dtype x1_j, const Dtype y1_j, const Dtype x2_j, const Dtype y2_j )
{
    Dtype area_j = x2_j-x1_j+1 * y2_j-y1_j+1;

    Dtype area_i = x2_i-x1_i+1 * y2_i-y1_i+1;

    Dtype  x1_inter = d_max<Dtype>(x1_j, x1_i);
    Dtype  y1_inter = d_max<Dtype>(y1_j, y1_i);
    Dtype  x2_inter = d_min<Dtype>(x2_j, x2_i);
    Dtype  y2_inter = d_min<Dtype>(y2_j, y2_i);
    Dtype inter = d_max<Dtype>(0.0, x2_inter - y1_inter + 1) * d_max<Dtype>(0.0, y2_inter - y1_inter + 1);
    
    return inter / (area_i + area_j - inter);
}



template <typename Dtype>
__global__ void calc_iou_matrix(const int nthreads, const int dmax,  const int* indexes, const Dtype* proposals, Dtype* scores, int* ioumat, const Dtype threshold )
{
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    int i = index / dmax;
    int j = index % dmax;
    
    int i_ind = indexes[i];
    int j_ind = indexes[j];
    
    if( scores[i_ind]<scores[j_ind] )
    {
      Dtype x1_j = proposals[j_ind*4];
      Dtype x2_j = proposals[j_ind*4+2];
      Dtype y1_j = proposals[j_ind*4+1];
      Dtype y2_j = proposals[j_ind*4+3];      
      Dtype area_j = x2_j-x1_j+1 * y2_j-y1_j+1;
      
      Dtype x1_i = proposals[i_ind*4];
      Dtype y1_i = proposals[i_ind*4+1];
      Dtype x2_i = proposals[i_ind*4+2];
      Dtype y2_i = proposals[i_ind*4+3];
      Dtype area_i = x2_i-x1_i+1 * y2_i-y1_i+1;
      
      Dtype  x1_inter = d_max<Dtype>(x1_j, x1_i);
      Dtype  y1_inter = d_max<Dtype>(y1_j, y1_i);
      Dtype  x2_inter = d_min<Dtype>(x2_j, x2_i);
      Dtype  y2_inter = d_min<Dtype>(y2_j, y2_i);
      Dtype inter = d_max<Dtype>(0.0, x2_inter - y1_inter + 1) * d_max<Dtype>(0.0, y2_inter - y1_inter + 1);
      
      ioumat[index] = threshold >= inter / (area_i + area_j - inter)? 1 : 0;

    }
  }

}



template <typename Dtype>
__global__ void nms_improve(const int indexes_count, const int* indexes, const Dtype* proposals, Dtype* scores, const Dtype threshold)
{
  __shared__ Dtype cache_x[4*16];
  __shared__ Dtype cache_y[4*16];
 
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = (blockIdx.y * blockDim.y) + threadIdx.y;
  if ( i < indexes_count && j < indexes_count )
  { 
    int j_ind= indexes[i];
    int i_ind= indexes[j];
    if(threadIdx.x == threadIdx.y)
    {
      cache_x[threadIdx.x*4] = proposals[i_ind*4];
      cache_x[threadIdx.x*4+1] = proposals[i_ind*4+1];
      cache_x[threadIdx.x*4+2] = proposals[i_ind*4+2];
      cache_x[threadIdx.x*4+3] = proposals[i_ind*4+3];
      
      cache_y[threadIdx.y*4] = proposals[j_ind*4];
      cache_y[threadIdx.y*4+1] = proposals[j_ind*4+1];
      cache_y[threadIdx.y*4+2] = proposals[j_ind*4+2];
      cache_y[threadIdx.y*4+3] = proposals[j_ind*4+3];
    } 
  }

   __syncthreads();
  if ( i < indexes_count && j < indexes_count )
  {
    Dtype iou = IOU<Dtype>(cache_x[threadIdx.x*4],cache_x[threadIdx.x*4+1],cache_x[threadIdx.x*4+2],cache_x[threadIdx.x*4+3],
                        cache_y[threadIdx.y*4],cache_y[threadIdx.y*4+1],cache_y[threadIdx.y*4+2],cache_y[threadIdx.y*4+3]);
    if(threshold >= iou )
    {
      int j_ind= indexes[i];
      int i_ind= indexes[j];
      scores[i_ind]<scores[j_ind] ? scores[i_ind] : scores[j_ind] = -1; 
      
    }
  } 
}


  
template <typename Dtype>
void ProposalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  if(bottom.size()==4)
  {
    m_pre_nms_topN = bottom[3]->cpu_data()[bottom[3]->offset(0,0)];
    m_post_nms_topN = bottom[3]->cpu_data()[bottom[3]->offset(0,1)];
    m_nms_thresh =bottom[3]->cpu_data()[bottom[3]->offset(0,2)];
    m_min_size = bottom[3]->cpu_data()[bottom[3]->offset(0,3)];
  }
  
  int count = m_anchors.count()/4;
//   {
//     float time;
//     cudaEvent_t start, stop;
// 
//    cudaEventCreate(&start) ;
//    cudaEventCreate(&stop) ;
//    cudaEventRecord(start, 0);
  
    create_full_anchors<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
    m_base_anchors.gpu_data(), m_num_anchors, bottom[0]->shape(3), m_feat_stride,
    m_anchors.mutable_gpu_data());
    
//     cudaEventRecord(stop, 0);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&time, start, stop);
// 
//     printf("Time to create_full_anchors:  %3.10f ms \n", time);
//   }


  
  

  
  const Dtype* bottom_1 = bottom[1]->gpu_data();
  const Dtype* bottom_0 = bottom[0]->gpu_data();
  
  Dtype* proposals = m_proposals.mutable_gpu_data(); 
  Dtype* scores = m_scores.mutable_gpu_data();
  Dtype* anchors = m_anchors.mutable_gpu_data();
  int* indexes = m_indexes.mutable_gpu_data();
  int* ioumat = m_iou.mutable_gpu_data();
  
//   {
//   float time;
//   cudaEvent_t start, stop;
// 
//   cudaEventCreate(&start) ;
//   cudaEventCreate(&stop) ;
//   cudaEventRecord(start, 0) ;
  
  create_full_proposals<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
    bottom_1, bottom[1]->shape(1), bottom[1]->shape(2), bottom[1]->shape(3),
    bottom_0, bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
    proposals, anchors, scores, indexes,
    bottom[2]->gpu_data(), m_num_anchors, m_min_size);
  
//     cudaEventRecord(stop, 0) ;
//     cudaEventSynchronize(stop) ;
//     cudaEventElapsedTime(&time, start, stop) ;
// 
//     printf("Time to create_full_proposals:  %3.10f ms \n", time);
//   }  
    
  CUDA_POST_KERNEL_CHECK;
  
//     {
//   float time;
//   cudaEvent_t start, stop;
// 
//    cudaEventCreate(&start) ;
//    cudaEventCreate(&stop) ;
//    cudaEventRecord(start, 0) ;
  int j, k;
  for (k = 2; k <= count; k <<= 1) {
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_base<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, j, k, indexes, scores);
    }
  }
//     cudaEventRecord(stop, 0) ;
//     cudaEventSynchronize(stop) ;
//     cudaEventElapsedTime(&time, start, stop) ;
// 
//     printf("Time to sort1:  %3.10f ms \n", time);
//   }  
  CUDA_POST_KERNEL_CHECK;
  
/*  
  {
  float time;
  cudaEvent_t start, stop;

   cudaEventCreate(&start) ;
   cudaEventCreate(&stop) ;
   cudaEventRecord(start, 0) ;
  
  base_gpu_nms<Dtype><<<CAFFE_GET_BLOCKS(m_pre_nms_topN), CAFFE_CUDA_NUM_THREADS>>>(m_pre_nms_topN, indexes, scores, proposals, m_nms_thresh);
  CUDA_POST_KERNEL_CHECK;
    cudaEventRecord(stop, 0) ;
    cudaEventSynchronize(stop) ;
    cudaEventElapsedTime(&time, start, stop) ;

    printf("Time to nms:  %3.1f ms \n", time);
  }  
 */
//   {
//     float time;
//     cudaEvent_t start, stop;
// 
//     cudaEventCreate(&start) ;
//     cudaEventCreate(&stop) ;
//     cudaEventRecord(start, 0) ;
//     calc_iou_matrix<Dtype><<<CAFFE_GET_BLOCKS(m_pre_nms_topN * m_pre_nms_topN), CAFFE_CUDA_NUM_THREADS>>>(m_pre_nms_topN*m_pre_nms_topN, m_pre_nms_topN, indexes, proposals, scores, ioumat ,m_nms_thresh);
//     CUDA_POST_KERNEL_CHECK;
//     cudaEventRecord(stop, 0) ;
//     cudaEventSynchronize(stop) ;
//     cudaEventElapsedTime(&time, start, stop) ;
// 
//     printf("Time to nms:  %3.10f ms \n", time);
//   }  
// 
//   
//   {
//     float time;
//     cudaEvent_t start, stop;
// 
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start, 0);
    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((m_pre_nms_topN +15) /threadsPerBlock.x,
                   (m_pre_nms_topN +15) /threadsPerBlock.y);  
    
    nms_improve<Dtype><<<numBlocks, threadsPerBlock>>>(m_pre_nms_topN, indexes, proposals, scores, m_nms_thresh);
    
    
    CUDA_POST_KERNEL_CHECK;
//     cudaEventRecord(stop, 0) ;
//     cudaEventSynchronize(stop) ;
//     cudaEventElapsedTime(&time, start, stop) ;
// 
//     printf("Time to nms:  %3.10f ms \n", time);
//   }  
  
  
  
  
  
//       {
//   float time;
//   cudaEvent_t start, stop;
// 
//   cudaEventCreate(&start) ;
//   cudaEventCreate(&stop) ;
//   cudaEventRecord(start, 0) ;
//   int j, k;
  for (k = 2; k <= m_pre_nms_topN; k <<= 1) {
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_base<<<CAFFE_GET_BLOCKS(m_pre_nms_topN), CAFFE_CUDA_NUM_THREADS>>>(m_pre_nms_topN, j, k, indexes, scores);
    }
  }
//     cudaEventRecord(stop, 0) ;
//     cudaEventSynchronize(stop) ;
//    cudaEventElapsedTime(&time, start, stop) ;
// 
//     printf("Time to sort2:  %3.10f ms \n", time);
//   }  
  
//   {
//   float time;
//   cudaEvent_t start, stop;
// 
//    cudaEventCreate(&start) ;
//    cudaEventCreate(&stop) ;
//    cudaEventRecord(start, 0);
//   
  data_to_top<Dtype><<<CAFFE_GET_BLOCKS(m_post_nms_topN), CAFFE_CUDA_NUM_THREADS>>>(m_post_nms_topN, indexes, proposals, top[0]->mutable_gpu_data() );
  CUDA_POST_KERNEL_CHECK;
//     cudaEventRecord(stop, 0) ;
//     cudaEventSynchronize(stop) ;
//     cudaEventElapsedTime(&time, start, stop);
// 
//     printf("Time to copy to top:  %3.10f ms \n", time);
//   }  
//   
  
  //TODO write scores to output blob
}


template <typename Dtype>
void ProposalLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
   return;
}

INSTANTIATE_LAYER_GPU_FUNCS(ProposalLayer);
}