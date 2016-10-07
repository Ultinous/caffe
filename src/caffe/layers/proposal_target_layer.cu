#include "caffe/layers/proposal_target_layer.hpp"
#include <math.h>
#include <boost/concept_check.hpp>
#include <algorithm>

namespace caffe
{

template <typename Dtype>
__device__ Dtype IOU(const Dtype x1_i, const Dtype y1_i, const Dtype x2_i, const Dtype y2_i, 
                     const Dtype x1_j, const Dtype y1_j, const Dtype x2_j, const Dtype y2_j )
{
    
    Dtype area_j = (x2_j-x1_j+1) * (y2_j-y1_j+1);

    Dtype area_i = (x2_i-x1_i+1) * (y2_i-y1_i+1);

    Dtype  x1_inter = max(x1_j, x1_i);
    Dtype  y1_inter = max(y1_j, y1_i);
    Dtype  x2_inter = min(x2_j, x2_i);
    Dtype  y2_inter = min(y2_j, y2_i);
    Dtype inter = max(0.0, x2_inter - x1_inter + 1) * max(0.0, y2_inter - y1_inter + 1);
    
    return inter / (area_i + area_j - inter);
}





template <typename Dtype>
__global__ void overlaps_maximum_kernel(const int nthreads,
                                        const Dtype* all_rois, const int num_all_rois,
                                        const Dtype* gt_boxes, const int num_gt_boxes,
                                        Dtype* max_overlaps)
{
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    if(index<num_all_rois)
    {
      max_overlaps[index*2]=0;
      max_overlaps[index*2+1]= IOU<Dtype>(all_rois[index*5+1],all_rois[index*5+2],all_rois[index*5+3],all_rois[index*5+4],
                                          gt_boxes[0], gt_boxes[1], gt_boxes[2], gt_boxes[3]);
      for(int i=1; i<num_gt_boxes; ++i)
      {
          Dtype iou= IOU<Dtype>(all_rois[index*5+1],all_rois[index*5+2],all_rois[index*5+3],all_rois[index*5+4],
                                gt_boxes[i*5], gt_boxes[i*5+1], gt_boxes[i*5+2], gt_boxes[i*5+3]);
          max_overlaps[index*2]= max_overlaps[index*2+1] < iou ? i : max_overlaps[index*2];
          max_overlaps[index*2+1]=max_overlaps[index*2+1] < iou ? iou : max_overlaps[index*2+1];
      }
    }
    else
    {
      int new_index= index-num_all_rois;
      max_overlaps[index*2]=new_index;
      max_overlaps[index*2+1]=1;
    }
  }
}




template <typename Dtype>
__global__ void prop_target_kernel(const int nthreads, const int* indexes, 
                                   const Dtype* all_rois, const int num_all_rois,
                                   const Dtype* gt_boxes, const int num_gt_boxes,
                                   const Dtype* max_overlaps, const int fg_rois_per_this_image,
                                   Dtype* labels, Dtype* rois, const bool normalize, const int num_class,
                                   const Dtype mean_x1, const Dtype mean_y1, const Dtype mean_x2, const Dtype mean_y2,
                                   const Dtype std_x1, const Dtype std_y1, const Dtype std_x2, const Dtype std_y2,
                                   const Dtype bbiw_x1, const Dtype bbiw_y1, const Dtype bbiw_x2, const Dtype bbiw_y2,
                                   Dtype* target, Dtype* bbox_inside_weights, Dtype* bbox_outside_weights
                                  )
{
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    int i = indexes[index];
    labels[index] = index < fg_rois_per_this_image ? gt_boxes[static_cast<int>(max_overlaps[i*2])*5+4] : 0.0;
    rois[index * 5 ] = 0;
    rois[index * 5 + 1 ] = i < num_all_rois ? all_rois[i*5 + 1] : gt_boxes[(i-num_all_rois)*5]; 
    rois[index * 5 + 2 ] = i < num_all_rois ? all_rois[i*5 + 2] : gt_boxes[(i-num_all_rois)*5 + 1];
    rois[index * 5 + 3 ] = i < num_all_rois ? all_rois[i*5 + 3] : gt_boxes[(i-num_all_rois)*5 + 2];
    rois[index * 5 + 4 ] = i < num_all_rois ? all_rois[i*5 + 4] : gt_boxes[(i-num_all_rois)*5 + 3];
    __syncthreads();
   //Transform bboxes, this is a code duplication a proposal layer and maybe in anchor targets.
   //Handle gpu kernel code duplactions
   //moving this into a function maybe cause bad influence on the performence.
    Dtype ex_width = rois[index * 5 + 3 ] - rois[index * 5 + 1 ] + 1.0;
    Dtype ex_height = rois[index * 5 + 4 ] - rois[index * 5 + 2 ] + 1.0;
    Dtype ex_ctr_x = rois[index * 5 + 1 ] + 0.5 * ex_width;
    Dtype ex_ctr_y = rois[index * 5 + 2 ] + 0.5 * ex_height;

    Dtype gt_width = gt_boxes[static_cast<int>(max_overlaps[i*2])*5+2] - gt_boxes[static_cast<int>(max_overlaps[i*2])*5] + 1.0;
    Dtype gt_height = gt_boxes[static_cast<int>(max_overlaps[i*2])*5+3] - gt_boxes[static_cast<int>(max_overlaps[i*2])*5+1] + 1.0;
    Dtype gt_ctr_x = gt_boxes[static_cast<int>(max_overlaps[i*2])*5] + 0.5 * gt_width;
    Dtype gt_ctr_y = gt_boxes[static_cast<int>(max_overlaps[i*2])*5+1] + 0.5 * gt_height;

    Dtype target_data_label = labels [index];
    Dtype target_data_x = (((gt_ctr_x-ex_ctr_x)/ex_width) - (normalize ? mean_x1 : 0)) / (normalize ? std_x1 : 1.0);
    Dtype target_data_y = (((gt_ctr_y-ex_ctr_y)/ex_height) - (normalize ? mean_y1 : 0)) / (normalize ? std_y1 : 1.0);
    Dtype target_data_w = ((log(gt_width/ex_width)) - (normalize ? mean_x2 : 0)) / (normalize ? std_x2 : 1.0);
    Dtype target_data_h = ((log(gt_height/ex_height)) - (normalize ? mean_y2 : 0)) / (normalize ? std_y2 : 1.0);
    
    for(int j=0; j<num_class; ++j)
    {
     int temp = (index * 4 * num_class) + j*4;
     target[temp]= target_data_label!=0 && target_data_label==j ? target_data_x : 0;
     target[temp+1]= target_data_label!=0 && target_data_label==j ? target_data_y : 0;
     target[temp+2]= target_data_label!=0 && target_data_label==j ? target_data_w : 0;
     target[temp+3]= target_data_label!=0 && target_data_label==j ? target_data_h : 0;
     
     
     bbox_inside_weights[temp] = target_data_label!=0 && target_data_label==j ? bbiw_x1 : 0;
     bbox_inside_weights[temp+1] = target_data_label!=0 && target_data_label==j ? bbiw_y1 : 0;
     bbox_inside_weights[temp+2] = target_data_label!=0 && target_data_label==j? bbiw_x2 : 0;
     bbox_inside_weights[temp+3] = target_data_label!=0 && target_data_label==j ? bbiw_y2 : 0;
     
     bbox_outside_weights[temp] = target_data_label!=0 && target_data_label==j ? 1 : 0;
     bbox_outside_weights[temp+1] = target_data_label!=0 && target_data_label==j ? 1 : 0;
     bbox_outside_weights[temp+2] = target_data_label!=0 && target_data_label==j ? 1 : 0;
     bbox_outside_weights[temp+3] = target_data_label!=0 && target_data_label==j ? 1 : 0;  
    }
    
  }
}


  
template <typename Dtype>
void ProposalTargetLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
//   std::cout<<"////////////////////////DEBUG///////////////////////"<<std::endl;
  const Dtype* all_rois = bottom[0]->gpu_data();
  const Dtype* gt_boxes = bottom[1]->gpu_data();
  Dtype* max_overlaps = m_maximum_temp.mutable_gpu_data();
  
  
  int num_images = 1;
  int rois_per_image = m_minibatch_size/num_images;
  int fg_rois_per_image = round(m_fg_fraction*(Dtype)rois_per_image);
  /*
  std::cout<<"rois_per_image: "<<rois_per_image <<std::endl;
  std::cout<<"fg_rois_per_image"<< fg_rois_per_image<<std::endl;
  
  std::cout<<"all_rois_shape(0): " <<bottom[0]->shape(0)<<std::endl;
  std::cout<<"all_rois_shape(1): " <<bottom[0]->shape(1)<<std::endl;
  
  std::cout<<"gt_boxes_shape(0): " <<bottom[1]->shape(0)<<std::endl;
  std::cout<<"gt_boxes_shape(1): " <<bottom[1]->shape(1)<<std::endl;
  
  std::cout<<"m_maximum_temp_shape" << m_maximum_temp.shape(0) <<","<< m_maximum_temp.shape(1)<<std::endl;
  */
  overlaps_maximum_kernel<Dtype><<<CAFFE_GET_BLOCKS(m_maximum_temp.shape(0)), CAFFE_CUDA_NUM_THREADS>>>(                                                                                                    
    m_maximum_temp.shape(0),
    all_rois, bottom[0]->shape(0),
    gt_boxes, bottom[1]->shape(0), max_overlaps);
  CUDA_POST_KERNEL_CHECK;

  std::vector<int> fg_inds;
  std::vector<int> bg_inds;
//   std::cout<<"m_maximum_temp:"<<std::endl;
//   std::cout<<"["<<std::endl;
  for(int i=0; i<m_maximum_temp.shape(0); ++i)
  { 
    if(m_maximum_temp.mutable_cpu_data()[i*2+1]>=m_fg_treshold)
    {
      fg_inds.push_back(i);
    }
    else if(m_maximum_temp.mutable_cpu_data()[i*2+1]<m_bg_treshold_hi &&
            m_maximum_temp.mutable_cpu_data()[i*2+1]>=m_bg_treshold_lo)
    {
      bg_inds.push_back(i);
    }
  }
//   std::cout<<std::endl<<"]"<<std::endl;
//   
//   std::cout<<"fg_inds.size(): "<<fg_inds.size()<<std::endl;
//   std::cout<<"bg_inds.size(): "<<bg_inds.size()<<std::endl;
  
  
  int fg_rois_per_this_image = std::min<int>(fg_rois_per_image, fg_inds.size());
  int bg_rois_per_this_image = std::min<int>(rois_per_image - fg_rois_per_this_image, bg_inds.size());

  
  
  std::random_shuffle(fg_inds.begin(),fg_inds.end());
  std::random_shuffle(bg_inds.begin(),bg_inds.end());
  
  bg_inds.resize(bg_rois_per_this_image);
  fg_inds.resize(fg_rois_per_this_image);
 
  
  
  std::vector<int> shape;
  
//   std::cout<<"this image fg and bg: "<<fg_rois_per_this_image<<",  "<<bg_rois_per_this_image<<std::endl;
  
  shape.push_back(fg_rois_per_this_image+bg_rois_per_this_image);
  shape.push_back(1);
  
  m_keep_inds.Reshape(shape);

  for(std::size_t i=0; i<fg_rois_per_this_image; ++i )
    m_keep_inds.mutable_cpu_data()[i]=fg_inds[i];
  for(std::size_t i=0; i<bg_rois_per_this_image; ++i )
    m_keep_inds.mutable_cpu_data()[fg_rois_per_this_image+i] = bg_inds[i];
  
  
//   for(int index=0; index<m_keep_inds.shape(0);++index)
//   {
//     bool k = index < fg_rois_per_this_image;
//     int i = m_keep_inds.cpu_data()[index];
//     std::cout<<(k ? "fg: ": "bg: " )<< i << "("<<m_maximum_temp.cpu_data()[i*2]<<" -->"<<( k ? bottom[1]->cpu_data()[ (int)m_maximum_temp.cpu_data()[i*2]*5+4]: 0 )<<")"<<std::endl;
//   }
  

  shape[1]=5;
  top[0]->Reshape(shape);
  Dtype* rois = top[0]->mutable_gpu_data();
  shape[1]=1;
  top[1]->Reshape(shape);
  Dtype* labels = top[1]->mutable_gpu_data();
  shape[1]=4*m_num_classes;
  top[2]->Reshape(shape);
  Dtype* targets = top[2]->mutable_gpu_data();
  top[3]->Reshape(shape);
  Dtype* bbox_inside_weights = top[3]->mutable_gpu_data();
  top[4]->Reshape(shape);
  Dtype* bbox_outside_weights = top[4]->mutable_gpu_data();
  
  
   prop_target_kernel<Dtype><<<CAFFE_GET_BLOCKS(m_keep_inds.shape(0)), CAFFE_CUDA_NUM_THREADS >>>(m_keep_inds.shape(0), m_keep_inds.gpu_data(), 
                                                                                                  all_rois, bottom[0]->shape(0),
                                                                                                  gt_boxes, bottom[1]->shape(0),
                                                                                                  max_overlaps, fg_rois_per_this_image,
                                                                                                  labels, rois, m_bbox_normalize_targets_precomputed, m_num_classes,
                                                                                                  m_bbox_normalize_means[0], m_bbox_normalize_means[1], m_bbox_normalize_means[2], m_bbox_normalize_means[3],
                                                                                                  m_bbox_normalize_stds[0], m_bbox_normalize_stds[1], m_bbox_normalize_stds[2], m_bbox_normalize_stds[3],
                                                                                                  m_bbox_inside_weights[0], m_bbox_inside_weights[1], m_bbox_inside_weights[2], m_bbox_inside_weights[3],
                                                                                                  targets, bbox_inside_weights, bbox_outside_weights);
   CUDA_POST_KERNEL_CHECK;
//    std::cout<<"inside weights:"<<m_bbox_inside_weights[0]<<","<<m_bbox_inside_weights[1]<<","<<m_bbox_inside_weights[2]<<","<<m_bbox_inside_weights[3]<<","<<std::endl;
//    std::cout <<"rois: ["<<std::endl;
//    for(int i=0; i<top[0]->shape(0); ++i)
//    {
//       std::cout<<"(" <<top[0]->cpu_data()[i*5]<<", "<<top[0]->cpu_data()[i*5+1]<<", "<<top[0]->cpu_data()[i*5+2]<<", "<<top[0]->cpu_data()[i*5+3]<<", "<<top[0]->cpu_data()[i*5+4]<<" ), ";
//    }
//    std::cout<<std::endl<<"]"<<std::endl;
//    
//    
//    std::cout <<"labels: ["<<std::endl;
//    for(int i=0; i<top[0]->shape(0); ++i)
//    {
//       std::cout<<"(" <<top[1]->cpu_data()[i]<<" ), ";
//    }
//    std::cout<<std::endl<<"]"<<std::endl;
//    
//     std::cout <<"targets: ["<<std::endl;
//    for(int i=0; i<top[3]->shape(0); ++i )
//    {
//      for(int j=0; j<top[3]->shape(1); ++j)
//      {
//         std::cout<<top[3]->cpu_data()[(i*4*m_num_classes)+j]<<" ,";
//      }
//      std::cout<<std::endl;
//    }
//    
}


template <typename Dtype>
void ProposalTargetLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
   return;
}

INSTANTIATE_LAYER_GPU_FUNCS(ProposalTargetLayer);
}