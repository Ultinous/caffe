//#include "caffe/ultinous/anchor_target_layer.hpp"
//#include "caffe/util/gpu_util.cuh"
//#include "../../../../../../../../usr/include/c++/5/fstream"
//
//#include <thrust/partition.h>
//#include <thrust/sort.h>
//#include <thrust/device_vector.h>
//#include <thrust/iterator/constant_iterator.h>
//
//namespace caffe
//{
//namespace ultinous
//{
//
//template <typename Dtype>
//struct is_equal
//{
//  int a;
//  thrust::device_ptr<Dtype> data;
//  is_equal(int _a, thrust::device_ptr<Dtype> _data) : a(_a), data(_data) {}
//  __host__ __device__
//  bool operator()(const int& x) const
//  { return a == data[x]; }
//};
//
//template <typename Dtype>
//struct is_less : public thrust::binary_function<int,int,bool>
//{
//  is_less(int number_of_base_anchors_, thrust::device_ptr<Dtype> scores_)
//      : number_of_base_anchors(number_of_base_anchors_), scores(scores_) {}
//  __host__ __device__
//  bool operator()(const int & a, const int & b )
//  {
//    int place_a = a%number_of_base_anchors;
//    int place_b = b%number_of_base_anchors;
//    if(place_a < place_b)
//      return true;
//    else if (place_a > place_b)
//      return false;
//    else
//      return scores[a]>scores[b];
//  }
//  int number_of_base_anchors;
//  thrust::device_ptr<Dtype> scores;
//};
//
//struct is_key_equal
//{
//  is_key_equal(int number_of_base_anchors_) : number_of_base_anchors(number_of_base_anchors_){}
//
//  int  number_of_base_anchors;
//  __host__ __device__
//  bool operator() (const int&a, const int& b)
//  {
//    int place_a = a%number_of_base_anchors;
//    int place_b = b%number_of_base_anchors;
//    return place_a == place_b;
//  }
//
//};
//
//template <typename  Dtype>
//struct is_key_stay_in
//{
//  Dtype barrier;
//  thrust::device_ptr<Dtype> rnd;
//  is_key_stay_in(Dtype barrier_, thrust::device_ptr<Dtype> rnd_):barrier(barrier_),rnd(rnd_){}
//  __host__ __device__
//  bool operator() (int a)
//  {
//    return barrier > rnd[a];
//  }
//};
//
//
//
//
//template <typename  Dtype>
//__global__
//void shift_kernel(const int n_threads, const int feat_stride, const int base_anchor_num
//    , const int blob_h, const int blob_w, const int allowed_border, const int img_w, const int img_h
//    , Dtype* const base_anchors, Dtype* anchors, int* anchors_validation)
//{
//  CUDA_KERNEL_LOOP(index, n_threads)
//  {
////    int ba_ind = index % base_anchor_num;
////    int w_ind = (index / base_anchor_num) % blob_w;
////    int h_ind = (index / base_anchor_num) / blob_w;
//    int w_ind = (index % blob_w );
//    int h_ind = (index / blob_w ) % blob_h;
//    int ba_ind = (index / blob_w) / blob_h;
//
//    Dtype prop_x1 = base_anchors[ba_ind * 4 + 0] + w_ind * feat_stride;
//    Dtype prop_y1 = base_anchors[ba_ind * 4 + 1] + h_ind * feat_stride;
//    Dtype prop_x2 = base_anchors[ba_ind * 4 + 2] + w_ind * feat_stride;
//    Dtype prop_y2 = base_anchors[ba_ind * 4 + 3] + h_ind * feat_stride;
//
//    anchors_validation[index] = prop_x1 >= -allowed_border && prop_y1 >= -allowed_border
//             && prop_x2 < img_w + allowed_border && prop_y2 < img_w + allowed_border;
//
//    anchors[index * 4 + 0] = prop_x1;
//    anchors[index * 4 + 1] = prop_y1;
//    anchors[index * 4 + 2] = prop_x2;
//    anchors[index * 4 + 3] = prop_y2;
//
//  }
//}
//
//template <typename  Dtype>
//__global__
//void overlap_kernel(const int n_threads, const int bbox_num,
//                    Dtype *const anchors, Dtype * const bboxes, Dtype* overlaps)
//{
//  CUDA_KERNEL_LOOP(index, n_threads)
//  {
//    for(int j = 0 ; j < bbox_num; ++j)
//      overlaps[index * bbox_num + j] = dev_IoU(anchors+(index*4), bboxes+ (j*5));
//  }
//}
//
//template <typename Dtype>
//__global__
//void maximum_kernel_over_rows(const int rows, const int cols , const Dtype* overlaps, int * result)
//{
//  CUDA_KERNEL_LOOP(index, rows)
//  {
//      int max_idx =  0;
//      Dtype max_value = overlaps[ index * cols + 0 ];
//      for(int j =  1; j< cols; ++j )
//      {
//        int idx =  index * cols + j;
//        if(max_value > overlaps[idx])
//        {
//          max_idx = j;
//          max_value = overlaps[idx];
//        }
//      }
//      result[index] =  max_idx;
//  }
//}
//
//template <typename Dtype>
//__global__
//void maximum_kernel_over_cols(const int rows, const int cols , const Dtype* overlaps, int * result, const int * validation)
//{
//  CUDA_KERNEL_LOOP(index, cols)
//  {
//    int max_idx =  -1;
//    Dtype max_value;
//    for(int j =  0; j< rows; ++j )
//    {
//      int idx =  index + cols * j;
//      if(validation[j] && max_idx == -1 )
//      {
//        max_idx = j;
//        max_value = overlaps[idx];
//      }
//      else if(validation[j] && max_value > overlaps[idx])
//      {
//        max_idx = j;
//        max_value = overlaps[idx];
//      }
//    }
//    result[index] =  max_idx;
//  }
//}
//
//template <typename Dtype>
//void maximu_kernel(const int &rows, const int &cols ,const Dtype* overlaps, int * result, const int *validation, bool over_cols = false)
//{
//  if(over_cols)
//    maximum_kernel_over_cols<Dtype><<<CAFFE_GET_BLOCKS(cols),CAFFE_CUDA_NUM_THREADS>>>(rows,cols,overlaps,result,validation);
//  else
//    maximum_kernel_over_rows<Dtype><<<CAFFE_GET_BLOCKS(rows),CAFFE_CUDA_NUM_THREADS>>>(rows,cols,overlaps,result);
//}
//
//
//template <typename  Dtype>
//__global__
//void negative_label_kernel(const int rows,const int cols, Dtype min_overlap, const Dtype * overlaps, const int* anchors_validation, const int* max_overlaps_ids, Dtype* labels  )
//{
//  CUDA_KERNEL_LOOP(index, rows)
//  {
//    if(anchors_validation[index])
//    {
//      int gt_index = max_overlaps_ids[index];
//      if(overlaps[index*cols+gt_index] <= min_overlap)
//      {
//        labels[index] = 0;
//      }
//    }
//  }
//}
//
//
//template <typename Dtype>
//__global__
//void max_assign_kernel(const int nthreads,const int * max_for_gt, Dtype * labels )
//{
//  CUDA_KERNEL_LOOP(index, nthreads)
//  {
//    int idx = max_for_gt[index];
//    if(idx >= 0 )
//    {
//      labels[idx] = 1;
//    }
//  }
//}
//
//
//template <typename  Dtype>
//__global__
//void positive_label_kernel(const int rows,const int cols, Dtype max_overlap, const Dtype * overlaps, const int* anchors_validation, const int* max_overlaps_ids, Dtype* labels  )
//{
//  CUDA_KERNEL_LOOP(index, rows)
//  {
//    if(anchors_validation[index])
//    {
//      int gt_index = max_overlaps_ids[index];
//      if(overlaps[index*cols+gt_index] >= max_overlap)
//      {
//        labels[index] = 1;
//      }
//    }
//  }
//}
//
//
//template <typename Dtype>
//__global__
//void set_through_hash(const int nthreads, int* hash, Dtype* target, const Dtype val )
//{
//  CUDA_KERNEL_LOOP(index, nthreads)
//  {
//    int idx = hash[index];
//    target[idx] = val;
//  }
//}
//
//
//template <typename Dtype>
//__global__
//void get_scores_for_anchors(const int nthreads,
///*int blob_w, int blob_h, int base_anchor_num,*/
//int shift, const Dtype * data, Dtype* scores)
//{
//  CUDA_KERNEL_LOOP(index, nthreads)
//  {
////    int ba_ind = index % base_anchor_num;
////    int w_ind = (index / base_anchor_num) % blob_w;
////    int h_ind = (index / base_anchor_num) / blob_w;
////    int pos_ind = ((base_anchor_num + ba_ind) * blob_h + h_ind) * blob_w + w_ind;
////    int neg_ind = ((base_anchor_num + ba_ind) * blob_h + h_ind) * blob_w + w_ind; == index
////    Simplify
////    pos_ind == aHW + AHW + hW + w and neg_ind == aHW + hW + w
////    pos_ind == AHW + index  and neg_ind == index
////
////
//    Dtype pos_data = data[shift + index];
//    Dtype neg_data = data[index];
//    scores[index] =  exp(pos_data)/(exp(pos_data)+exp(neg_data));
//  }
//
//}
//
//
////template <typename Dtype>
////__device__
////void compute_bbox_target(Dtype* anchor, Dtype* gt, Dtype* target)
////{
////  float ex_width = anchor[2] - anchor[0] + 1.0;
////  float ex_height = anchor[3] - anchor[1] + 1.0;
////  float ex_ctr_x = anchor[0] + 0.5 * ex_width;
////  float ex_ctr_y = anchor[1] + 0.5 * ex_height;
////
////  float gt_width = gt[2] - gt[0] + 1.0;
////  float gt_height = gt[3] - gt[1] + 1.0;
////  float gt_ctr_x = gt[0] + 0.5 * gt_width;
////  float gt_ctr_y = gt[1] + 0.5 * gt_height;
////
////  target[0] = (gt_ctr_x - ex_ctr_x) / ex_width;
////  target[1] = (gt_ctr_y - ex_ctr_y) / ex_height;
////  target[2] = std::log(gt_width / ex_width);
////  target[3] = std::log(gt_height / ex_height);
////}
//
//
//template <typename Dtype>
//__global__
//void compute_bbox_targets(const int nthreads, const int WH ,Dtype * anchors, const int* validation, Dtype* gt_boxes, const int * anchor_gt_maximal_overlap_idx_, Dtype* targets)
//{
//  CUDA_KERNEL_LOOP(index, nthreads)
//  {
//    if(validation[index])
//    {
//      Dtype* anchor = anchors + index*4;
//      Dtype* gt = gt_boxes + anchor_gt_maximal_overlap_idx_[index]*5;
////      Dtype* target =  targets + index*4;
////      compute_bbox_target(anchor,gt,target);
//
//      Dtype ex_width = anchor[2] - anchor[0] + 1.0;
//      Dtype ex_height = anchor[3] - anchor[1] + 1.0;
//      Dtype ex_ctr_x = anchor[0] + 0.5 * ex_width;
//      Dtype ex_ctr_y = anchor[1] + 0.5 * ex_height;
//
//      Dtype gt_width = gt[2] - gt[0] + 1.0;
//      Dtype gt_height = gt[3] - gt[1] + 1.0;
//      Dtype gt_ctr_x = gt[0] + 0.5 * gt_width;
//      Dtype gt_ctr_y = gt[1] + 0.5 * gt_height;
//
//      targets[index + 0] = (gt_ctr_x - ex_ctr_x) / ex_width;
//      targets[index + 1] = (gt_ctr_y - ex_ctr_y) / ex_height;
//      targets[index + 2] = std::log(gt_width / ex_width);
//      targets[index + 3] = std::log(gt_height / ex_height);
//    }
//  }
//}
//
//template <typename Dtype>
//__global__
//void compute_inside_weights(const int nthreads, Dtype* labels, Dtype* RPN_BBOX_INSIDE_WEIGHTS, Dtype* inside_weights)
//{
//  CUDA_KERNEL_LOOP(index, nthreads)
//  {
//    if(labels[index]==1)
//    {
//      inside_weights[index*4] = RPN_BBOX_INSIDE_WEIGHTS[0];
//      inside_weights[index*4+1] = RPN_BBOX_INSIDE_WEIGHTS[0];
//      inside_weights[index*4+2] = RPN_BBOX_INSIDE_WEIGHTS[0];
//      inside_weights[index*4+3] = RPN_BBOX_INSIDE_WEIGHTS[0];
//    }
//  }
//
//}
//
//template <typename Dtype>
//__global__
//void compute_outside_weights(const int nthreads, const Dtype * labels, Dtype const pos_w, const Dtype neg_w, Dtype* outside_weights)
//{
//  CUDA_KERNEL_LOOP(index, nthreads)
//  {
//    Dtype label =  labels[index];
//    if(label == 1 || label == 0)
//    {
//      Dtype w = label == 1 ? pos_w : neg_w;
//      outside_weights[index*4] = w;
//      outside_weights[index*4+1] = w;
//      outside_weights[index*4+2] = w;
//      outside_weights[index*4+3] = w;
//    }
//  }
//}
//
//template <typename Dtype>
//void AnchorTargetLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
//{
//  int image_num = bottom[0]->shape(0);
//  int height = bottom[0]->shape(2);
//  int width = bottom[0]->shape(3);
//
//  // labels
//  top[0]->Reshape(image_num, 1, base_anchors_size_ * height, width);
//  caffe_gpu_set(top[0]->count(),Dtype(-1),top[0]->mutable_gpu_data());
//  // bbox_targets
//  top[1]->Reshape(image_num, base_anchors_size_ * 4, height, width);
//  caffe_gpu_set(top[1]->count(),Dtype(0),top[1]->mutable_gpu_data());
//    // bbox_inside_weights
//  top[2]->Reshape(image_num, base_anchors_size_ * 4, height, width);
//  caffe_gpu_set(top[2]->count(),Dtype(0),top[2]->mutable_gpu_data());
//  // bbox_outside_weights
//  top[3]->Reshape(image_num, base_anchors_size_ * 4, height, width);
//  caffe_gpu_set(top[3]->count(),Dtype(0),top[3]->mutable_gpu_data());
//
//  int anchors_num_on_img = base_anchors_size_*height*width;
//  int bbox_shift = 0;
//  anchors_.Reshape(anchors_num_on_img, 4, 1, 1 );
//  anchors_validation_.Reshape(anchors_num_on_img, 1, 1 ,1);
//  anchors_scores_.Reshape(anchors_num_on_img, 1, 1 ,1);
//
//  random_seq_.Reshape(anchors_num_on_img, 1, 1 ,1);
//
//  for(int i=0; i<random_seq_.shape(0);++i)
//  {
//    random_seq_.mutable_cpu_data()[i]=i;
//  }
//  std::random_shuffle(random_seq_.mutable_cpu_data(),random_seq_.mutable_cpu_data()+anchors_num_on_img);
//
//  random_temp_.Reshape(anchors_num_on_img, 1, 1 ,1);
//
//  int* random_seq = random_seq_.mutable_gpu_data();
//
//  Dtype* base_anchors = base_anchors_.mutable_gpu_data();
//  Dtype* anchors = anchors_.mutable_gpu_data();
//  int* anchors_validation = anchors_validation_.mutable_gpu_data();
//
//  float RPN_POSITIVE_OVERLAP = anchorTargetParam_.positive_overlap();
//  float RPN_NEGATIVE_OVERLAP = anchorTargetParam_.negative_overlap();
//  bool RPN_CLOBBER_POSITIVES = anchorTargetParam_.clobber_positives();
//  float RPN_FG_FRACTION = anchorTargetParam_.fg_fraction();
//  int RPN_BATCHSIZE = anchorTargetParam_.batchsize();
//  Dtype RPN_POSITIVE_WEIGHT = anchorTargetParam_.positive_weight();
//
//  for(int batch_index = 0; batch_index < image_num; ++batch_index)
//  {
//    Dtype* labels = top[0]->mutable_gpu_data() + top[0]->offset(batch_index);
//    Dtype* bbox_targets = top[1]->mutable_gpu_data() + top[1]->offset(batch_index);
//    Dtype* bbox_inside_weights = top[1]->mutable_gpu_data() + top[1]->offset(batch_index);
//    Dtype* bbox_outside_weights = top[1]->mutable_gpu_data() + top[1]->offset(batch_index);
//
//    Dtype img_h = (bottom[2]->mutable_cpu_data() + bottom[2]->offset(batch_index))[0];
//    Dtype img_w = (bottom[2]->mutable_cpu_data() + bottom[2]->offset(batch_index))[1];
//
//    Dtype img_bbox_num =  (bottom[2]->mutable_cpu_data() + bottom[2]->offset(batch_index))[3];
//    Dtype* gt_boxes =  bottom[0]->mutable_gpu_data() + bbox_shift;
//    bbox_shift += img_bbox_num;
//
//    shift_kernel<Dtype><<<CAFFE_GET_BLOCKS(anchors_num_on_img), CAFFE_CUDA_NUM_THREADS>>>(
//        anchors_num_on_img, feat_stride_, base_anchors_size_,
//        height, width, allowed_border_, img_w, img_h,
//        base_anchors, anchors, anchors_validation
//    );
//    CUDA_POST_KERNEL_CHECK;
//
//    for(int i =  0; i< anchors_.shape(0); ++i)
//      std::cout<< anchors_.cpu_data()[i*4+0]<<" "<<anchors_.cpu_data()[i*4+1]<<" "<<anchors_.cpu_data()[i*4+2]<<" "<<anchors_.cpu_data()[i*4+3]<<" "<<std::endl;
//    std::cout<<"__________"<<std::endl;
//
//    anchor_overlaps_.Reshape(anchors_num_on_img,img_bbox_num,1,1);
//
//    overlap_kernel<Dtype><<<CAFFE_GET_BLOCKS(anchors_num_on_img),CAFFE_CUDA_NUM_THREADS>>>(
//        anchors_num_on_img, img_bbox_num, anchors, gt_boxes,  anchor_overlaps_.mutable_gpu_data()
//    );
//    CUDA_POST_KERNEL_CHECK;
//
//    anchor_gt_maximal_overlap_idx_.Reshape(anchors_num_on_img,1,1,1);
//    gt_anchor_maximal_overlap_idx_.Reshape(img_bbox_num,1,1,1);
//
//    maximu_kernel<Dtype>(anchor_overlaps_.shape(0),anchor_overlaps_.shape(1),anchor_overlaps_.gpu_data(),anchor_gt_maximal_overlap_idx_.mutable_gpu_data(), anchors_validation);
//
//    maximu_kernel<Dtype>(anchor_overlaps_.shape(0),anchor_overlaps_.shape(1),anchor_overlaps_.gpu_data(),gt_anchor_maximal_overlap_idx_.mutable_gpu_data(), anchors_validation,true);
//
//    if(!RPN_CLOBBER_POSITIVES)
//    {
//      negative_label_kernel<Dtype><<<CAFFE_GET_BLOCKS(anchors_num_on_img),CAFFE_CUDA_NUM_THREADS>>>(
//          anchors_num_on_img, img_bbox_num, RPN_NEGATIVE_OVERLAP,
//          anchor_overlaps_.mutable_gpu_data(), anchors_validation, anchor_gt_maximal_overlap_idx_.mutable_gpu_data(), labels
//      );
//      CUDA_POST_KERNEL_CHECK;
//    }
//    max_assign_kernel<Dtype><<<CAFFE_GET_BLOCKS(img_bbox_num),CAFFE_CUDA_NUM_THREADS>>>(
//      img_bbox_num, gt_anchor_maximal_overlap_idx_.mutable_gpu_data(), labels
//    );
//    CUDA_POST_KERNEL_CHECK;
//
//    positive_label_kernel<Dtype><<<CAFFE_GET_BLOCKS(anchors_num_on_img),CAFFE_CUDA_NUM_THREADS>>>(
//        anchors_num_on_img, img_bbox_num, RPN_POSITIVE_OVERLAP,
//            anchor_overlaps_.mutable_gpu_data(), anchors_validation, anchor_gt_maximal_overlap_idx_.mutable_gpu_data(), labels
//    );
//    CUDA_POST_KERNEL_CHECK;
//
//    if(RPN_CLOBBER_POSITIVES)
//    {
//      negative_label_kernel<Dtype><<<CAFFE_GET_BLOCKS(anchors_num_on_img),CAFFE_CUDA_NUM_THREADS>>>(
//          anchors_num_on_img, img_bbox_num, RPN_NEGATIVE_OVERLAP,
//              anchor_overlaps_.mutable_gpu_data(), anchors_validation, anchor_gt_maximal_overlap_idx_.mutable_gpu_data(), labels
//      );
//      CUDA_POST_KERNEL_CHECK;
//    }
//
//    thrust::device_ptr<int> random_seq_t = thrust::device_pointer_cast<int>(random_seq);
//    thrust::device_ptr<Dtype> labels_t = thrust::device_pointer_cast<Dtype>(labels);
//
//    //std::cout<<"Thrust1"<<std::endl;
//    int num_positives = thrust::stable_partition(random_seq_t,random_seq_t+anchors_num_on_img,is_equal<Dtype>(1,labels_t)) - random_seq_t;
//    int num_fg = RPN_FG_FRACTION * RPN_BATCHSIZE;
//    if (RPN_BATCHSIZE > 0 && num_positives > num_fg) {
//      int number_to_set = num_positives - num_fg;
//      set_through_hash<Dtype><<<CAFFE_GET_BLOCKS(number_to_set),CAFFE_CUDA_NUM_THREADS>>>(number_to_set,random_seq+num_fg,labels,-1.0);
//    } else
//      num_fg = num_positives;
//
//    //std::cout<<"Thrust2"<<std::endl;
//    int num_negatives = thrust::stable_partition(random_seq_t,random_seq_t+anchors_num_on_img,is_equal<Dtype>(0, labels_t)) - random_seq_t;
//    int num_bg = (RPN_BATCHSIZE > 0)
//                 ? (RPN_BATCHSIZE - num_fg)
//                 : static_cast<int>( std::ceil(static_cast<float>(num_fg) * (1.0f - RPN_FG_FRACTION) / RPN_FG_FRACTION));
//
//    // subsample negative labels if we have too many,
//    //Dtype const *scores = bottom[0]->cpu_data();
//
//    if(RPN_BATCHSIZE > 0 && num_negatives > num_bg)
//    {
//      if (anchorTargetParam_.hard_negative_mining() == 0.0f) {
//        int number_to_set = num_negatives - num_bg;
//        set_through_hash<Dtype> <<< CAFFE_GET_BLOCKS(number_to_set), CAFFE_CUDA_NUM_THREADS >> >
//                                                                     (number_to_set, random_seq+num_bg, labels, -1.0);
//      }
//      else {
//        int num_bg_per_baseAnchor = static_cast<int>(
//            std::ceil(static_cast<float>(num_bg)
//                      / base_anchors_size_)
//        );
//        //Do a softmax to calculate anchor scores
//        Dtype *anchors_scores = anchors_scores_.mutable_gpu_data();
//        get_scores_for_anchors<Dtype> <<< CAFFE_GET_BLOCKS(anchors_num_on_img), CAFFE_CUDA_NUM_THREADS >>>
//            (anchors_num_on_img, anchors_num_on_img, bottom[0]->gpu_data(), anchors_scores);
//        CUDA_POST_KERNEL_CHECK;
//
//        //sort by base_anchors and scores as second the full negative example list
//        //std::cout<<"Thrust3"<<std::endl;
//
//        thrust::device_ptr<Dtype> anchors_scores_t = thrust::device_pointer_cast<Dtype> (anchors_scores);
//
//        thrust::sort(random_seq_t, random_seq_t + num_negatives, is_less<Dtype>(base_anchors_size_, anchors_scores_t));
//
//        thrust::device_vector<int> numbers(base_anchors_size_);
//
//        thrust::device_vector<int> keys(base_anchors_size_);
//
//        //negative example histogram per baseanchor
//        //std::cout<<"Thrust4"<<std::endl;
//        thrust::reduce_by_key(random_seq_t, random_seq_t + num_negatives, thrust::constant_iterator<int>(1), keys.begin(),
//                              numbers.begin(), is_key_equal(base_anchors_size_));
//        thrust::host_vector<int> numbers_cpu(numbers);
//
//        //if not fully hard negative mining den randomize negative for every base anchor
//        //std::cout<<"Thrust5"<<std::endl;
//        if (anchorTargetParam_.hard_negative_mining() != 1.0f) {
//          caffe_gpu_rng_uniform<Dtype>(anchors_num_on_img, 0.0f, 1.0f, random_temp_.mutable_gpu_data());
//          thrust::device_ptr<int> pointer_steper_t = random_seq_t;
//          for (thrust::detail::vector_base< int, std::allocator<int> >::iterator it = numbers_cpu.begin(); it!= numbers_cpu.begin(); ++it ) {
//            const int& bg_anchor_num =  *it;
//            if (bg_anchor_num > num_bg_per_baseAnchor)
//                thrust::partition(pointer_steper_t,pointer_steper_t + bg_anchor_num,
//                                is_key_stay_in<Dtype>(anchorTargetParam_.hard_negative_mining(),
//                                                      thrust::device_pointer_cast<Dtype> (random_temp_.mutable_gpu_data())));
//            pointer_steper_t += bg_anchor_num;
//          }
//        }
//
//        num_bg = 0;
//        int *pointer_steper = random_seq;
//        for (thrust::detail::vector_base< int, std::allocator<int> >::iterator it = numbers_cpu.begin(); it!= numbers_cpu.begin(); ++it ) {
//          const int& bg_anchor_num =  *it;
//          if (bg_anchor_num > num_bg_per_baseAnchor) {
//            int elim_size = bg_anchor_num - num_bg_per_baseAnchor;
//            set_through_hash<Dtype> <<< CAFFE_GET_BLOCKS(elim_size), CAFFE_CUDA_NUM_THREADS >>>(elim_size,pointer_steper,labels, -1 );
//            num_bg+=num_bg_per_baseAnchor;
//          }
//          else
//            num_bg+= bg_anchor_num;
//          pointer_steper += bg_anchor_num;
//        }
//
//      }
//    }
//    else
//    {
//      num_bg = num_negatives;
//    }
//
//
//
//    compute_bbox_targets<Dtype><<<CAFFE_GET_BLOCKS(anchors_num_on_img), CAFFE_CUDA_NUM_THREADS>>>(anchors_num_on_img, height*width ,anchors,anchors_validation,gt_boxes,anchor_gt_maximal_overlap_idx_.mutable_gpu_data(),top[1]->mutable_gpu_data());
//    CUDA_POST_KERNEL_CHECK;
//
//    //debugblob(*top[1]);
//    compute_inside_weights<Dtype><<<CAFFE_GET_BLOCKS(anchors_num_on_img), CAFFE_CUDA_NUM_THREADS>>>(anchors_num_on_img,labels,rpn_bbox_inside_weights_.mutable_gpu_data(),top[2]->mutable_gpu_data());
//    CUDA_POST_KERNEL_CHECK;
//
//    std::cout<<"Num_bg "<< num_bg<<std::endl;
//    std::cout<<"Num_fg "<< num_fg<<std::endl;
//
//    int num_examples = num_fg + num_bg;
//    Dtype positive_weight;
//    Dtype negative_weight;
//    if (RPN_POSITIVE_WEIGHT < 0) {
//      positive_weight = 1.0 / num_examples;
//      negative_weight = 1.0 / num_examples;
//    } else {
//      CHECK((RPN_POSITIVE_WEIGHT > 0) && (RPN_POSITIVE_WEIGHT < 1));
//      positive_weight = (RPN_POSITIVE_WEIGHT / num_fg);
//      negative_weight = ((1.0 - RPN_POSITIVE_WEIGHT) / num_bg);
//    }
//
//    compute_outside_weights<Dtype><<<CAFFE_GET_BLOCKS(anchors_num_on_img), CAFFE_CUDA_NUM_THREADS>>>(anchors_num_on_img,labels,positive_weight,negative_weight,top[2]->mutable_gpu_data());
//    CUDA_POST_KERNEL_CHECK;
//  }
//}
//
//template <typename Dtype>
//void AnchorTargetLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
//const vector<Blob<Dtype> *> &bottom)
//{}
//
//
//
//
//INSTANTIATE_LAYER_GPU_FUNCS(AnchorTargetLayer);
//
//}//ultinous
//}//caffe
