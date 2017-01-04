#ifndef CAFFE_UTIL_ANCHORS_HPP_
#define CAFFE_UTIL_ANCHORS_HPP_

#include <vector> 

/* Anchor is an std::vector<Dtype> with x1,y1,x2,y1 cordinates. 
 * It always holds 4 cord.
 **/


namespace caffe{
/* 
 * Generate  Anchor for every scale and ratio for reference window (0,0,@base_size-1,@base_size-1)
 **/ 
template <typename Dtype>
std::vector< std::vector< Dtype > > generate_anchors(
                               const std::vector<Dtype>& scales = std::vector<Dtype>({8, 16, 32}), 
                               const std::vector<Dtype>& ratios = std::vector<Dtype>({0.5, 1, 2}),
                               const int& base_size = 16 );

template <typename Dtype>
std::vector< std::vector< Dtype > > ratio_enum(const std::vector< Dtype >& anchor, const std::vector<Dtype>& ratios);

template <typename Dtype>
std::vector< std::vector< Dtype > > scale_enum(const std::vector< Dtype >& anchor, const std::vector<Dtype>& ratios);

template <typename Dtype>
void whctrs(const std::vector< Dtype >& anchor, Dtype& w, Dtype& h, Dtype& x_ctr, Dtype& y_ctr);  

template <typename Dtype>
std::vector< std::vector<Dtype> > mkanchors(const std::vector<Dtype>& ws, const std::vector<Dtype>& hs,
                              const Dtype&  x_ctr, const Dtype& y_ctr);

}//caffe

#endif