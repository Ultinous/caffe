#ifndef CAFFE_UTIL_ANCHORS_HPP_
#define CAFFE_UTIL_ANCHORS_HPP_

#include <vector> 

namespace caffe{
typedef std::vector<double> Anchor;

std::vector<Anchor> generate_anchors(
                               const std::vector<int>& scales = std::vector<int>({8, 16, 32}), 
                               const std::vector<double>& ratios = std::vector<double>({0.5, 1, 2}),
                               const int& base_size = 16 );

std::vector<Anchor> ratio_enum(const Anchor& anchor, const std::vector<double>& ratios);

std::vector<Anchor> scale_enum(const Anchor& anchor, const std::vector<int>& ratios);

void whctrs(const Anchor& anchor, double& w, double& h, double& x_ctr, double& y_ctr);  

std::vector<Anchor> mkanchors(const std::vector<double>& ws, const std::vector<double>& hs,
                              const double&  x_ctr, const double& y_ctr);

}//caffe

#endif