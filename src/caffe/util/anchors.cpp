#include <math.h>

#include "caffe/util/anchors.hpp"

#include "glog/logging.h"



namespace caffe{

template <typename Dtype>  
std::vector< std::vector< Dtype > > generate_anchors( const std::vector< Dtype >& scales, const std::vector< Dtype >& ratios, const int& base_size )
{
  std::vector< std::vector< Dtype > > anchors;
  std::vector< Dtype > base_anchor({0, 0, (Dtype)(base_size-1), (Dtype)(base_size-1)});
  std::vector< std::vector< Dtype > > ratio_anchors = ratio_enum(base_anchor, ratios);
  
  for( typename std::vector< std::vector < Dtype > >::const_iterator it= ratio_anchors.begin(); it!=ratio_anchors.end(); ++it)
  {
    std::vector< std::vector< Dtype > > scale_acnhors = scale_enum((*it),scales);
    anchors.insert(anchors.end(), scale_acnhors.begin() ,scale_acnhors.end());
  }
  return anchors;
}

template std::vector< std::vector< float > > generate_anchors <float>( const std::vector< float >& scales, const std::vector< float >& ratios, const int& base_size );
template std::vector< std::vector< double > > generate_anchors <double>( const std::vector< double >& scales, const std::vector< double >& ratios, const int& base_size );
 

template <typename Dtype>
std::vector< std::vector< Dtype > > ratio_enum( const std::vector< Dtype >& anchor, const std::vector< Dtype >& ratios )
{
  Dtype w;
  Dtype h;
  Dtype x_ctr;
  Dtype y_ctr;
  
  whctrs(anchor,w,h,x_ctr,y_ctr);
  Dtype size = w * h;
  std::vector< Dtype > ws;
  std::vector< Dtype > hs;
 
  for( typename std::vector< Dtype >::const_iterator it = ratios.begin(); it!= ratios.end(); ++it)
  {  
    Dtype value = round( sqrt( size / (*it) ) );
    ws.push_back( value );
    hs.push_back( round( value * (*it) ));
  }
  
  return mkanchors(ws, hs, x_ctr, y_ctr );
}
template std::vector< std::vector< float > > ratio_enum <float> ( const std::vector< float  >& anchor, const std::vector< float >& ratios );
template std::vector< std::vector< double > > ratio_enum <double> ( const std::vector< double >& anchor, const std::vector< double >& ratios );




template <typename Dtype>
std::vector< std::vector< Dtype > > scale_enum( const std::vector< Dtype >& anchor, const std::vector<Dtype>& scales)
{
  Dtype w;
  Dtype h;
  Dtype x_ctr;
  Dtype y_ctr;
  whctrs(anchor,w,h,x_ctr,y_ctr);
  
  std::vector< Dtype > ws;
  std::vector< Dtype > hs;
  
  for(typename std::vector<Dtype>::const_iterator it=scales.begin(); it!=scales.end(); ++it)
  {
    ws.push_back(w*(Dtype)(*it));
    hs.push_back(h*(Dtype)(*it));
  }
  return mkanchors(ws, hs, x_ctr, y_ctr);
}
template std::vector< std::vector< float > > scale_enum<float>( const std::vector< float >& anchor, const std::vector<float>& scales);
template std::vector< std::vector< double > > scale_enum<double>( const std::vector< double >& anchor, const std::vector<double>& scales);




template <typename Dtype>
void whctrs( const std::vector<Dtype>& anchor, Dtype& w, Dtype& h, Dtype& x_ctr, Dtype& y_ctr )
{
  w = anchor[2] - anchor[0] + 1;
  h = anchor[3] - anchor[1] + 1;
  x_ctr = anchor[0] + 0.5 * (w - 1);
  y_ctr = anchor[1] + 0.5 * (h - 1);
}

template void whctrs<float>( const std::vector<float>& anchor, float& w, float& h, float& x_ctr, float& y_ctr );
template void whctrs<double>( const std::vector<double>& anchor, double& w, double& h, double& x_ctr, double& y_ctr );


template <typename Dtype>
std::vector< std::vector< Dtype > > mkanchors(const std::vector<Dtype>& ws, const std::vector<Dtype>& hs,
                              const Dtype&  x_ctr, const Dtype& y_ctr)
{
  //CHECK(ws.size() == hs.size());
  std::vector< std::vector< Dtype > > anchors;
  
  for(size_t i=0; i < ws.size(); ++i)
  {
    anchors.push_back( std::vector< Dtype >( { x_ctr - (Dtype)0.5 * (ws[i] - (Dtype)1),
                                               y_ctr - (Dtype)0.5 * (hs[i] - (Dtype)1),
                                               x_ctr + (Dtype)0.5 * (ws[i] - (Dtype)1),
                                               y_ctr + (Dtype)0.5 * (hs[i] - (Dtype)1) } ) );
  }
  return anchors;
}

template std::vector< std::vector< float > > mkanchors<float>(const std::vector<float>& ws, const std::vector<float>& hs,
                              const float&  x_ctr, const float& y_ctr);
template std::vector< std::vector< double > > mkanchors<double>(const std::vector<double>& ws, const std::vector<double>& hs,
                              const double&  x_ctr, const double& y_ctr);


} // caffe