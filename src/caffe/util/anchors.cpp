#include "caffe/util/anchors.hpp"

#include <math.h>

#include "glog/logging.h"



namespace caffe{
std::vector<Anchor> generate_anchors( const std::vector< int >& scales, const std::vector< double >& ratios, const int& base_size )
{
  std::vector<Anchor> anchors;
  Anchor base_anchor({0, 0, (double)(base_size-1), (double)(base_size-1)});
  std::vector<Anchor> ratio_anchors = ratio_enum(base_anchor, ratios);
  
  for(std::vector<Anchor>::const_iterator it=ratio_anchors.begin(); it!=ratio_anchors.end(); ++it)
  {
    std::vector<Anchor> scale_acnhors = scale_enum((*it),scales);
    anchors.insert(anchors.end(), scale_acnhors.begin() ,scale_acnhors.end());
  }
  return anchors;
}

std::vector<Anchor> ratio_enum( const Anchor& anchor, const std::vector< double >& ratios )
{
  double w;
  double h;
  double x_ctr;
  double y_ctr;
  
  whctrs(anchor,w,h,x_ctr,y_ctr);
  double size = w * h;
  std::vector<double> ws;
  std::vector<double> hs;
 
  for(std::vector<double>::const_iterator it = ratios.begin(); it!= ratios.end(); ++it)
  {  
    double value = round( sqrt( size / (*it) ) );
    ws.push_back( value );
    hs.push_back( round( value * (*it) ));
  }
  
  return mkanchors(ws, hs, x_ctr, y_ctr );
}


std::vector<Anchor> scale_enum( const Anchor& anchor, const std::vector<int>& scales)
{
  double w;
  double h;
  double x_ctr;
  double y_ctr;
  whctrs(anchor,w,h,x_ctr,y_ctr);
  
  std::vector<double> ws;
  std::vector<double> hs;
  
  for(std::vector<int>::const_iterator it=scales.begin(); it!=scales.end(); ++it)
  {
    ws.push_back(w*(double)(*it));
    hs.push_back(h*(double)(*it));
  }
  return mkanchors(ws, hs, x_ctr, y_ctr);
}


void whctrs( const Anchor& anchor, double& w, double& h, double& x_ctr, double& y_ctr )
{
    w = anchor[2] - anchor[0] + 1;
    h = anchor[3] - anchor[1] + 1;
    x_ctr = anchor[0] + 0.5 * (w - 1);
    y_ctr = anchor[1] + 0.5 * (h - 1);
}


std::vector<Anchor> mkanchors(const std::vector<double>& ws, const std::vector<double>& hs,
                              const double&  x_ctr, const double& y_ctr)
{
  //CHECK(ws.size() == hs.size());
  std::vector<Anchor> anchors;
  
  for(size_t i=0; i < ws.size(); ++i)
  {
      anchors.push_back( Anchor( { x_ctr - 0.5 * (ws[i] - 1),
                                   y_ctr - 0.5 * (hs[i] - 1),
                                   x_ctr + 0.5 * (ws[i] - 1),
                                   y_ctr + 0.5 * (hs[i] - 1) } ) );
  }
  return anchors;
}

} // caffe