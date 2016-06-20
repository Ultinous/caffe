#pragma once

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


namespace caffe {
namespace ultinous {

class UltinousTransformer
{
public:
  UltinousTransformer( UltinousTransformationParameter const& params, Phase phase )
    : m_params( params )
    , m_phase( phase )
  {
  }

  void transform( cv::Mat& cv_img )
  {
    if( m_phase != TRAIN ) return;

    // Apply color transformation
    if( m_params.luminanceminscale() != 0.0f
      || m_params.luminancemaxscale() != 0.0f
      || m_params.saturationminscale() != 0.0f
      || m_params.saturationmaxscale() != 0.0f )
    {
      float lumCoef = m_params.luminanceminscale()
        + (m_params.luminancemaxscale()-m_params.luminanceminscale())
          *static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

      float satCoef = m_params.saturationminscale()
        + (m_params.saturationmaxscale()-m_params.saturationminscale())
          *static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

      cv::cvtColor( cv_img, cv_img, CV_BGR2HLS );

      for( size_t x = 0; x < cv_img.rows; ++x )
      {
        for( size_t y = 0; y < cv_img.cols; ++y )
        {
          cv::Vec3b &pixel = cv_img.at<cv::Vec3b>( x, y );

          pixel[1] = static_cast<uint8_t>( std::min(255.0f, float(pixel[1])*lumCoef) );
          pixel[2] = static_cast<uint8_t>( std::min(255.0f, float(pixel[2])*satCoef) );

          cv_img.at<cv::Vec3b>( x, y ) = pixel;
        }
      }
      cv::cvtColor( cv_img, cv_img, CV_HLS2BGR );
    }

    // Apply random scale
    if( m_params.verticalminscale() != 0.0f
      || m_params.verticalmaxscale() != 0.0f
      || m_params.horizontalminscale() != 0.0f
      || m_params.horizontalmaxscale() != 0.0f )
    {
      float xScale = m_params.verticalminscale()
        + (m_params.verticalmaxscale()-m_params.verticalminscale())
          *static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

      float yScale = m_params.horizontalminscale()
        + (m_params.horizontalmaxscale()-m_params.horizontalminscale())
          *static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      cv::Mat cv_scaled;
      cv::resize( cv_img, cv_scaled, cv::Size(), xScale, yScale, cv::INTER_LINEAR );
      cv_img = cv_scaled;
    }
  }
private:
  UltinousTransformationParameter const& m_params;
  Phase m_phase;
};

}  // namespace ultinous
}  // namespace caffe
#endif  // USE_OPENCV
