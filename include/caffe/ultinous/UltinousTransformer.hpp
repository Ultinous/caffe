#pragma once

#ifdef USE_OPENCV
#include <boost/generator_iterator.hpp>
#include <boost/random.hpp>

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
private:
  typedef boost::mt19937 RNGType;

public:
  UltinousTransformer( UltinousTransformationParameter const& params, Phase phase )
    : m_params( params )
    , m_phase( phase )
  {
    CHECK_LT(m_params.uniformnoisestrength(), 128)
      << "uniformNoiseStrength must not exceed 127!";

    m_uniformNoiseStrength = static_cast<int16_t>( m_params.uniformnoisestrength() );

  }

  void transform( cv::Mat& cv_img )
  {
    if( m_phase != TRAIN ) return;

    // Apply color transformation
    if( m_params.luminanceminscale() != 1.0f
      || m_params.luminancemaxscale() != 1.0f
      || m_params.saturationminscale() != 1.0f
      || m_params.saturationmaxscale() != 1.0f )
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
    if( m_params.verticalminscale() != 1.0f
      || m_params.verticalmaxscale() != 1.0f
      || m_params.horizontalminscale() != 1.0f
      || m_params.horizontalmaxscale() != 1.0f )
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


    if( m_uniformNoiseStrength != 0 )
    {
      CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
      CHECK( cv_img.isContinuous() ) << "OpenCV image matrix must be contiuously stored in memory!";

      boost::uniform_int<int16_t> distribution( -m_uniformNoiseStrength, m_uniformNoiseStrength );
      boost::variate_generator< RNGType, boost::uniform_int<int16_t> > noiseGenerator(m_rng, distribution);

      size_t N = cv_img.channels() * cv_img.rows * cv_img.cols;

      uint8_t * p = cv_img.data;
      int16_t MIN = static_cast<int16_t>(0);
      int16_t MAX = static_cast<int16_t>(std::numeric_limits<uint8_t>::max());
      for( size_t i =0; i < N; ++i, ++p)
        *p = static_cast<uint8_t>(
          std::max( MIN, std::min( MAX,
            static_cast<int16_t>(static_cast<int16_t>(*p) + static_cast<int16_t>(noiseGenerator()))
          ) )
        );
    }
  }
private:
  UltinousTransformationParameter const& m_params;
  Phase m_phase;

  int16_t m_uniformNoiseStrength;
  RNGType m_rng;
};

}  // namespace ultinous
}  // namespace caffe
#endif  // USE_OPENCV
