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

    CHECK_LE(m_params.minocclusionradius(), m_params.maxocclusionradius())
      << "minOcclusionRadius must be less than maxOcclusionRadius!";

    m_uniformNoiseStrength = static_cast<int16_t>( m_params.uniformnoisestrength() );

    xorshf96_x=123456789;
    xorshf96_y=362436069;
    xorshf96_z=521288629;
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

    // Apply affine transformation
    cv::Mat cv_affine;
    applyAffine( cv_img, cv_affine );
    cv_img = cv_affine;

    // Occlusion
    if( m_params.numocclusions() != 0 && m_params.maxocclusionradius() != 0 )
    {

      for( size_t i = 0; i < m_params.numocclusions(); ++ i )
      {
        size_t r = m_params.minocclusionradius()
        + (m_params.maxocclusionradius()-m_params.minocclusionradius())
          *static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

        size_t cx = r + std::rand() % (cv_img.rows - 2*r);
        size_t cy = r + std::rand() % (cv_img.cols - 2*r);

        cv::circle( cv_img, cv::Point(cx,cy), r, cv::Scalar(0), -1 );
      }
    }

    // Uniform noise
    if( m_uniformNoiseStrength != 0 )
    {
      CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
      CHECK( cv_img.isContinuous() ) << "OpenCV image matrix must be contiuously stored in memory!";

      //boost::uniform_int<int16_t> distribution( -m_uniformNoiseStrength, m_uniformNoiseStrength );
      //boost::variate_generator< RNGType, boost::uniform_int<int16_t> > noiseGenerator(m_rng, distribution);

      size_t N = cv_img.channels() * cv_img.rows * cv_img.cols;

      //vector<int16_t> rndVec(N);
      //generate(rndVec.begin(), rndVec.end(), noiseGenerator);


      uint8_t * p = cv_img.data;
      int16_t MIN = static_cast<int16_t>(0);
      int16_t MAX = static_cast<int16_t>(std::numeric_limits<uint8_t>::max());
      for( size_t i =0; i < N; ++i, ++p) {
        *p = static_cast<uint8_t>(
          std::max( MIN, std::min( MAX,
            static_cast<int16_t>(static_cast<int16_t>(*p)
              + static_cast<int16_t>(myRandom(1+2*m_uniformNoiseStrength))-m_uniformNoiseStrength )
          ) )
        );
      }
    }

    // Padding
    if( m_params.padtop() != 0 || m_params.padbottom() != 0
      || m_params.padleft() != 0 || m_params.padright() != 0 )
    {
      cv::Mat dst;

      cv::copyMakeBorder( cv_img, dst, m_params.padtop(), m_params.padbottom()
                    , m_params.padleft(), m_params.padright(), cv::BORDER_CONSTANT );

      cv_img = dst;
    }


    /*static int ccc = 0;
    std::string fname = boost::lexical_cast<std::string>(ccc) + ".png";
    cv::imwrite( fname, cv_img );
    ++ccc;*/

  }

private:
  void applyAffine( cv::Mat& src, cv::Mat& dst )
  {
    const double pi = std::acos(-1);

    float sx=1.0f, sy=1.0f, ex=0.0f, ey=0.0f, ca=1.0f, sa=0.0f;

    float ty = static_cast<float>(src.rows)/2.0f;
    float tx = static_cast<float>(src.cols)/2.0f;

    if( m_params.verticalminscale() != 1.0f || m_params.verticalmaxscale() != 1.0f)
    {
      sy = m_params.verticalminscale()
        + (m_params.verticalmaxscale()-m_params.verticalminscale())
          *static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    if( m_params.horizontalminscale() != 1.0f || m_params.horizontalmaxscale() != 1.0f )
    {
      sx = m_params.horizontalminscale()
        + (m_params.horizontalmaxscale()-m_params.horizontalminscale())
          *static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    if( m_params.verticalminshear() != 0.0f || m_params.verticalmaxshear() != 0.0f)
    {
      ey = m_params.verticalminshear()
        + (m_params.verticalmaxshear()-m_params.verticalminshear())
          *static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    if( m_params.horizontalminshear() != 0.0f || m_params.horizontalmaxshear() != 0.0f)
    {
      ex = m_params.horizontalminshear()
        + (m_params.horizontalmaxshear()-m_params.horizontalminshear())
          *static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    if( m_params.minrotation() != 0.0f || m_params.maxrotation() != 0.0f)
    {
      float deg = m_params.minrotation()
        + (m_params.maxrotation()-m_params.minrotation())
          *static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      float rad = deg / 180.0f * pi;

      ca = std::cos(rad);
      sa = std::sin(rad);
    }

    cv::Mat affine( 2, 3, CV_32FC1 );

    affine.at<float>( 0, 0 ) = sx*(ca+sa*ex);
    affine.at<float>( 0, 1 ) = sx*(-sa+ca*ex);

    affine.at<float>( 1, 0 ) = sy*(ca*ey+sa);
    affine.at<float>( 1, 1 ) = sy*(-sa*ey+ca);

    affine.at<float>( 0, 2 ) = affine.at<float>(0,0)*(-tx)
              + affine.at<float>(0,1)*(-ty)
              + tx;
    affine.at<float>( 1, 2 ) = affine.at<float>(1,0)*(-tx)
              + affine.at<float>(1,1)*(-ty)
              + ty;

    cv::warpAffine( src, dst, affine, src.size(), CV_INTER_CUBIC );
  }

  uint32_t xorshf96(void) {          //period 2^96-1
    xorshf96_x ^= xorshf96_x << 16;
    xorshf96_x ^= xorshf96_x >> 5;
    xorshf96_x ^= xorshf96_x << 1;
    xorshf96_t = xorshf96_x;
    xorshf96_x = xorshf96_y;
    xorshf96_y = xorshf96_z;
    xorshf96_z = xorshf96_t ^ xorshf96_x ^ xorshf96_y;
    return xorshf96_z;
  }

  uint64_t myRandom( uint64_t N )
  {
    if( N-1 <= std::numeric_limits<uint32_t>::max() )
    {
      return uint64_t(xorshf96()) % N;
    }
    uint64_t r = 0;
    r += uint64_t(xorshf96());
    r += uint64_t(xorshf96())<<32;
    return r % N;
  }
private:
  UltinousTransformationParameter const& m_params;
  Phase m_phase;

  int16_t m_uniformNoiseStrength;
  RNGType m_rng;

    /* Random generator variables */
  uint32_t xorshf96_x, xorshf96_y, xorshf96_z;
  uint32_t xorshf96_t;

};

}  // namespace ultinous
}  // namespace caffe
#endif  // USE_OPENCV
