#include <vector>

#include <iomanip>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

#include "caffe/ultinous/imwrite_layer.hpp"
#include "caffe/ultinous/FeatureMap.hpp"

namespace caffe {
namespace ultinous {

template <typename Dtype>
void ImwriteLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
ImwriteLayer<Dtype>::~ImwriteLayer<Dtype>() {
}

template <typename Dtype>
void ImwriteLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  std::string output_path = this->layer_param_.imwrite_param().output_path();

  if( this->phase_ == TEST ) {
    output_path = this->layer_param_.imwrite_param().test_output_path();
    if( output_path.empty() )
      return;
  }

  int frequency = this->layer_param_.imwrite_param().iterations();
  ++m_iterations;
  if( 0!=frequency && 0!=(m_iterations%frequency) )
    return;


  boost::filesystem::path dir(output_path);
  boost::filesystem::create_directories(dir);

  std::string postfix = this->layer_param_.imwrite_param().postfix();
  std::string extension = this->layer_param_.imwrite_param().extension();
  Dtype int_shift = this->layer_param_.imwrite_param().intensity_shift();
  Dtype int_mult = this->layer_param_.imwrite_param().intensity_mult();

  int num = bottom[0]->num();
  int height = bottom[0]->height();
  int channels = bottom[0]->channels();
  int width = bottom[0]->width();
  int spatialSize = channels*height*width;

  for( int i = 0; i < num; ++i )
  {
    std::stringstream ss;

    ss << output_path << "/";
    ss << std::setfill('0') << std::setw(3) << i;
    if( !postfix.empty() ) ss << postfix;
    ss<< "." << extension;

    std::string filename(ss.str());

    cv::Mat im( height, width, (channels==3?CV_8UC3:CV_8UC1), cv::Scalar(0) );

    Dtype const * data = bottom[0]->cpu_data() + i*spatialSize;

    for( size_t c = 0; c < channels; ++c )
      for( size_t x = 0; x < height; ++x )
        for( size_t y = 0; y < width; ++y )
        {
          int value = int_shift+data[c*height*width+x*width+y]*int_mult;
          value = std::max(0, std::min(255, value ) );
          im.data[ channels*(x*width+y) + c ] = value;
        }
    cv::imwrite( filename, im );
  }
}

#ifdef CPU_ONLY
STUB_GPU(ImwriteLayer);
#endif

INSTANTIATE_CLASS(ImwriteLayer);
REGISTER_LAYER_CLASS(Imwrite);

}  // namespace ultinous
}  // namespace caffe
