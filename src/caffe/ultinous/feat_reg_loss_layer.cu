#include "caffe/ultinous/feat_reg_loss_layer.hpp"

#include<algorithm>
#include <numeric>

namespace caffe {
namespace ultinous {




template <typename Dtype>
void FeatRegLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  int num = bottom[0]->num();
  int spatialSize = bottom[0]->count() / num;


  m_batch_norms.clear();
  for( int i = 0; i < num; ++i )
  {
    Dtype norm;

    caffe_gpu_dot(spatialSize, bottom[0]->gpu_data()+i*spatialSize, bottom[0]->gpu_data()+i*spatialSize, &norm);

    norm = std::sqrt(norm);

    m_batch_norms.push_back( norm );

    m_norms.push_back( norm );
    while( m_norms.size() > 1000 )
      m_norms.pop_front();
  }

    static uint64_t iter = 0;
    Dtype sum = std::accumulate(m_norms.begin(), m_norms.end(), 0.0);
    Dtype mean = sum / m_norms.size();

    std::vector<Dtype> diff(m_norms.size());
    std::transform(m_norms.begin(), m_norms.end(), diff.begin(),
                  std::bind2nd(std::minus<Dtype>(), mean));
    Dtype sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    Dtype stdev = std::sqrt(sq_sum / m_norms.size());

    Dtype const & (*minfv) (Dtype const &, Dtype const &) = std::min<Dtype>;
    Dtype min = std::accumulate(m_norms.begin() + 1, m_norms.end(), m_norms.front(), minfv);
    Dtype const & (*maxfv) (Dtype const &, Dtype const &) = std::max<Dtype>;
    Dtype max = std::accumulate(m_norms.begin() + 1, m_norms.end(), m_norms.front(), maxfv);

  if( iter >= 500)
  {
//    m_min_norm = mean-2.2*stdev;
  //  m_max_norm = mean+2.2*stdev;
    m_min_norm = mean;// / 1.5;
    m_max_norm = mean;// * 1.5;
  }

  if( 0 == (iter % 100) )
  {
    std::cout << "mean=" << mean << " stddev=" << stdev << " min=" << min << " max=" << max << " min_norm=" << m_min_norm << " max_norm=" << m_max_norm << std::endl;
  }

  ++iter;

  CUDA_POST_KERNEL_CHECK;

  Dtype loss = Dtype(stdev);
  //caffe_gpu_dot(count, m_ones.gpu_data(), m_errors.gpu_data(), &loss);
  top[0]->mutable_cpu_data()[0] = loss;// / bottom[0]->num();
}




template <typename Dtype>
void FeatRegLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int spatialSize = bottom[0]->count() / num;

  if( m_min_norm < 0 || m_max_norm < 0 )
    return;
//  std::cout << "OK "<< m_min_norm << " " << m_max_norm << std::endl;

  for( int i = 0; i < num; ++i )
  {
    Dtype norm = m_batch_norms.at(i);
  //  if( norm < 0.1 )
//      std::cout << "PARA" << std::endl;

    if( (norm >= m_min_norm && norm <= m_max_norm) || norm < 0.1)
      continue;

    Dtype scale = Dtype(0.0);

    if( norm < m_min_norm )
      scale = 1.0 - (m_min_norm/norm);
    else
      scale = 1.0 - (m_max_norm/norm);

    caffe_gpu_axpy( spatialSize, scale, bottom[0]->gpu_data()+i*spatialSize, bottom[0]->mutable_gpu_diff()+i*spatialSize );
//    caffe_gpu_scal( spatialSize, scale, bottom[0]->mutable_gpu_diff()+i*spatialSize );
  }

  CUDA_POST_KERNEL_CHECK;

  const Dtype loss_weight = top[0]->cpu_diff()[0];
  caffe_gpu_scal(count, loss_weight, bottom[0]->mutable_gpu_diff() );
}

INSTANTIATE_LAYER_GPU_FUNCS(FeatRegLossLayer);

}  // namespace ultinous
}  // namespace caffe
