#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/ultinous/data_augment_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {
namespace ultinous {

template <typename Dtype>
__global__ void DataAugmentDataAugmentOcclusionGPU(
    Dtype* data,
    int num, int channels, int height, int width,
    const Dtype* circles,
    int numCircles ) {

  int nthreads = num * numCircles;
  int spatialSize = channels*height*width;

	CUDA_KERNEL_LOOP(index, nthreads) {
    int image_ix = index / numCircles;
    int circle_ix = index % numCircles;

    Dtype cx = circles[ image_ix*3*numCircles + circle_ix*3 ];
    Dtype cy = circles[ image_ix*3*numCircles + circle_ix*3 + 1 ];
    Dtype r = circles[ image_ix*3*numCircles + circle_ix*3 + 2 ];

    Dtype * D = data + image_ix*spatialSize;

    for( int x = max(0,(int)floor(cx-r)); x <= min(height-1,(int)ceil(cx+r)); ++x )
    {
      for( int y = max(0,(int)floor(cy-r)); y <= min(width-1,(int)ceil(cy+r)); ++y )
      {
        if( (cx-x)*(cx-x) + (cy-y)*(cy-y) <= r*r )
        {
          for( int c = 0; c < channels; ++c )
            D[c*height*width+x*width+y] = 0;
        }
      }

    }
  }
}

template <typename Dtype>
void DataAugmentLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK(top[0]->count() == bottom[0]->count()) << "Error: in Forward_gpu of DataAugmentLayer.";

	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

  caffe_gpu_memcpy( bottom[0]->count()*sizeof(Dtype), bottom[0]->gpu_data(), top[0]->mutable_gpu_data() );

  if( m_numOcclusions > 0 )
  {
    for( int i = 0; i < num; ++i )
    {
      Dtype *circleData = m_circles.mutable_cpu_data() + i*3*m_numOcclusions;
      for( int j = 0; j < m_numOcclusions; ++j )
      {
        Dtype r = m_minOcclusionRadius
                  + (m_maxOcclusionRadius - m_minOcclusionRadius)
                    * (static_cast<float>(rand()) / static_cast<float>(RAND_MAX));

        size_t cx = r + (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * (static_cast<float>(height) - 2 * r);
        size_t cy = r + (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * (static_cast<float>(width) - 2 * r);

        circleData[3*j+0] = cx;
        circleData[3*j+1] = cy;
        circleData[3*j+2] = r;
      }
    }

    const int nthreads = num * m_numOcclusions;
    DataAugmentDataAugmentOcclusionGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
    CAFFE_CUDA_NUM_THREADS,0,Caffe::cuda_stream()>>>(top[0]->mutable_gpu_data()
    , num, channels, height, width, m_circles.gpu_data(), m_numOcclusions);
  }
}

/*template <typename Dtype>
__global__ void DataAugmentBackwardGPU(const int nthreads, 
	const Dtype* output, Dtype* input) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {
		
		input[index] = output[index];
	}
}*/

template <typename Dtype>
void DataAugmentLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  CHECK(top[0]->count() == bottom[0]->count()) << "Error: in Backward_gpu of DataAugmentLayer.";

  caffe_gpu_memcpy( bottom[0]->count()*sizeof(Dtype), top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff() );

  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  const int nthreads = num * m_numOcclusions;

  if( m_numOcclusions > 0 )
  {
    DataAugmentDataAugmentOcclusionGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
    CAFFE_CUDA_NUM_THREADS,0,Caffe::cuda_stream()>>>(bottom[0]->mutable_gpu_diff()
    , num, channels, height, width, m_circles.gpu_data(), m_numOcclusions);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DataAugmentLayer);

}  // namespace ultinous
}	// namespace caffe
