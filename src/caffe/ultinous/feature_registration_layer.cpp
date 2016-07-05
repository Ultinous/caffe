#include <vector>

#include "caffe/ultinous/feature_registration_layer.hpp"
#include "caffe/ultinous/FeatureMap.hpp"

namespace caffe {
namespace ultinous {

template <typename Dtype>
void FeatureRegistrationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
FeatureRegistrationLayer<Dtype>::~FeatureRegistrationLayer<Dtype>() {
}

template <typename Dtype>
void FeatureRegistrationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  string featureMapId = this->layer_param_.feature_registration_param().featuremapid();
  FeatureMap<Dtype>& featureMap = FeatureMapContainer<Dtype>::instance( featureMapId );

  size_t batchSize = bottom[0]->num();
  for( int i = 0; i < batchSize; i++ ) {
    int imageDim = bottom[0]->count() / batchSize;

    std::vector<Dtype> feature_a(imageDim);
    std::vector<Dtype> feature_p(imageDim);
    std::vector<Dtype> feature_n(imageDim);

    for( int j = 0; j < imageDim; j++ ) {
      feature_a[j] = bottom[0]->cpu_data()[ i*imageDim + j ];
      feature_p[j] = bottom[1]->cpu_data()[ i*imageDim + j ];
      feature_n[j] = bottom[2]->cpu_data()[ i*imageDim + j ];
    }

    typename FeatureMap<Dtype>::Index index_a =  bottom[3]->cpu_data()[i];
    typename FeatureMap<Dtype>::Index index_p =  bottom[3]->cpu_data()[batchSize+i];
    typename FeatureMap<Dtype>::Index index_n =  bottom[3]->cpu_data()[2*batchSize+i];

    featureMap.update( index_a, feature_a );
    featureMap.update( index_p, feature_p );
    featureMap.update( index_n, feature_n );
  }
}

#ifdef CPU_ONLY
STUB_GPU(FeatureRegistrationLayer);
#endif

INSTANTIATE_CLASS(FeatureRegistrationLayer);
REGISTER_LAYER_CLASS(FeatureRegistration);

}  // namespace ultinous
}  // namespace caffe
