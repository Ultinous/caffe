#pragma once

#include <caffe/layers/conv_layer.hpp>
#include <caffe/layers/relu_layer.hpp>
#include <caffe/layers/pooling_layer.hpp>
#include <caffe/layers/lrn_layer.hpp>
#include <caffe/layers/concat_layer.hpp>
#include <caffe/layers/dropout_layer.hpp>
#include <caffe/layers/inner_product_layer.hpp>
#include <caffe/layers/softmax_layer.hpp>

/*
#define REGISTER_LAYER(type, creator)                                  \
  static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>);     \
  static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)    \


#define REGISTER_LAYER_CLASS(type)                                             \
  template <typename Dtype>                                                    \
  shared_ptr<Layer<Dtype> > Creator_##type##Layer(const LayerParameter& param) \
  {                                                                            \
  return shared_ptr<Layer<Dtype> >(new type##Layer<Dtype>(param));           \
  }                                                                            \
*/

namespace caffe
{

template<typename Dtype, typename LayerType>
shared_ptr<Layer<Dtype> > GeneralLayerCreator(const LayerParameter& param)
{
  return shared_ptr<Layer<Dtype> >(new LayerType(param));
}


#define REGISTER_LAYER(type) \
if(LayerRegistry<float>::Registry().count(#type) == 0 ) \
{ \
  LayerRegistry<float>::AddCreator(#type, GeneralLayerCreator<float, type##Layer<float> >); \
  LayerRegistry<double>::AddCreator(#type, GeneralLayerCreator<double, type##Layer<double> >); \
}\

class RegisterHelper
{
public:
  static void register_layers()
  {
    REGISTER_LAYER(Convolution);
    REGISTER_LAYER(ReLU);
    REGISTER_LAYER(Pooling);
    REGISTER_LAYER(LRN);
    REGISTER_LAYER(Concat);
    REGISTER_LAYER(Dropout);
    REGISTER_LAYER(InnerProduct);
    REGISTER_LAYER(Softmax);
  }
};

#undef REGISTER_LAYER

} // namespace caffe