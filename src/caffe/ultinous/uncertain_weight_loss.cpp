//
// Created by bzambo on 12/17/19.
// based on: https://arxiv.org/pdf/1705.07115.pdf
//

#include <algorithm>
#include <vector>

#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/ultinous/uncertain_weight_loss.hpp"

namespace caffe {
    namespace ultinous {


        template <typename Dtype>
        void UncertainWeightLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {


            UncertainWeightLossLayer::log_sig_1 = Dtype(caffe_rng_rand());
            UncertainWeightLossLayer::log_sig_2 = Dtype(caffe_rng_rand());
            UncertainWeightLossLayer::log_sig_3 = Dtype(caffe_rng_rand());

            UncertainWeightLossLayer::euler_num = Dtype(std::exp(1.0));
        }



        template <typename Dtype>
        void UncertainWeightLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {

        const Dtype *b1 = bottom[0]->cpu_data();
        const Dtype *b2 = bottom[1]->cpu_data();
        const Dtype *b3 = bottom[2]->cpu_data();

        Dtype loss = Dtype(0);






            Dtype exp_res = Dtype(std::exp(-1 * UncertainWeightLossLayer::log_sig_1 ));


            loss +=     exp_res * b1[0] + UncertainWeightLossLayer::log_sig_1;

            exp_res = Dtype(std::exp(-1 * UncertainWeightLossLayer::log_sig_2 ));
            loss +=  exp_res * b2[0] + UncertainWeightLossLayer::log_sig_2;

            exp_res = Dtype(std::exp(-1 * UncertainWeightLossLayer::log_sig_3 ));
            loss += exp_res * b3[0] + UncertainWeightLossLayer::log_sig_3;


            top[0]->mutable_cpu_data()[0] = loss;

            LOG(INFO) << "Sigmas: " << UncertainWeightLossLayer::log_sig_1 << ", " << UncertainWeightLossLayer::log_sig_2
            << ", " << UncertainWeightLossLayer::log_sig_3;

        }


        template <typename Dtype>
        void UncertainWeightLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


            if(propagate_down[0]) {

                bottom[0]->mutable_cpu_diff()[0] = bottom[0]->cpu_data()[0] * Dtype(-1) *
                        std::exp( Dtype(-1.0) * UncertainWeightLossLayer::log_sig_1) + Dtype(1);

                bottom[1] ->mutable_cpu_diff()[0] = bottom[0]->cpu_data()[0] * Dtype(-1) *
                          std::exp( Dtype(-1.0) * UncertainWeightLossLayer::log_sig_2) + Dtype(1);

                bottom[2]->mutable_cpu_diff()[0] = bottom[0]->cpu_data()[0] * Dtype(-1) *
                          std::exp(Dtype(-1.0) * UncertainWeightLossLayer::log_sig_3) + Dtype(1);


                //update the log_sig variables
                UncertainWeightLossLayer::log_sig_1 += bottom[0]->cpu_data()[0] * Dtype(-1) *
                        std::exp( Dtype(-1.0) * UncertainWeightLossLayer::log_sig_1) + Dtype(1);

                UncertainWeightLossLayer::log_sig_2 += bottom[0]->cpu_data()[0] * Dtype(-1) *
                                                        std::exp( Dtype(-1.0) * UncertainWeightLossLayer::log_sig_2) + Dtype(1);


                UncertainWeightLossLayer::log_sig_3 += bottom[0]->cpu_data()[0] * Dtype(-1) *
                                                       std::exp(Dtype(-1.0) * UncertainWeightLossLayer::log_sig_3) + Dtype(1);
            }


            }



#ifdef CPU_ONLY
STUB_GPU(uncertain_weight_loss);
#endif

        INSTANTIATE_CLASS(UncertainWeightLossLayer);
        REGISTER_LAYER_CLASS(UncertainWeightLoss);

    }
  }