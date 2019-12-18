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

	     LossLayer<Dtype>::LayerSetUp(bottom, top);
             LOG(INFO) << "Setting up uncertain weight loss" << std::endl;
            UncertainWeightLossLayer::log_sig_1 = Dtype(2);
            UncertainWeightLossLayer::log_sig_2 = Dtype(2);
            UncertainWeightLossLayer::log_sig_3 = Dtype(2);

            UncertainWeightLossLayer::euler_num = Dtype(std::exp(1.0));
             LOG(INFO) << "End of layerSetup call of  uncertain weight loss" << std::endl;
        }

	template<typename Dtype>
	void UncertainWeightLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		                                  const vector<Blob<Dtype>*>& top) 
	{

	    top[0]->ReshapeLike(*bottom[0]);
	}



        template <typename Dtype>
        void UncertainWeightLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        const Dtype *b1 = bottom[0]->cpu_data();
        const Dtype *b2 = bottom[1]->cpu_data();
        const Dtype *b3 = bottom[2]->cpu_data();

        Dtype loss = Dtype(0.0);






            Dtype exp_res = Dtype(std::exp(-1.0 * UncertainWeightLossLayer::log_sig_1 ));


            loss +=     exp_res * b1[0] + UncertainWeightLossLayer::log_sig_1/Dtype(2.0);

            exp_res = Dtype(std::exp(-1.0 * UncertainWeightLossLayer::log_sig_2 ));
            loss +=  exp_res * b2[0] + UncertainWeightLossLayer::log_sig_2 / Dtype(2.0);

            exp_res = Dtype(std::exp(-1.0 * UncertainWeightLossLayer::log_sig_3 ));
            loss += exp_res * b3[0] + UncertainWeightLossLayer::log_sig_3 / Dtype(2.0);


            LOG_EVERY_N(INFO, 50) << "Sigmas: " << UncertainWeightLossLayer::log_sig_1 << ", " << UncertainWeightLossLayer::log_sig_2
            << ", " << UncertainWeightLossLayer::log_sig_3 << std::endl;
            top[0]->mutable_cpu_data()[0] = Dtype(loss);


        }


        template <typename Dtype>
        void UncertainWeightLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


            if(propagate_down[0]) {

                LOG_EVERY_N(INFO, 50) << "Bottom[0] loss: "<< bottom[0]->cpu_data()[0] << std::endl;
                LOG_EVERY_N(INFO, 50) << "Bottom[1] loss: "<< bottom[1]->cpu_data()[0] << std::endl;
                LOG_EVERY_N(INFO, 50) << "Bottom[2] loss: "<< bottom[2]->cpu_data()[0] << std::endl;

		/*LOG(INFO) << "Update s1 " <<   Dtype(-1.0) *
			                        Dtype(std::exp( Dtype(-1.0) * UncertainWeightLossLayer::log_sig_1))  << std::endl;
        	*/
		//update the log_sig variables
                UncertainWeightLossLayer::log_sig_1 += bottom[0]->cpu_data()[0] * Dtype(-1.0) *  Dtype(std::exp( Dtype(-1.0) * UncertainWeightLossLayer::log_sig_1)) + Dtype(0.5);


	        UncertainWeightLossLayer::log_sig_2 += bottom[1]->cpu_data()[0] * Dtype(-1.0) * Dtype( std::exp( Dtype(-1.0) * UncertainWeightLossLayer::log_sig_2) ) + Dtype(0.5);


                UncertainWeightLossLayer::log_sig_3 +=  bottom[2]->cpu_data()[0] *  Dtype(-1.0) *  Dtype(std::exp(Dtype(-1.0) * UncertainWeightLossLayer::log_sig_3)) + Dtype(0.5);

		


                bottom[0]->mutable_cpu_diff()[0] = Dtype(std::exp( Dtype(-1.0) *  UncertainWeightLossLayer::log_sig_1 ));

                bottom[1] ->mutable_cpu_diff()[0] =  Dtype(std::exp( Dtype(-1.0) *  UncertainWeightLossLayer::log_sig_2 ));

                bottom[2]->mutable_cpu_diff()[0] =  Dtype(std::exp( Dtype(-1.0) *  UncertainWeightLossLayer::log_sig_3 ));


            }

            }




        INSTANTIATE_CLASS(UncertainWeightLossLayer);
        REGISTER_LAYER_CLASS(UncertainWeightLoss);

    }
  }
