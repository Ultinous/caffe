//
// Created by bzambo on 12/17/19.
//

#ifndef CAFFE_UNCERTAIN_WEIGHT_LOSS_H
#define CAFFE_UNCERTAIN_WEIGHT_LOSS_HPP

#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"


namespace caffe
{
    namespace ultinous {

        template <typename Dtype>
        class UncertainWeightLossLayer : public LossLayer<Dtype> {

        public:
            explicit UncertainWeightLossLayer(const LayerParameter& param)
                    : LossLayer<Dtype>(param){}

            virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
	    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			          const vector<Blob<Dtype>*>& top);

            virtual inline int ExactNumBottomBlobs() const { return 3; }
            virtual inline const char* type() const { return "UncertainWeight"; }
	     virtual inline int ExactNumTopBlobs() const { return -1; }
	             virtual inline int MinTopBlobs() const { return 1; }
		     virtual inline int MaxTopBlobs() const { return 2; }

        protected:
            virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

            virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


            //Three logsig param for the three bottom layer
            Blob<Dtype> log_sig_1;
            Blob<Dtype> log_sig_2;
            Blob<Dtype> log_sig_3;

            Dtype euler_num;

        };




    }
}


#endif //CAFFE_UNCERTAIN_WEIGHT_LOSS_H
