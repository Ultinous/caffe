//
// Created by bzambo on 11/4/19.
//

#pragma once


#ifndef CAFFE_ROT_LOSS_H
#define CAFFE_ROT_LOSS_H
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

namespace caffe {

    template <typename Dtype>
    class RotLossLayer : public LossLayer<Dtype> {


    public:
        explicit RotLossLayer(const LayerParameter& param)
                : LossLayer<Dtype>(param) {}

        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "Rot"; }
        virtual inline int ExactNumTopBlobs() const { return -1; }
        virtual inline int MinTopBlobs() const { return 1; }
        virtual inline int MaxTopBlobs() const { return 2; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        shared_ptr<Layer<Dtype> >rotloss_layer_;

        Blob<Dtype> pred_t_;

        Blob<Dtype> identity_3x3_;

        vector<Dtype> mult_mtx_const;




    };
}


#endif //CAFFE_ROT_LOSS_H
