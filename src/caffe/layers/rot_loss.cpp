//
// Created by bzambo on 11/4/19.
//


#include <algorithm>
#include <cfloat>
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/layers/rot_loss.hpp"

namespace caffe {

    template <typename Dtype>
    void RotLossLayer<Dtype>::LayerSetUp(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

        Blob<Dtype> identity_3x3_tmp;
        identity_3x3_tmp.Reshape(1, 1, 3, 3);
        float *indentity_i = (float *) identity_3x3_tmp.mutable_cpu_data();


        for (int l = 0; l < 3; l++) {
            for (int k = 0; k < 3; k++) {
                if (k == l) {

                    indentity_i[l * 3 + k] = 1.0f;

                } else {
                    indentity_i[l * 3 + k] = 0.0f;
                }
            }
        }


      //  caffe_copy(bottom[0]->num(),identity_3x3_tmp.mutable_cpu_data(), identity_3x3_.mutable_cpu_data());


        //mult_mtx_const = vector<Dtype>(bottom[0]->count(), Dtype(1));



        }


template <typename Dtype>
void RotLossLayer<Dtype>::Forward_cpu(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {


    //Calculate the ||I - Bottom * Label^T ||

    //Preds 3x3
    const Dtype* preds = bottom[0]->mutable_cpu_data();

    //Label 3x3
    const Dtype* labels = bottom[1]->cpu_data();


    float result= 0.0f;
    for(int i =0; i< bottom[0]->num(); i++)
    {
        //A=||I -preds^T * label||

        float p11 = preds[i*9+0];
        float p12 = preds[i*9+3];
        float p13 = preds[i*9+6];
        float p21 = preds[i*9+1];
        float p22 = preds[i*9+4];
        float p23 = preds[i*9+7];
        float p31 = preds[i*9+2];
        float p32 = preds[i*9+5];
        float p33 = preds[i*9+8];

        float l11 = labels[i*9+0];
        float l12 = labels[i*9+3];
        float l13 = labels[i*9+6];
        float l21 = labels[i*9+1];
        float l22 = labels[i*9+4];
        float l23 = labels[i*9+7];
        float l31 = labels[i*9+2];
        float l32 = labels[i*9+5];
        float l33 = labels[i*9+8];

        float a11 = 1.0f-(p11*l11 + p21*l21 + p31*l31);
        float a12 = (p11*l12 + p21*l22 + p31*l32);
        float a13 = (p11*l13 + p21*l23 + p31*l33);

        float a21 = (p12*l11 + p22*l21 + p32*l31);
        float a22 = 1.0f-(p12*l12 + p22*l22 + p32*l32);
        float a23 = (p12*l13 + p22*l23 + p32*l33);

        float a31 = (p13*l11 + p23*l21 + p33*l31);
        float a32 = (p13*l12 + p23*l22 + p33*l32);
        float a33 = 1.0f-(p13*l13 + p23*l23 + p33*l33);

        result += a11*a11 + a12*a12 + a13*a13 + a21*a21 + a22*a22 + a23 *a23 + a31*a31 + a32*a32+ a33*a33;

    }



    result = result / (float)bottom[0]->num();


    top[0]->mutable_cpu_data()[0] = Dtype(result);

}

template <typename Dtype>
void RotLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {



        if(propagate_down[0]) {


            //X
            const Dtype *x = bottom[0]->cpu_data();

            //Y
            const Dtype *y = bottom[1]->cpu_data();

            const Dtype *top_diff = top[0]->cpu_diff();
            Dtype* _diff = bottom[0]->mutable_cpu_diff();

            int num = bottom[0]->num();

            for(int i=0; i< num; i++)
            {

                float x11 = x[9*i+0];
                float x12 = x[9*i+3];
                float x13 = x[9*i+6];
                float x21 = x[9*i+1];
                float x22 = x[9*i+4];
                float x23 = x[9*i+7];
                float x31 = x[9*i+2];
                float x32 = x[9*i+5];
                float x33 = x[9*i+8];

                float y11 = y[9*i+0];
                float y12 = y[9*i+3];
                float y13 = y[9*i+6];
                float y21 = y[9*i+1];
                float y22 = y[9*i+4];
                float y23 = y[9*i+7];
                float y31 = y[9*i+2];
                float y32 = y[9*i+5];
                float y33 = y[9*i+8];



                float fdx11 =  - 2.0f*y11 + 2.0f*x11*y11*y11 + 2.0f*y11*x21*y21
                               + 2.0f*y11*x31*y31 + 2.0f*x11*y12*y12 + 2.0f*y12*x21*y22
                               + 2.0f*y12*x31*y32 + 2.0f*x11*y13*y13 + 2.0f*y13*x21*y23
                               + 2.0f*y13*x31*y33;

                float fdx12 =  - 2.0f*y21 + 2.0f*x21*y21*y21 + 2.0f*x11*y11*y21 + 2.0f*y21*x31*y31
                               + 2.0f*x21*y22*y22 + 2.0f*x11*y12*y22 + 2.0f*y22*x31*y32 + 2.0f*x21*y23*y23
                               + 2.0f*x11*y13*y23 + 2.0f*y23*x31*y33;

                float fdx13 =  - 2.0f*y31 + 2.0f*x31*y31*y31 + 2.0f*x11*y11*y31 + 2.0f*x21*y21*y31
                               + 2.0f*x31*y32*y32 + 2.0f*x11*y12*y32 + 2.0f*x21*y22*y32 + 2.0f*x31*y33*y33
                               + 2.0f*x11*y13*y33 + 2.0*x21*y23*y33;

                float fdx21 = 2.0f*x12*y11*y11 + 2.0f*y11*x22*y21 + 2.0f*y11*x32*y31 - 2.0f*y12
                              + 2.0f*x12*y12*y12 + 2.0f*y12*x22*y22 + 2.0f*y12*x32*y32 + 2.0f*x12*y13*y13
                              + 2.0f*y13*x22*y23 + 2.0f*y13*x32*y33;

                float fdx22 = 2.0f*x22*y21*y21 + 2.0f*x12*y11*y21 + 2.0f*y21*x32*y31 - 2.0f*y22
                              + 2.0f*x22*y22*y22 + 2.0f*x12*y12*y22 + 2.0f*y22*x32*y32 + 2.0f*x22*y23*y23
                              + 2.0f*x12*y13*y23 + 2.0f*y23*x32*y33;

                float fdx23 = 2.0f*x32*y31*y31 + 2.0f*x12*y11*y13 + 2.0f*x22*y21*y31 - 2.0f*y32 + 2.0f*x32*y32*y32
                              + 2.0f*x12*y12*y32 + 2.0f*x22*y22*y32 + 2.0f*x32*y33*y33 + 2.0f*x12*y13*y33 + 2.0f*x22*y23*y33;

                float fdx31 = 2.0f*x13*y11*y11 + 2.0f*y11*x23*y21 + 2.0f*y11*x33*y31 + 2.0f*x13*y12*y12 + 2.0f*y12*x23*y22
                              + 2.0f*y12*x33*y32 - 2.0f*y13 + 2.0f*x13*y13*y13 + 2.0f*y13*x23*y23 + 2.0f*y13*x33*y33;

                float fdx32 = 2.0f*x23*y21*y21 + 2.0f*x13*y11*y21 + 2.0f*y21*x33*y31 + 2.0f*x23*y22*y22 + 2.0*x13*y12*y22
                              + 2.0f*y22*x33*y32 - 2.0f*y23 + 2.0f*x23*y23*y23 + 2.0f*x13*y13*y23 + 2.0*y23*x33*y33;

                float fdx33 = 2.0f*x33*y31*y31 + 2.0f*x13*y11*y31 + 2.0f*x23*y21*y31 + 2.0f*x33*y32*y32 + 2.0f*x13*y12*y32
                              + 2.0f*x23*y22*y32 - 2.0f*y33 + 2.0f*x33*y33*y33 + 2.0f*x13*y13*y33 + 2.0f*x23*y23*y33;



                bottom[0]->mutable_cpu_diff()[9*i +0]=  fdx11;
                bottom[0]->mutable_cpu_diff()[9*i +1]=  fdx12;
                bottom[0]->mutable_cpu_diff()[9*i +2]=  fdx13;
                bottom[0]->mutable_cpu_diff()[9*i +3]=  fdx21;
                bottom[0]->mutable_cpu_diff()[9*i +4]=  fdx22;
                bottom[0]->mutable_cpu_diff()[9*i +5]=  fdx23;
                bottom[0]->mutable_cpu_diff()[9*i +6]=  fdx31;
                bottom[0]->mutable_cpu_diff()[9*i +7]=  fdx32;
                bottom[0]->mutable_cpu_diff()[9*i +8]=  fdx33;

            }

    }




}


INSTANTIATE_CLASS(RotLossLayer);
REGISTER_LAYER_CLASS(RotLoss);


} // caffe namespace