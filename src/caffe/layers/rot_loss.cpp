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
        //A=||I -preds * label^T||

        float a11 = 1.0f- (preds[i*9 + 0] * labels[i*9 + 0] + preds[i*9 + 1] * labels[i*9 + 1] + preds[i*9 + 2] * labels[i*9 + 2]);
        float a12 =  (preds[i*9 + 0] * labels[i*9 + 3] + preds[i*9 + 1] * labels[i*9 + 4] + preds[i*9 + 2] * labels[i*9 + 5]);
        float a13 =  (preds[i*9 + 0] * labels[i*9 + 6] + preds[i*9 + 1] * labels[i*9 + 7] + preds[i*9 + 2] * labels[i*9 + 8]);

        float a21 = (preds[i*9 + 3] * labels[i*9 + 0] + preds[i*9 + 4] * labels[i*9 + 1] + preds[i*9 + 5] * labels[i*9 + 2]);
        float a22 = 1.0f -(preds[i*9 + 3] * labels[i*9 + 3] + preds[i*9 + 4] * labels[i*9 + 4] + preds[i*9 + 5] * labels[i*9 + 5]);
        float a23 = (preds[i*9 + 3] * labels[i*9 + 6] + preds[i*9 + 4] * labels[i*9 + 7] + preds[i*9 + 5] * labels[i*9 + 8]);

        float a31 = (preds[i*9 + 6] * labels[i*9 + 0] + preds[i*9 + 7] * labels[i*9 + 1] + preds[i*9 + 8] * labels[i*9 + 2]);
        float a32 = (preds[i*9 + 6] * labels[i*9 + 3] + preds[i*9 + 7] * labels[i*9 + 4] + preds[i*9 + 8] * labels[i*9 + 5]);
        float a33 = 1.0f -(preds[i*9 + 6] * labels[i*9 + 6] + preds[i*9 + 7] * labels[i*9 + 7] + preds[i*9 + 8] * labels[i*9 + 8]);

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
                //x1
                float fdx11 = -2.0f*y[9*i + 0] + 2.0f *x[9*i + 0] * y[9*i + 0] * y[9*i + 0] + 2.0f*y[9*i +0] *x[9*i +1] * y[9*i +1]
                             +2.0f*y[9*i +0] * x[9*i  + 2]*y[9*i +2] +2.0f * x[9*i + 0]* y[9*i+3] * y[9*i+3]+ 2.0f * y[9*i +3] + x[9*i +1] * y[9*i +4]
                        + 2.0f * y[9*i  + 3] * x[9*i + 2] * y[9*i  +5 ] + 2.0f * x[9*i+0 ] * y[9*i + 6] * y[9*i +6] + 2.0f * y[9*i +6 ] * x[9*i +1] *y[9*i  + 7]
                        + 2.0f * y[9*i  + 6]  * x[9*i + 2] * y[9*i  + 8];

                //x2
                float fdx12 = -2.0f * y[9*i +1] + 2.0f*x[9*i +1] * y[9*i +1] * y[9*i+1] + 2.0f*x[9*i +0] * y[9*i+0]* y[9*i+1]+ 2.0f*y[9*i+1] * x[9*i+2]*y[9*i+2]
                        +2.0f*x[9*i+1]*y[9*i+4]*y[9*i+4]+ 2.0f*x[9*i+0]*y[9*i+3]+y[9*i+4] + 2.0f*y[9*i+4]*x[9*i+2] *y[9*i +5] + 2.0f *x[9*i +1]*y[9*i+7]*y[9*i+7]
                        +2.0f *x[9*i +0]*y[9*i +6] *y[9*i+7]+2.0f*y[9*i + 7]*x[9*i+2]*y[9*i +8];
                //x3
                float fdx13 = -2.0f * y[9*i+2] * y[9*i +2] +2.0f*x[9*i +2]*y[9*i+2]*y[9*i+2] +2.0f*x[9*i+0]*y[9*i+0]*y[9*i+2]+ 2.0f*x[9*i+1]*y[9*i+1]*y[9*i+2]
                        +2.0f*x[9*i +2] *y[9*i +5]*y[9*i+5]+ 2.0f*x[9*i +0]*y[9*i +3]*y[9*i+5] + 2.0f*x[9*i+1]*y[9*i+4]*y[9*i +5]+ 2.0f*x[9*i +2] *y[9*i +8]*y[9*i +8]
                        +2.0f*x[9*i +0]*y[9*i+6]*y[9*i+8]+ 2.0*x[9*i +1]*y[9*i+7]*y[9*i+8];

                //x4
                float fdx21 = 2.0f*x[9*i+3]*y[9*i+0] *y[9*i +0]+ 2.0f* y[9*i +0]*x[9*i +4] * y[9*i+1] +2.0f*y[9*i+0]*x[9*i+5]*y[9*i +2] - 2.0f *y[9*i +3]*y[9*i+3]
                        +2.0f*x[9*i +3] * y[9*i+3]*y[9*i+3] +2.0f* y[9*i +3]*x[9*i+4]*y[9*i+4]+ 2.0f*y[9*i+3]*x[9*i+5]*y[9*i+5]+ 2.0f*x[9*i+3]*y[9*i+6]*y[9*i+6]
                        +2.0f*y[9*i+6]*x[9*i+4]*y[9*i+7]+2.0f*y[9*i+6]*x[9*i+5]*y[9*i+8];

                //x5
                float fdx22 = 2.0f*x[9*i+4]  *y[9*i+1] *y[9*i+1] +2.0f*x[9*i +3]*y[9*i+0]*y[9*i+1] +2.0f*y[9*i+1]*x[9*i+5]*y[9*i+2] -2.0f*y[9*i +4] * y[9*i+4]
                        +2.0f*x[9*i+4]*y[9*i+4]*y[9*i+4] + 2.0f*x[9*i+3]*y[9*i+3]*y[9*i+4] +2.0f*y[9*i+4]*x[9*i+5]*y[9*i+5]+ 2.0f*x[9*i+4]*y[9*i+7]*y[9*i+7]
                        +2.0f*x[9*i+3]*y[9*i+6]*y[9*i+7]+2.0f*y[9*i+7]*x[9*i+5]*y[9*i+8];

                //x6
                float fdx23 = 2.0f*x[9*i +5] * y[9*i+2]*y[9*i+2]+2.0f*x[9*i+3]*y[9*i +0] *y[9*i +2] +2.0f*x[9*i+4]*y[9*i+1]*y[9*i+2] -2.0f*y[9*i+5]+2.0f*x[9*i+5]*y[9*i+5]*y[9*i+5]
                        +2.0f*x[9*i+3]*y[9*i+3]*y[9*i+5]+2.0f*x[9*i+4]*y[9*i+4]*y[9*i+5] + 2.0f*x[9*i+5]*y[9*i+8]*y[9*i+8]+2.0f*x[9*i+3]*y[9*i+6]*y[9*i+8]+2.0f*x[9*i+4]*y[9*i+7]*y[9*i+8];


                //x7
                float fdx31 =2.0f * x[9*i+6]*y[9*i+0]*y[9*i+0]+2.0f*y[9*i+0]*x[9*i+7]*y[9*i+1]+2.0f*y[9*i +0]*x[9*i+8]*y[9*i+2]+2.0f*x[9*i+6]*y[9*i+3]*y[9*i+3]+2.0f*y[9*i+3]*x[9*i+7]*y[9*i+4]
                        +2.0f*y[9*i+3]*x[9*i+8]*y[9*i+5]-2.0f*y[9*i+6]+2.0f*x[9*i+6]*y[9*i+6]*y[9*i+6]+2.0f*y[9*i+6]*x[9*i+7]*y[9*i+7]+2.0f*y[9*i+6]*x[9*i+8]*y[9*i+8];

                //x8
                float fdx32 =2.0f*x[9*i+7]*y[9*i+1]*y[9*i+1]+2.0f*x[9*i+6]*y[9*i+0]*y[9*i+1]+2.0f*y[9*i+1]*x[9*i+8]*y[9*i+2]+2.0f*x[9*i+7]*y[9*i+4]*y[9*i+4]+2.0*x[9*i+6]*y[9*i+3]*y[9*i+4]
                        +2.0f*y[9*i+4]*x[9*i+8]*y[9*i+5] -2.0f*y[9*i+7] + 2.0f*x[9*i+7]*y[9*i+7]*y[9*i+7]+2.0f*x[9*i+6]*y[9*i+6]*y[9*i+7]+2.0*y[9*i+7]*x[9*i+8]*y[9*i+8];

                //x9
                float fdx33 = 2.0f*x[9*i+8]*y[9*i+2]*y[9*i+2] +2.0f*x[9*i+6]*y[9*i+0]*y[9*i+2]+2.0f*x[9*i+7]*y[9*i+1]*y[9*i+2] +2.0f*x[9*i+8]*y[9*i+5]*y[9*i+5]+2.0f*x[9*i+6]*y[9*i+3]*y[9*i+5]
                        +2.0f*x[9*i+7]*y[9*i+4]*y[9*i+5]-2.0f*y[9*i+8]+2.0f*x[9*i+8]*y[9*i+8]*y[9*i+8]+2.0f*x[9*i+6]*y[9*i+6]*y[9*i+8]+2.0f*x[9*i+7]*y[9*i+7]*y[9*i+8];



                bottom[0]->mutable_cpu_diff()[9*i +0]= fdx11;
                bottom[0]->mutable_cpu_diff()[9*i +1]= fdx12;
                bottom[0]->mutable_cpu_diff()[9*i +2]= fdx13;
                bottom[0]->mutable_cpu_diff()[9*i +3]= fdx21;
                bottom[0]->mutable_cpu_diff()[9*i +4]= fdx22;
                bottom[0]->mutable_cpu_diff()[9*i +5]= fdx23;
                bottom[0]->mutable_cpu_diff()[9*i +6]= fdx31;
                bottom[0]->mutable_cpu_diff()[9*i +7]= fdx32;
                bottom[0]->mutable_cpu_diff()[9*i +8]= fdx33;

            }

    }




}


INSTANTIATE_CLASS(RotLossLayer);
REGISTER_LAYER_CLASS(RotLoss);


} // caffe namespace