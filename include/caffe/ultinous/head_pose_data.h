#ifndef CAFFE_HEAD_POSE_DATA_H
#define CAFFE_HEAD_POSE_DATA_H

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "../ultinous/UltinousTransformer.hpp"


namespace caffe {

    namespace ultinous{

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
        template <typename Dtype>
        class HeadPoseDataLayer : public BasePrefetchingDataLayer<Dtype> {
        public:
            explicit HeadPoseDataLayer(const LayerParameter& param)
                    : BasePrefetchingDataLayer<Dtype>(param)
                    , m_unTransformer(this->layer_param_.ultinous_transform_param(), this->phase_)
            { }
            ~HeadPoseDataLayer() override;
            void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) override;

            inline const char* type() const override { return "HeadPoseDataLayer"; }
            inline int ExactNumBottomBlobs() const override { return 0; }
            inline int ExactNumTopBlobs() const override { return 2; }

        protected:
            shared_ptr<Caffe::RNG> prefetch_rng_;
            virtual void ShuffleImages();
            void load_batch(Batch<Dtype>* batch) override;

            vector<std::pair<std::string, std::vector<double> > > lines_;
            int lines_id_;

            cv::Point3f get_random_head_pose_direction();
            int select_line_id_for_head_pose_direction();

            ultinous::UltinousTransformer m_unTransformer;
        };

    } // namespace ultinous
} // namespace caffe

#endif //CAFFE_HEAD_POSE_DATA_H
