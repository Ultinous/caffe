#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif

#include "caffe/ultinous/cpm_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe
{
namespace ultinous
{
  template <typename Dtype>
  CPMDataLayer<Dtype>::CPMDataLayer(const LayerParameter& param)
    : BasePrefetchingDataLayer<Dtype>(param)
    , offset_()
  {
    db_.reset(db::GetDB(param.data_param().backend()));
    db_->Open(param.data_param().source(), db::READ);
    cursor_.reset(db_->NewCursor());
    cpm_data_transformer_.reset(new CPMDataTransformer<Dtype>(param.cpm_transform_param(), param.phase()));
  }

  template <typename Dtype>
  CPMDataLayer<Dtype>::~CPMDataLayer()
  {
    this->StopInternalThread();
  }

  template <typename Dtype>
  void CPMDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top)
  {
    const int batch_size = this->layer_param_.data_param().batch_size();
    // Read a data point, and use it to initialize the top blob.
    Datum datum;
    datum.ParseFromString(cursor_->value());
    LOG(INFO) << datum.height() << " " << datum.width() << " " << datum.channels();

    // Use cpm_data_transformer to infer the expected blob shape from datum.
    vector<int> top_shape = this->cpm_data_transformer_->InferBlobShape(datum);
    this->transformed_data_.Reshape(top_shape);
    // Reshape top[0] and prefetch_data according to the batch_size.
    top_shape[0] = batch_size;
    top[0]->Reshape(top_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i)
    {
      this->prefetch_[i]->data_.Reshape(top_shape);
    }
    LOG_IF(INFO, Caffe::root_solver())
      << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

    // label
    if (this->output_labels_)
    {
      /*
      const int stride = this->layer_param_.cpm_transform_param().stride();
      const int height = this->phase_ != TRAIN ? datum.height() :
                         this->layer_param_.cpm_transform_param().crop_size_y();
      const int width = this->phase_ != TRAIN ? datum.width() :
                        this->layer_param_.cpm_transform_param().crop_size_x();
      */

      std::vector<int> label_shape = this->cpm_data_transformer_->InferLabelBlobShape(datum);
      this->transformed_label_.Reshape(label_shape);
      label_shape[0] = batch_size;
      top[1]->Reshape(label_shape);
      for (int i = 0; i < this->prefetch_.size(); ++i)
      {
        this->prefetch_[i]->label_.Reshape(label_shape);
      }

    }
  }

  template<typename Dtype>
  void CPMDataLayer<Dtype>::Next()
  {
    cursor_->Next();
    if (!cursor_->valid()) {
      LOG_IF(INFO, Caffe::root_solver())
      << "Restarting data prefetching from start.";
      cursor_->SeekToFirst();
    }
    offset_++;
  }

  // TODO(zssanta): Is this necessary?
  template <typename Dtype>
  bool CPMDataLayer<Dtype>::Skip()
  {
    int size = Caffe::solver_count();
    int rank = Caffe::solver_rank();
    bool keep = (offset_ % size) == rank ||
                // In test mode, only rank 0 runs, so avoid skipping
                this->layer_param_.phase() == TEST;
    return !keep;
  }

  template<typename Dtype>
  void CPMDataLayer<Dtype>::load_batch(Batch<Dtype>* batch)
  {
    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0;
    double trans_time = 0;
    CPUTimer timer;
    CHECK(batch->data_.count());
    CHECK(this->transformed_data_.count());

    const int batch_size = this->layer_param_.data_param().batch_size();

    auto* top_data = batch->data_.mutable_cpu_data();
    auto* top_label = (this->output_labels_)? batch->label_.mutable_cpu_data() : nullptr;

    Datum datum;
    for (int item_id = 0; item_id < batch_size; ++item_id) {
      timer.Start();
      while (Skip()) {
        Next();
      }
      datum.ParseFromString(cursor_->value());
      read_time += timer.MicroSeconds();

      if (item_id == 0) {
        // Reshape according to the first datum of each batch
        // on single input batches allows for inputs of varying dimension.
        // Use data_transformer to infer the expected blob shape from datum.
        vector<int> top_shape = this->cpm_data_transformer_->InferBlobShape(datum);
        this->transformed_data_.Reshape(top_shape);
        // Reshape batch according to the batch_size.
        top_shape[0] = batch_size;
        batch->data_.Reshape(top_shape);
      }

      // Apply data transformations (mirror, scale, crop...)
      timer.Start();
      const auto offset_data = batch->data_.offset(item_id);
      const auto offset_label = batch->label_.offset(item_id);

      this->transformed_data_.set_cpu_data(top_data + offset_data);
      if (top_label)
      {
        this->transformed_label_.set_cpu_data(top_label + offset_label);
        this->cpm_data_transformer_->Transform(datum, &(this->transformed_data_), &(this->transformed_label_));
      }
      else
      {
        this->cpm_data_transformer_->Transform(datum, &(this->transformed_data_));
      }

      trans_time += timer.MicroSeconds();
      Next();
    }
    timer.Stop();
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
  }

  INSTANTIATE_CLASS(CPMDataLayer);
  REGISTER_LAYER_CLASS(CPMData);

} // namespace ultinous
} // namespace caffe