#include "caffe/ultinous/queue_prediction_data_layer.h"

#include "caffe/util/rng.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe
{
namespace ultinous
{

template <typename Dtype>
QueuePredictionDataLayer<Dtype>::~QueuePredictionDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void QueuePredictionDataLayer<Dtype>::DataLayerSetUp(std::vector<caffe::Blob<Dtype> *> const &bottom
  , std::vector<caffe::Blob<Dtype> *> const &top)
{
  // read in the parameters
  const uint32_t batch_size = this->layer_param_.queue_prediction_data_param().batch_size();
  const uint32_t input_size = this->layer_param_.queue_prediction_data_param().input_size();
  const uint32_t output_size = this->layer_param_.queue_prediction_data_param().output_size();

  // read in the input enters, exits, queue, cashier and output queue, enter, exit in lines_
  const std::string& source = this->layer_param_.queue_prediction_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());

  // 1 line = split between spaces, then first input_size is enters, second input_size is exits, one queue, one cashier
  // output_size enters, output_size exits and output_size queues
  string line;
  m_dataLength = 2*input_size + 2;
  m_labelLength = 3*output_size;
  while (std::getline(infile, line)) {
    std::stringstream lineToParse(line);
    std::string token;
    std::vector<float> parsedLine;
    while(getline(lineToParse, token, ' '))
      parsedLine.emplace_back(static_cast<Dtype>(std::stof(token)));
    CHECK_GT(m_dataLength, 0) << "Input length must be greater than 0.";
    CHECK_GT(m_labelLength, 0) << "Output length must be greater than 0.";
    CHECK(parsedLine.size() == m_dataLength + m_labelLength)
      << "Input size and output size are not consistent with the training data line length.";
    std::vector<Dtype> data(parsedLine.begin(), parsedLine.begin()+m_dataLength);
    std::vector<Dtype> label(parsedLine.begin()+m_dataLength, parsedLine.end());
    lines_.push_back(std::make_pair(data, label));
  }
  CHECK(!lines_.empty()) << "File is empty " << source;

  // shuffle data
  if(this->layer_param_.queue_prediction_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleData();
  } else {
    LOG(INFO) << "Data not shuffled";
  }
  LOG(INFO) << "A total of " << lines_.size() << " data line were read.";

  lines_id_ = 0;
  // reshape top blob and label
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  std::vector<int> topShape({ int(batch_size), static_cast<int>(m_dataLength), 1, 1});
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(topShape);
  }
  top[0]->Reshape(topShape);

  std::vector<int> labelShape({int(batch_size), static_cast<int>(m_labelLength)});
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(labelShape);
  }
  top[1]->Reshape(labelShape);
}

template <typename Dtype>
void QueuePredictionDataLayer<Dtype>::ShuffleData() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void QueuePredictionDataLayer<Dtype>::load_batch(caffe::Batch<Dtype> *batch)
{
  CPUTimer batch_timer;
  batch_timer.Start();

  CHECK(batch->data_.count());

  // prepare the next batch
  const uint32_t batch_size = this->layer_param_.queue_prediction_data_param().batch_size();

  // reshape batch data and label
  std::vector<int> topShape({int(batch_size), static_cast<int>(m_dataLength), 1, 1});
  batch->data_.Reshape(topShape);

  std::vector<int> labelShape({int(batch_size), static_cast<int>(m_labelLength)});
  batch->label_.Reshape(labelShape);

  // load data and label into the batch
  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  const size_t lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id)
  {
    CHECK_GT(lines_size, lines_id_);
    auto const &current = lines_[lines_id_];

    int dataOffset = batch->data_.offset(item_id);
    std::copy(current.first.begin(), current.first.end(), prefetch_data + dataOffset);

    int labelOffset = batch->label_.offset(item_id);
    std::copy(current.second.begin(), current.second.end(), prefetch_label + labelOffset);

    // go to the next iter
    lines_id_++;
    if(lines_id_ >= lines_size)
    {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if(this->layer_param_.image_data_param().shuffle())
        ShuffleData();
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
}


INSTANTIATE_CLASS(QueuePredictionDataLayer);
REGISTER_LAYER_CLASS(QueuePredictionData);


} // namespace ultinous
} // namespace caffe
