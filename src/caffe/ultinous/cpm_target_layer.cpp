#include <caffe/ultinous/cpm_target_layer.hpp>
#include <boost/range.hpp>

namespace caffe
{
namespace ultinous 
{

template<typename Dtype>
void CPMTargetLayer<Dtype>::LayerSetUp(vector<Blob<Dtype>*> const &bottom, vector<Blob<Dtype>*> const &top)
{
  m_positive_threshold = m_param.positive_threshold();
  m_target_pixel_count = m_param.target_pixel_count();
  m_foreground_fraction = m_param.foreground_fraction();
  m_hnm_threshold = m_param.hnm_threshold();
}

template<typename Dtype>
void CPMTargetLayer<Dtype>::Reshape(vector<Blob<Dtype>*> const &bottom, vector<Blob<Dtype>*> const &top)
{
  CHECK_EQ(bottom[0]->shape_string(), bottom[1]->shape_string());
  if (bottom.size() > 2)
    CHECK_EQ(bottom[0]->shape_string(), bottom[2]->shape_string());

  top[0]->Reshape(bottom[0]->shape());
}

template<typename Dtype>
void CPMTargetLayer<Dtype>::Forward_cpu(vector<Blob<Dtype> *> const &bottom, vector<Blob<Dtype> *> const &top)
{
  auto const * const label_data = bottom[0]->cpu_data();
  auto const * const predicted_data = bottom[1]->cpu_data();
  auto const * const mask_data = (bottom.size() > 2)? bottom[2]->cpu_data() : nullptr;
  auto * const output_data = top[0]->mutable_cpu_data();

  std::vector<std::size_t> foreground;
  std::vector<std::tuple<Dtype, std::size_t>> background;

  auto const image_size = bottom[0]->count(1);
  auto const channel_size = bottom[0]->count(2);  
  for (std::size_t image = 0; image < bottom[0]->shape(0); ++image)
  {
    for (std::size_t channel = 0; channel < bottom[0]->shape(1); ++channel)
    {
      auto const * const label = label_data + (image * image_size) + (channel * channel_size);
      auto const * const predicted = predicted_data + (image * image_size) + (channel * channel_size);
      auto const * const mask = (mask_data)? mask_data + (image * image_size) + (channel * channel_size) : nullptr;
      auto * const output = output_data + (image * image_size) + (channel * channel_size);

      std::fill(output, output+channel_size, Dtype(0));
      foreground.clear();
      background.clear();
      for (std::size_t i = 0; i < channel_size; ++i)
      {
        if (mask && std::abs(mask[i]) < 1e-10) // special true negative
          continue;

        if (label[i] > m_positive_threshold) // true positive + false negative      
          foreground.emplace_back(i);
        else // true negative + false positive
          background.emplace_back(predicted[i], i);
      }


      auto const target_fg_count = (m_target_pixel_count > 0)? 
        static_cast<std::uint32_t>(m_target_pixel_count * m_foreground_fraction) : foreground.size();

      if (target_fg_count < foreground.size())
      {
        std::random_shuffle(foreground.begin(), foreground.end());
        foreground.erase(std::next(foreground.begin(), target_fg_count), foreground.end());
      }

      // LOG(INFO) << "target fg count " << target_fg_count;
      // LOG(INFO) << "num_fg " << foreground.size();

      auto const target_bg_count = (m_target_pixel_count > 0)?
        (m_target_pixel_count - foreground.size())
          : static_cast<std::uint32_t>(foreground.size() * (1.0f - m_foreground_fraction) / m_foreground_fraction);

      std::sort(background.begin(), background.end(), std::greater<std::tuple<Dtype, std::size_t>>());

      for (std::size_t i = background.size()-1; i > 0; --i)
      {
        std::size_t j = rand() % (i + 1);
        if (m_hnm_threshold < (static_cast<Dtype>(rand()) / RAND_MAX))
          std::swap(background[i], background[j]);
      }

      if (target_bg_count < background.size())
        background.erase(background.begin()+target_bg_count, background.end());

      // LOG(INFO) << "target bg count " << target_bg_count;
      // LOG(INFO) << "num bg " << background.size();

      for (auto const &ind : foreground)
        output[ind] = Dtype(1.0);
            
      for (auto const &infos : background)
        output[std::get<1>(infos)] = Dtype(1.0);
    }
  }
}

INSTANTIATE_CLASS(CPMTargetLayer);
REGISTER_LAYER_CLASS(CPMTarget);

} // namespace ultinous
} // namespace caffe
