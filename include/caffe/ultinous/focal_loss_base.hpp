#pragma once

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

#include <boost/range/combine.hpp>
#include <boost/range/algorithm/transform.hpp>
#include <boost/range/numeric.hpp>

namespace caffe {

template <typename Dtype, typename BaseLoss, typename DiffEstimator>
class FocalLossBase : public LossLayer<Dtype> {
public:
  explicit FocalLossBase(const LayerParameter& param)
    : LossLayer<Dtype>(param)
    , mAlpha(1.0)
    , mGamma(1.0)
  {}

  void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) override
  {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
  }

  void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) override
  {
    LossLayer<Dtype>::Reshape(bottom, top);
    CHECK_EQ(bottom[1]->channels(), bottom[0]->channels());
    CHECK_EQ(bottom[1]->height(), 1);
    CHECK_EQ(bottom[1]->width(), 1);
    mBaseBlob.Reshape({bottom[0]->shape(0), 1});
    mDiffBlob.Reshape({bottom[0]->shape(0), 1});
    mDiffBlobBack.Reshape(bottom[0]->shape());
  }

protected:
  void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) override
  {
    mBaseLoss.forward(bottom, mBaseBlob);
    mDiffEst.forward(bottom, mDiffBlob);

    auto blobRange = [](const Blob<Dtype>& blob) {
      return std::make_pair(blob.cpu_data(), blob.cpu_data() + blob.count());
    };
    auto mutableBlobRange = [](Blob<Dtype>& blob) {
      return std::make_pair(blob.mutable_cpu_data(), blob.mutable_cpu_data() + blob.count());
    };
    const auto diffRange = blobRange(mDiffBlob);
    const auto baseRange = mutableBlobRange(mBaseBlob);
    boost::transform(baseRange, diffRange, baseRange.first, [this](const Dtype& a, const Dtype& b) {
      return this->doForward(a, b);
    });
    top[0]->mutable_cpu_data()[0] = boost::accumulate(baseRange, Dtype(0.0)) / bottom[0]->shape(0);
  }

  void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) override
  {
    if (propagate_down[1])
    {
      LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0])
    {
      auto blobRange = [](const Blob<Dtype>& blob) {
        return std::make_pair(blob.cpu_data(), blob.cpu_data() + blob.count());
      };
      auto blobDiffRange = [](const Blob<Dtype>& blob) {
        return std::make_pair(blob.cpu_diff(), blob.cpu_diff() + blob.count());
      };
      auto mutableBlobDiffRange = [](Blob<Dtype>& blob) {
        return std::make_pair(blob.mutable_cpu_diff(), blob.mutable_cpu_diff() + blob.count());
      };

      mBaseLoss.backward(bottom, *bottom[0]);
      mDiffEst.backward(bottom, mDiffBlobBack);

      const auto scale = top[0]->cpu_diff()[0] / bottom[0]->shape(0);
      const auto f1 = blobRange(mDiffBlob);
      const auto f1d = blobDiffRange(mDiffBlobBack);
      const auto f2 = blobRange(mBaseBlob);
      const auto f2d = mutableBlobDiffRange(*bottom[0]);

      const auto stepSize = bottom[0]->shape(1);
      auto outputIt = f2d.first;
      auto f1dIt = f1d.first;
      auto f2dIt = f2d.first;
      for (const auto& fTuple : boost::combine(f1, f2))
      {
        const auto f1 = boost::get<0>(fTuple);
        const auto f2 = boost::get<1>(fTuple);
        const auto f1dRange = std::make_pair(f1dIt, f1dIt+stepSize);
        const auto f2dRange = std::make_pair(f2dIt, f2dIt+stepSize);
        const auto combined = boost::combine(f1dRange, f2dRange);
        boost::transform(combined, outputIt, [this,&scale,f1,f2](const typename decltype(combined)::value_type& a) {
          const auto f1d = boost::get<0>(a);
          const auto f2d = boost::get<1>(a);
          return scale * this->doBackward(f1, f1d, f2, f2d);
        });

        f1dIt += stepSize;
        f2dIt += stepSize;
        outputIt += stepSize;
      }
    }
  }

private:
  Dtype mAlpha;
  Dtype mGamma;

  BaseLoss mBaseLoss;
  Blob<Dtype> mBaseBlob;

  DiffEstimator mDiffEst;

  Blob<Dtype> mDiffBlob;
  Blob<Dtype> mDiffBlobBack;

  inline Dtype doForward(const Dtype& a, const Dtype& b)
  {
    return mAlpha*a*pow(b, mGamma);
  }

  inline Dtype doBackward(const Dtype& f1, const Dtype& f1d, const Dtype& f2, const Dtype& f2d)
  {
    //return mAlpha * (mGamma * pow(f1, mGamma-1) * f1d * f2 + pow(f1, mGamma) * f2d);
    return mAlpha * pow(f1, mGamma-1) * (mGamma * f1d * f2 + f1 * f2d);
  }
};

struct SquaredDiff {
  template<typename DType>
  void forward(const vector<Blob<DType>*>& input, Blob<DType>& blob)
  {
    CHECK_EQ(input.size(), 2);

    auto blobRange = [](const Blob<DType>& blob) {
      return std::make_pair(blob.cpu_data(), blob.cpu_data() + blob.count());
    };
    auto mutableBlobRange = [](Blob<DType>& blob) {
      return std::make_pair(blob.mutable_cpu_data(), blob.mutable_cpu_data() + blob.count());
    };


    const auto firstInput = blobRange(*input[0]);
    const auto secondInput = blobRange(*input[1]);
    auto output = mutableBlobRange(blob);
    const auto dataSize = input[0]->count(1);

    auto outputIt = output.first;
    for (auto firstIt = firstInput.first, secondIt = secondInput.first; firstIt != firstInput.second; firstIt+=dataSize, secondIt+=dataSize, ++outputIt)
    {
      const auto first = std::make_pair(firstIt, firstIt+dataSize);
      const auto second = std::make_pair(secondIt, secondIt+dataSize);
      *outputIt = 0;
      for (const auto& values : boost::combine(first, second)) {
        const auto diff = boost::get<0>(values)-boost::get<1>(values);
        *outputIt += diff*diff;
      }
    }
  }

  template<typename DType>
  void backward(const vector<Blob<DType>*>& input, Blob<DType>& blob)
  {
    CHECK_EQ(input.size(), 2);

    auto blobRange = [](const Blob<DType>& blob) {
      return std::make_pair(blob.cpu_data(), blob.cpu_data() + blob.count());
    };
    auto mutableBlobDiffRange = [](Blob<DType>& blob) {
      return std::make_pair(blob.mutable_cpu_diff(), blob.mutable_cpu_diff() + blob.count());
    };

    const auto in_1 = blobRange(*input[0]);
    const auto in_2 = blobRange(*input[1]);
    auto output = mutableBlobDiffRange(blob);

    const auto dataSize = input[0]->count(1);
    auto outputIt = output.first;
    for (auto iter_1 = in_1.first, iter_2 = in_2.first; iter_1 != in_1.second; iter_1+=dataSize, iter_2+=dataSize, outputIt += dataSize)
    {
      const auto first = std::make_pair(iter_1, iter_1+dataSize);
      const auto second = std::make_pair(iter_2, iter_2+dataSize);
      boost::transform(first, second, outputIt, [](const DType& a, const DType& b) {
        return 2*(a-b);
      });
    }
  }
};

struct KLDivergence
{
  template<typename DType>
  void forward(const vector<Blob<DType>*>& input, Blob<DType>& blob)
  {
    CHECK_EQ(input.size(), 2);

    auto blobRange = [](const Blob<DType>& blob) {
      return std::make_pair(blob.cpu_data(), blob.cpu_data() + blob.count());
    };
    auto mutableBlobRange = [](Blob<DType>& blob) {
      return std::make_pair(blob.mutable_cpu_data(), blob.mutable_cpu_data() + blob.count());
    };


    const auto firstInput = blobRange(*input[0]);
    const auto secondInput = blobRange(*input[1]);
    auto output = mutableBlobRange(blob);
    const auto dataSize = input[0]->count(1);

    auto outputIt = output.first;
    for (auto firstIt = firstInput.first, secondIt = secondInput.first; firstIt != firstInput.second; firstIt+=dataSize, secondIt+=dataSize, ++outputIt)
    {
      const auto first = std::make_pair(firstIt, firstIt+dataSize);
      const auto second = std::make_pair(secondIt, secondIt+dataSize);
      *outputIt = 0;
      for (const auto& values : boost::combine(first, second)) {
        const auto prob1 = std::max(boost::get<0>(values), DType(kLOG_THRESHOLD));
        const auto prob2 = std::max(boost::get<1>(values), DType(kLOG_THRESHOLD));
        *outputIt += prob2 * log(prob2/prob1);
      }
    }
  }

  template<typename DType>
  void backward(const vector<Blob<DType>*>& input, Blob<DType>& blob)
  {
    CHECK_EQ(input.size(), 2);

    auto blobRange = [](const Blob<DType>& blob) {
      return std::make_pair(blob.cpu_data(), blob.cpu_data() + blob.count());
    };
    auto mutableBlobDiffRange = [](Blob<DType>& blob) {
      return std::make_pair(blob.mutable_cpu_diff(), blob.mutable_cpu_diff() + blob.count());
    };

    const auto in_1 = blobRange(*input[0]);
    const auto in_2 = blobRange(*input[1]);
    auto output = mutableBlobDiffRange(blob);

    const auto dataSize = input[0]->count(1);
    auto outputIt = output.first;
    for (auto iter_1 = in_1.first, iter_2 = in_2.first; iter_1 != in_1.second; iter_1+=dataSize, iter_2+=dataSize, outputIt += dataSize)
    {
      const auto first = std::make_pair(iter_1, iter_1+dataSize);
      const auto second = std::make_pair(iter_2, iter_2+dataSize);
      boost::transform(first, second, outputIt, [](const DType& a, const DType& b) {
        const auto prob1 = std::max(a, DType(kLOG_THRESHOLD));
        const auto prob2 = std::max(b, DType(kLOG_THRESHOLD));
        return -prob2/prob1;
      });
    }
  }
};
} // namespace caffe
