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
    const auto baseRange = blobRange(mBaseBlob);
    const auto diffRange = blobRange(mDiffBlob);
    Dtype result = 0.0;
    for (const auto& it : boost::combine(baseRange, diffRange))
    {
      const auto base = boost::get<0>(it);
      const auto difference = boost::get<1>(it);
      result += this->doForward(base, difference);
    }
    top[0]->mutable_cpu_data()[0] = result / bottom[0]->shape(0);
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
      const auto diffRange = blobRange(mDiffBlob);
      const auto diffDerRange = blobDiffRange(mDiffBlobBack);
      const auto baseRange = blobRange(mBaseBlob);
      const auto baseDerRange = mutableBlobDiffRange(*bottom[0]);

      const auto stepSize = bottom[0]->shape(1);
      auto outputIt = baseDerRange.first;
      auto diffDerIt = diffDerRange.first;
      auto baseDerIt = baseDerRange.first;
      for (const auto& fTuple : boost::combine(baseRange, diffRange))
      {
        const auto base = boost::get<0>(fTuple);
        const auto diff = boost::get<1>(fTuple);
        const auto diffDerRangeLoc = std::make_pair(diffDerIt, diffDerIt+stepSize);
        const auto baseDerRangeLoc = std::make_pair(baseDerIt, baseDerIt+stepSize);
        const auto combined = boost::combine(baseDerRangeLoc, diffDerRangeLoc);
        boost::transform(combined, outputIt, [this,&scale,diff,base](const typename decltype(combined)::value_type& a) {
          const auto baseDer = boost::get<0>(a);
          const auto diffDer = boost::get<1>(a);
          return scale * this->doBackward(base, baseDer, diff, diffDer);
        });

        diffDerIt += stepSize;
        baseDerIt += stepSize;
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

  inline Dtype doForward(const Dtype& base, const Dtype& difference)
  {
    return mAlpha*pow(difference, mGamma) * base;
  }

  inline Dtype doBackward(const Dtype& base, const Dtype& baseDer, const Dtype& diff, const Dtype& diffDer)
  {
    //return mAlpha * (mGamma * pow(diff, mGamma-1) * diffDer * base + pow(diff, mGamma) * baseDer);
    return mAlpha * pow(diff, mGamma-1) * (mGamma * diffDer * base + diff * baseDer);
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
    for (auto firstIt = firstInput.first, secondIt = secondInput.first; firstIt != firstInput.second; firstIt += dataSize, secondIt += dataSize, ++outputIt)
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
    for (auto iter_1 = in_1.first, iter_2 = in_2.first; iter_1 != in_1.second; iter_1 += dataSize, iter_2 += dataSize, outputIt += dataSize)
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
