#include <vector>
#include <boost/graph/graph_concepts.hpp>

#include "caffe/filler.hpp"
#include "caffe/ultinous/anchor_target_layer.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
#include <boost/filesystem.hpp>

namespace caffe {
  namespace ultinous {

    template<typename Dtype>
    void AnchorTargetLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                              const vector<Blob<Dtype> *> &top) {
      if (anchorTargetParam_.debug())
      {
        using namespace std::chrono;
        milliseconds ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
        std::stringstream ss;
        ss << ms.count();
        int current_device;
        CUDA_CHECK(cudaGetDevice(&current_device));
        folderName_ = "debug_images/anchor_target_layer_gpu" + std::to_string(current_device) + "_" + ss.str();
        boost::filesystem::create_directories(folderName_);
      }

      auto bottom_scores = bottom[0];
//  auto bottom_bbox  = bottom[1];
//  auto bottom_info  = bottom[2];
//  auto bottom_image = bottom[3];

      auto top_labels               = top[0];
      auto top_bbox_targets         = top[1];
      auto top_bbox_inside_weights  = top[2];
      auto top_bbox_outside_weights = top[3];

      float hard_negative_mining = anchorTargetParam_.hard_negative_mining();
      float hard_negative_mining_sigmoid_param = anchorTargetParam_.hard_negative_mining_sigmoid_param();
      CHECK(hard_negative_mining <= 1 && hard_negative_mining >= 0);
      CHECK(hard_negative_mining_sigmoid_param > 0);

      std::vector<Dtype> anchor_scales; // TODO: = layer_params.get('scales', (8, 16, 32))
      for (auto s : anchorTargetParam_.scales())
        anchor_scales.push_back(s);
      if (!anchor_scales.size())
        anchor_scales = std::vector<Dtype>({8, 16, 32});

      std::vector<Dtype> anchor_ratios;
      for (auto r : anchorTargetParam_.ratios())
        anchor_ratios.push_back(r);
      if (!anchor_ratios.size())
        anchor_ratios = std::vector<Dtype>({0.5, 1.0, 2.0});

      feat_stride_ = anchorTargetParam_.feat_stride();
      allowed_border_ = anchorTargetParam_.allowed_border();
      base_anchors_ = generate_anchors(anchor_scales, anchor_ratios, feat_stride_);

      int batch_size = bottom_scores->shape(0);
      int height     = bottom_scores->shape(2);
      int width      = bottom_scores->shape(3);

      top_labels->              Reshape(batch_size, 1, base_anchors_.size() * height, width);
      top_bbox_targets->        Reshape(batch_size, base_anchors_.size() * 4, height, width);
      top_bbox_inside_weights-> Reshape(batch_size, base_anchors_.size() * 4, height, width);
      top_bbox_outside_weights->Reshape(batch_size, base_anchors_.size() * 4, height, width);
    }

    template<typename Dtype>
    void AnchorTargetLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                               const vector<Blob<Dtype> *> &top) {
      auto bottom_scores = bottom[0];
      auto bottom_bbox   = bottom[1];
      auto bottom_info   = bottom[2];
//  auto bottom_image =  bottom[3];

      auto top_labels               = top[0];
      auto top_bbox_targets         = top[1];
      auto top_bbox_inside_weights  = top[2];
      auto top_bbox_outside_weights = top[3];

      int batch_size = bottom_scores->shape(0);
      int height     = bottom_scores->shape(2);
      int width      = bottom_scores->shape(3);

      if (anchorTargetParam_.debug())
      {
        string name;
        {
          using namespace std::chrono;
          milliseconds ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
          std::stringstream ss;
          ss << ms.count();
          name = ss.str();
        }

        auto bottom_image =  bottom[3];

        std::stringstream ss;

        std::vector<int> bb_shape = bottom_bbox->shape();
        std::vector<int> info_shape = bottom_info->shape();
        std::vector<int> image_shape = bottom_image->shape();

        size_t C = image_shape[1];
        size_t H = image_shape[2];
        size_t W = image_shape[3];
        size_t imageSize = C*H*W;

        int bb_offset = 0;

        for (int batch_index=0; batch_index < batch_size; ++batch_index)
        {
          std::string fileName = folderName_ + "/iter" + std::to_string(Caffe::iter()) + "_batch" + std::to_string(batch_index) + "_" + name + ".jpg";
          LOG(INFO) << fileName;

          std::vector<Dtype> info_vector;
          for (int i=0; i<info_shape[1]; ++i)
            info_vector.push_back( bottom_info->cpu_data()[bottom_info->offset( batch_index, i )] );

          Dtype* array( bottom_image->mutable_cpu_data() );
          std::vector<Dtype> converted(imageSize);
          for (int c=0;c<C;++c)
          for (int h=0;h<H;++h)
          for (int w=0;w<W;++w)
          {
            auto intensity = array[bottom_image->offset( batch_index, c, h, w )];
            converted[h*W*C+w*C+c] = intensity + 127;
          }
          Dtype* array2 = converted.data();
          cv::Mat cv_img( bottom_image->height(), bottom_image->width(), CV_32FC3, array2 );

          std::vector<std::vector<Dtype>> bb_vector;
          for (int i=0; i<(int)info_vector.back(); ++i)
            bb_vector.push_back(std::vector<Dtype>{
                bottom_bbox->cpu_data()[bottom_bbox->offset( bb_offset+i, 0 )],
                bottom_bbox->cpu_data()[bottom_bbox->offset( bb_offset+i, 1 )],
                bottom_bbox->cpu_data()[bottom_bbox->offset( bb_offset+i, 2 )],
                bottom_bbox->cpu_data()[bottom_bbox->offset( bb_offset+i, 3 )],
                bottom_bbox->cpu_data()[bottom_bbox->offset( bb_offset+i, 4 )],
            });
          bb_offset += (int)info_vector.back();

          for (auto v : bb_vector)
          {
            cv::Scalar color;
            bool mask = v[4] == 0;
            if (mask)
              color = cv::Scalar(255, 0, 0);
            else
              color = cv::Scalar(0, 0, 255);

            cv::rectangle(cv_img, cv::Point(v[0], v[1]), cv::Point(v[2], v[3]), color);
          }

          cv::rectangle(cv_img, cv::Point(info_vector[2],info_vector[3]), cv::Point(info_vector[4],info_vector[5]), cv::Scalar(0, 255, 0));
          cv::putText(cv_img,"bb num: " + std::to_string((int)info_vector.back()),cv::Point(100,100),cv::FONT_HERSHEY_COMPLEX_SMALL,2.0,cv::Scalar(0,0,255),2,CV_AA);
          cv::imwrite(fileName, cv_img);
        }
      }

      top_labels->              Reshape(batch_size, 1, base_anchors_.size() * height, width);
      top_bbox_targets->        Reshape(batch_size, base_anchors_.size() * 4, height, width);
      top_bbox_inside_weights-> Reshape(batch_size, base_anchors_.size() * 4, height, width);
      top_bbox_outside_weights->Reshape(batch_size, base_anchors_.size() * 4, height, width);

      Offset bottom_scores_offset(bottom_scores->shape());
      Offset bottom_bbox_offset(bottom_bbox->shape());
      Offset bottom_info_offset(bottom_info->shape());
//      Offset bottom_image_offset(bottom_image->shape());

      Offset top_labels_offset(top_labels->shape());
      Offset top_bbox_targets_offset(top_bbox_targets->shape());
      Offset top_bbox_inside_weights_offset(top_bbox_inside_weights->shape());
      Offset top_bbox_outside_weights_offset(top_bbox_outside_weights->shape());

      auto im_info       = bottom_info->cpu_data();
      auto data_gt_boxes = bottom_bbox->cpu_data();

      auto labels = top_labels->mutable_cpu_data();
      std::fill(labels, labels+top_labels_offset.count, Dtype(-1));

      auto bbox_targets = top_bbox_targets->mutable_cpu_data();
      std::fill(bbox_targets, bbox_targets+top_bbox_targets_offset.count, Dtype(0));

      auto bbox_inside_weights = top_bbox_inside_weights->mutable_cpu_data();
      std::fill(bbox_inside_weights, bbox_inside_weights+top_bbox_inside_weights_offset.count, Dtype(0));

      auto bbox_outside_weights = top_bbox_outside_weights->mutable_cpu_data();
      std::fill(bbox_outside_weights, bbox_outside_weights+top_bbox_outside_weights_offset.count, Dtype(0));

      float RPN_POSITIVE_OVERLAP  = anchorTargetParam_.positive_overlap();
      float RPN_NEGATIVE_OVERLAP  = anchorTargetParam_.negative_overlap();
      bool  RPN_CLOBBER_POSITIVES = anchorTargetParam_.clobber_positives();
      float RPN_FG_FRACTION       = anchorTargetParam_.fg_fraction();
      int   RPN_BATCHSIZE         = anchorTargetParam_.batchsize();

      CHECK(anchorTargetParam_.bbox_inside_weights().size() == 4 || anchorTargetParam_.bbox_inside_weights().size() == 0);
      std::vector<Dtype> RPN_BBOX_INSIDE_WEIGHTS;
      for (auto w : anchorTargetParam_.bbox_inside_weights())
        RPN_BBOX_INSIDE_WEIGHTS.push_back(w);
      if (anchorTargetParam_.bbox_inside_weights().size() == 0)
        RPN_BBOX_INSIDE_WEIGHTS = std::vector<Dtype>(4, 1.0);

      Dtype RPN_POSITIVE_WEIGHT = anchorTargetParam_.positive_weight();

      using Overlaps = typename AnchorTargetLayer<Dtype>::Overlaps;
      using Boxes = typename AnchorTargetLayer<Dtype>::Boxes;

      struct Shift {
        int x, y;
        Shift(int x_, int y_) : x(x_), y(y_) {}
      };

      int gt_box_offset = 0;
      for (int batch_index=0; batch_index < batch_size; ++batch_index)
      {
        // Create all anchors
        std::vector<Anchor> anchors;
        std::vector<Shift> anchors_shifts;
        std::vector<int> anchor_base_indices;

        for (int shift_y = 0; shift_y < height; ++shift_y)
        for (int shift_x = 0; shift_x < width; ++shift_x)
        for (int baseAnchorIx = 0; baseAnchorIx < base_anchors_.size(); ++baseAnchorIx)
        {
          Anchor anchor{base_anchors_[baseAnchorIx][0] + shift_x * feat_stride_,
                        base_anchors_[baseAnchorIx][1] + shift_y * feat_stride_,
                        base_anchors_[baseAnchorIx][2] + shift_x * feat_stride_,
                        base_anchors_[baseAnchorIx][3] + shift_y * feat_stride_};

          bool useAnchor;
          if (anchorTargetParam_.strict_border())
            useAnchor = (
                 anchor[0] >= im_info[bottom_info_offset( batch_index, 2 )] - allowed_border_
              && anchor[1] >= im_info[bottom_info_offset( batch_index, 3 )] - allowed_border_
              && anchor[2] <= im_info[bottom_info_offset( batch_index, 4 )] + allowed_border_
              && anchor[3] <= im_info[bottom_info_offset( batch_index, 5 )] + allowed_border_
            );
          else
            useAnchor = (
                 anchor[2] > im_info[bottom_info_offset( batch_index, 2 )] - allowed_border_
              && anchor[3] > im_info[bottom_info_offset( batch_index, 3 )] - allowed_border_
              && anchor[0] < im_info[bottom_info_offset( batch_index, 4 )] + allowed_border_
              && anchor[1] < im_info[bottom_info_offset( batch_index, 5 )] + allowed_border_
            );

          if (useAnchor)
          {
            anchors.push_back(anchor);
            anchors_shifts.push_back(Shift(shift_x, shift_y));
            anchor_base_indices.push_back(baseAnchorIx);
          }
        }

        // Load gt_boxes and gt_masks;
        Boxes gt_boxes;
        std::vector<bool> gt_masks;
        size_t num_gt_boxes = im_info[bottom_info_offset( batch_index, 6 )];

        for (size_t i = gt_box_offset; i < gt_box_offset+num_gt_boxes; ++i) {
          gt_boxes.push_back(Anchor{
            data_gt_boxes[bottom_bbox_offset(i,0)],
            data_gt_boxes[bottom_bbox_offset(i,1)],
            data_gt_boxes[bottom_bbox_offset(i,2)],
            data_gt_boxes[bottom_bbox_offset(i,3)]
          });
          bool mask = data_gt_boxes[bottom_bbox_offset(i,4)] == 0;
          gt_masks.push_back(mask);
        }

        gt_box_offset += num_gt_boxes;

        // Compute overlaps between anchors and gt_boxes
        Overlaps overlaps = bbox_overlaps(anchors, gt_boxes);

        // Select maximally overlapping gt_box for each anchor
        std::vector<size_t> anchor_argmax_overlaps(anchors.size(), 0);
        std::vector<Overlap> anchor_max_overlaps(anchors.size(), 0);
        for (size_t i = 0; i < anchors.size(); ++i) {
          Overlap max = 0;
          for (size_t j = 0; j < gt_boxes.size(); ++j)
            if (overlaps[i][j] > max) {
              max = overlaps[i][j];
              anchor_argmax_overlaps[i] = j;
            }
          anchor_max_overlaps[i] = max;
        }

        // Select maximally overlapping anchor for each gt_box
        std::vector<size_t> gt_argmax_overlaps(gt_boxes.size(), 0);
        std::vector<Overlap> gt_max_overlaps(gt_boxes.size(), 0);
        for (size_t j = 0; j < gt_boxes.size(); ++j) {
          Overlap max = 0;
          for (size_t i = 0; i < anchors.size(); ++i)
            if (overlaps[i][j] > max) {
              max = overlaps[i][j];
              gt_argmax_overlaps[j] = i;
            }
          gt_max_overlaps[j] = max;
        }
        // NOTE: Figure out the meaning of the following part
        //   original python line: gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
        gt_argmax_overlaps.clear();
        for (size_t i = 0; i < anchors.size(); ++i)
          for (size_t j = 0; j < gt_boxes.size(); ++j)
            if (std::find(gt_max_overlaps.begin(), gt_max_overlaps.end(), overlaps[i][j]) != gt_max_overlaps.end())
              gt_argmax_overlaps.push_back(i);

        // Compute labels
        if (!RPN_CLOBBER_POSITIVES) {
          // assign bg labels first so that positive labels can clobber them
          for (size_t i = 0; i < anchors.size(); ++i) {
            if (anchor_max_overlaps[i] <= RPN_NEGATIVE_OVERLAP) {
              Shift anchorShift = anchors_shifts[i];
              labels[top_labels_offset(batch_index) + anchor_base_indices[i] * (width * height) + anchorShift.y * width + anchorShift.x] = 0;
            }
          }
        }

        //fg label: for each gt, anchor with highest overlap
        for (int anchorIx : gt_argmax_overlaps) {
          Shift anchorShift = anchors_shifts[anchorIx];
          labels[top_labels_offset(batch_index) + anchor_base_indices[anchorIx] * (width * height) + anchorShift.y * width + anchorShift.x] = 1;
        }

        // fg label: above threshold IOU
        for (size_t i = 0; i < anchors.size(); ++i) {
          if (anchor_max_overlaps[i] >= RPN_POSITIVE_OVERLAP) {
            Shift anchorShift = anchors_shifts[i];
            labels[top_labels_offset(batch_index) + anchor_base_indices[i] * (width * height) + anchorShift.y * width + anchorShift.x] = 1;
          }
        }

        // assign bg labels last so that negative labels can clobber positives
        if (RPN_CLOBBER_POSITIVES) {
          // assign bg labels first so that positive labels can clobber them
          for (size_t i = 0; i < anchors.size(); ++i) {
            if (anchor_max_overlaps[i] <= RPN_NEGATIVE_OVERLAP) {
              Shift anchorShift = anchors_shifts[i];
              labels[top_labels_offset(batch_index) + anchor_base_indices[i] * (width * height) + anchorShift.y * width + anchorShift.x] = 0;
            }
          }
        }

        // mask marked boxes
        for (size_t i = 0; i < anchors.size(); ++i) {
          Shift anchorShift = anchors_shifts[i];
          Dtype label = labels[top_labels_offset(batch_index) + anchor_base_indices[i] * (width * height) +
                               anchorShift.y * width + anchorShift.x];

          bool mask = gt_masks[anchor_argmax_overlaps[i]];
          if (mask)
            labels[top_labels_offset(batch_index) + anchor_base_indices[i] * (width * height) + anchorShift.y * width +
                   anchorShift.x] = -1;
        }

        // subsample positive labels if we have too many
        int num_fg = RPN_FG_FRACTION * RPN_BATCHSIZE;

        num_fg = randomMining(1, RPN_BATCHSIZE, num_fg, top_labels_offset, labels, width, height, batch_index);

        int num_bg = (RPN_BATCHSIZE > 0)
                     ? (RPN_BATCHSIZE - num_fg)
                     : static_cast<int>( std::ceil(static_cast<float>(num_fg) * (1.0f - RPN_FG_FRACTION) / RPN_FG_FRACTION));

        // subsample negative labels if we have too many,
        Dtype const *scores = bottom_scores->cpu_data();

        switch(anchorTargetParam_.hard_negative_mining_type())
        {
          case AnchorTargetParameter_HnmType_CONSTANT:
            if (anchorTargetParam_.hard_negative_mining() == 0.0f)
              num_bg = randomMining(0, RPN_BATCHSIZE, num_bg, top_labels_offset, labels, width, height, batch_index);
            else
              num_bg = hardNegativeMining(num_bg, bottom_scores_offset, scores, top_labels_offset, labels, width, height, batch_index);
            break;

          case AnchorTargetParameter_HnmType_SIGMOID:
            num_bg = hardNegativeMining(num_bg, bottom_scores_offset, scores, top_labels_offset, labels, width, height, batch_index);
            break;
        }

        // At this point labels are ready :)

        // Computing bbox_targets
        for (size_t i = 0; i < anchors.size(); ++i) {
          Shift anchorShift = anchors_shifts[i];
          Dtype label = labels[top_labels_offset(batch_index) + anchor_base_indices[i] * (width * height) + anchorShift.y * width + anchorShift.x];
          if (label == 1)
          {
            Anchor ex = anchors[i];
            Anchor gt = gt_boxes[anchor_argmax_overlaps[i]];

            // bbox_transform
            float ex_width = ex[2] - ex[0] + 1.0;
            float ex_height = ex[3] - ex[1] + 1.0;
            float ex_ctr_x = ex[0] + 0.5 * ex_width;
            float ex_ctr_y = ex[1] + 0.5 * ex_height;

            float gt_width = gt[2] - gt[0] + 1.0;
            float gt_height = gt[3] - gt[1] + 1.0;
            float gt_ctr_x = gt[0] + 0.5 * gt_width;
            float gt_ctr_y = gt[1] + 0.5 * gt_height;

            float targets_dx = (gt_ctr_x - ex_ctr_x) / ex_width;
            float targets_dy = (gt_ctr_y - ex_ctr_y) / ex_height;
            float targets_dw = std::log(gt_width / ex_width);
            float targets_dh = std::log(gt_height / ex_height);

            bbox_targets[top_bbox_targets_offset(
                batch_index, 4 * anchor_base_indices[i],     anchorShift.y, anchorShift.x )] = targets_dx;

            bbox_targets[top_bbox_targets_offset(
                batch_index, 4 * anchor_base_indices[i] + 1, anchorShift.y, anchorShift.x )] = targets_dy;

            bbox_targets[top_bbox_targets_offset(
                batch_index, 4 * anchor_base_indices[i] + 2, anchorShift.y, anchorShift.x )] = targets_dw;

            bbox_targets[top_bbox_targets_offset(
                batch_index, 4 * anchor_base_indices[i] + 3, anchorShift.y, anchorShift.x )] = targets_dh;
          }
        }
        // At this point bbox_targets are ready :)


        // Computing bbox_inside_weights
        for (size_t i = 0; i < anchors.size(); ++i) {
          Shift anchorShift = anchors_shifts[i];
          Dtype label = labels[top_labels_offset(batch_index) + anchor_base_indices[i] * (width * height) + anchorShift.y * width + anchorShift.x];
          if (label == 1)
          {
            bbox_inside_weights[top_bbox_inside_weights_offset(
              batch_index, 4 * anchor_base_indices[i],     anchorShift.y, anchorShift.x )] = RPN_BBOX_INSIDE_WEIGHTS[0];

            bbox_inside_weights[top_bbox_inside_weights_offset(
              batch_index, 4 * anchor_base_indices[i] + 1, anchorShift.y, anchorShift.x )] = RPN_BBOX_INSIDE_WEIGHTS[1];

            bbox_inside_weights[top_bbox_inside_weights_offset(
              batch_index, 4 * anchor_base_indices[i] + 2, anchorShift.y, anchorShift.x )] = RPN_BBOX_INSIDE_WEIGHTS[2];

            bbox_inside_weights[top_bbox_inside_weights_offset(
              batch_index, 4 * anchor_base_indices[i] + 3, anchorShift.y, anchorShift.x )] = RPN_BBOX_INSIDE_WEIGHTS[3];
          }
        }
        // At this point bbox_inside_weights are ready :)


        // Computing bbox_outside_weights
        int num_examples = num_fg + num_bg;
        Dtype positive_weight;
        Dtype negative_weight;
        if (RPN_POSITIVE_WEIGHT < 0) {
          positive_weight = 1.0 / num_examples;
          negative_weight = 1.0 / num_examples;
        } else {
          CHECK((RPN_POSITIVE_WEIGHT > 0) && (RPN_POSITIVE_WEIGHT < 1));
          positive_weight = (RPN_POSITIVE_WEIGHT / num_fg);
          negative_weight = ((1.0 - RPN_POSITIVE_WEIGHT) / num_bg);
        }

        for (size_t i = 0; i < anchors.size(); ++i) {
          Shift anchorShift = anchors_shifts[i];
          Dtype label = labels[top_labels_offset(batch_index) + anchor_base_indices[i] * (width * height) + anchorShift.y * width + anchorShift.x];
          if (label == 1 || label == 0) {
            Dtype weight = label == 1 ? positive_weight : negative_weight;

            bbox_outside_weights[top_bbox_outside_weights_offset(
              batch_index, 4 * anchor_base_indices[i],     anchorShift.y, anchorShift.x )] = weight;

            bbox_outside_weights[top_bbox_outside_weights_offset(
              batch_index, 4 * anchor_base_indices[i] + 1, anchorShift.y, anchorShift.x )] = weight;

            bbox_outside_weights[top_bbox_outside_weights_offset(
              batch_index, 4 * anchor_base_indices[i] + 2, anchorShift.y, anchorShift.x )] = weight;

            bbox_outside_weights[top_bbox_outside_weights_offset(
              batch_index, 4 * anchor_base_indices[i] + 3, anchorShift.y, anchorShift.x )] = weight;
          }
        }
      }
    }

    template<typename Dtype>
    uint32_t
    AnchorTargetLayer<Dtype>::hardNegativeMining(uint32_t num_bg,
                                                 Offset &bottom_scores_offset, Dtype const * const scores,
                                                 Offset &top_labels_offset,    Dtype * const labels,
                                                 const uint32_t width, const uint32_t height, const int batch_index) {
      int num_bg_per_baseAnchor = static_cast<int>(
          std::ceil(static_cast<float>(num_bg)
                    / base_anchors_.size())
      );

      float denoise = anchorTargetParam_.denoise();
      float hnm = 0.0f;
      if (anchorTargetParam_.hard_negative_mining_type() == AnchorTargetParameter_HnmType_SIGMOID)
      { // the hard negative mining parameter follows a transformed sigmoid function
        int iter = Caffe::iter();
        int max_iter = Caffe::max_iter();
        float arg = anchorTargetParam_.hard_negative_mining_sigmoid_param() * iter / max_iter - anchorTargetParam_.hard_negative_mining_sigmoid_param() / 2.0f;
        hnm = 1.0f / (1.0f+expf(-arg));
      }
      else if (anchorTargetParam_.hard_negative_mining_type() == AnchorTargetParameter_HnmType_CONSTANT)
        hnm = anchorTargetParam_.hard_negative_mining();


      typedef std::pair<size_t, Dtype> IndexScorePair;
      typedef std::vector<IndexScorePair> IndexScorePairs;
      std::vector<IndexScorePairs> bg_inds(base_anchors_.size()); //bg_inds.reserve( num_bg );

      for (size_t anchorIx = 0; anchorIx < base_anchors_.size(); ++anchorIx)
      for (size_t spatialIx = 0; spatialIx < width * height; ++spatialIx)
      {
        size_t labelIx    = top_labels_offset(batch_index)    + anchorIx * width * height + spatialIx;
        size_t scoreIxNeg = bottom_scores_offset(batch_index) + anchorIx * width * height + spatialIx;
        size_t scoreIxPos = bottom_scores_offset(batch_index) + (base_anchors_.size() + anchorIx) * width * height + spatialIx;
        if (labels[labelIx] == 0) {
          Dtype scoreNeg = scores[scoreIxNeg];
          Dtype scorePos = scores[scoreIxPos];
          Dtype score = exp(scorePos) / (exp(scorePos) + exp(scoreNeg)); // softmax

          bg_inds[anchorIx].push_back(std::make_pair(labelIx, score));
        }
      }

      num_bg = 0;
      for (size_t anchorIx = 0; anchorIx < base_anchors_.size(); ++anchorIx) {
        if (bg_inds[anchorIx].size() > num_bg_per_baseAnchor) {
          std::sort(bg_inds[anchorIx].begin(), bg_inds[anchorIx].end(),
                    [](IndexScorePair const &p1, IndexScorePair const &p2) { return p1.second > p2.second; });

          if (denoise > 0.0f)
          {
            size_t anchorsToRemove( float( bg_inds[anchorIx].size() ) * denoise );
            if (anchorsToRemove > 0)
              bg_inds[anchorIx].erase( begin(bg_inds[anchorIx]), begin(bg_inds[anchorIx])+anchorsToRemove );
          }

          for (size_t ix1 = bg_inds[anchorIx].size() - 1; ix1 > 0; --ix1) {
            size_t ix2 = caffe_rng_rand() % (ix1 + 1);

            int r;
            caffe_rng_bernoulli(1, 1 - hnm, &r);
            if (r)
              std::swap(bg_inds[anchorIx][ix1], bg_inds[anchorIx][ix2]);
          }

          std::for_each(bg_inds[anchorIx].begin() + num_bg_per_baseAnchor, bg_inds[anchorIx].end(),
                        [labels](IndexScorePair const &p) { labels[p.first] = -1; });
          num_bg += num_bg_per_baseAnchor;
        } else
          num_bg += bg_inds[anchorIx].size();
      }

      return num_bg;
    }

    template<typename Dtype>
    uint32_t
    AnchorTargetLayer<Dtype>::randomMining(const int comparisonValue, const int RPN_BATCHSIZE, uint32_t num,
                                           Offset &top_labels_offset, Dtype * const labels,
                                           const uint32_t width, const uint32_t height, const int batch_index) {
      std::vector<size_t> inds;
      inds.reserve(num);
      for
      (
        size_t i = top_labels_offset(batch_index);
               i < top_labels_offset(batch_index) + base_anchors_.size() * width * height;
             ++i
      )
      {
        if (labels[i] == comparisonValue)
          inds.push_back(i);
      }

      if ((RPN_BATCHSIZE > 0 || !comparisonValue) && inds.size() > num) {
        std::random_shuffle(inds.begin(), inds.end());
        std::for_each(inds.begin() + num, inds.end(), [labels](size_t ix) { labels[ix] = -1; });
      } else
        num = inds.size();

      return num;
    }

    template<typename Dtype>
    typename AnchorTargetLayer<Dtype>::Overlaps
    AnchorTargetLayer<Dtype>::bbox_overlaps(typename AnchorTargetLayer<Dtype>::Boxes &boxes,
                                            typename AnchorTargetLayer<Dtype>::Boxes &query_boxes) {
      using Boxes = typename AnchorTargetLayer<Dtype>::Boxes;
      using Overlaps = typename AnchorTargetLayer<Dtype>::Overlaps;

      typename Boxes::size_type N = boxes.size();
      typename Boxes::size_type K = query_boxes.size();

      Overlaps overlaps(N, std::vector<Overlap>(K, 0));

      for (size_t k = 0; k < K; ++k) {
        Overlap box_area =
            (query_boxes[k][2] - query_boxes[k][0] + 1) *
            (query_boxes[k][3] - query_boxes[k][1] + 1);
        for (size_t n = 0; n < N; ++n) {
          Overlap iw =
              std::min(boxes[n][2], query_boxes[k][2]) -
              std::max(boxes[n][0], query_boxes[k][0]) + 1;
          if (iw > 0) {
            Overlap ih =
                std::min(boxes[n][3], query_boxes[k][3]) -
                std::max(boxes[n][1], query_boxes[k][1]) + 1;

            if (ih > 0) {
              Overlap ua = Overlap((boxes[n][2] - boxes[n][0] + 1)
                                   * (boxes[n][3] - boxes[n][1] + 1)) + box_area - iw * ih;

              overlaps[n][k] = iw * ih / ua;
            }
          }
        }
      }

      return overlaps;
    }

    INSTANTIATE_CLASS(AnchorTargetLayer);

    REGISTER_LAYER_CLASS(AnchorTarget);

  }   //namespace ultinous
}  // namespace caffe
