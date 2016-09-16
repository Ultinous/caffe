#include <vector>
#include <boost/graph/graph_concepts.hpp>

#include "caffe/filler.hpp"
#include "caffe/layers/anchor_target_layer.hpp"

namespace caffe {

template <typename Dtype>
void AnchorTargetLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  std::vector<int> anchor_scales{8,16,32}; // TODO: = layer_params.get('scales', (8, 16, 32))
  base_anchors_ = generate_anchors(anchor_scales);

  feat_stride_ = 16; // TODO: = layer_params['feat_stride']
  allowed_border_ = 0;  // TODO: = layer_params.get('allowed_border', 0)

  int height = bottom[0]->shape(2);
  int width = bottom[0]->shape(3);

  // labels
  top[0]->Reshape(1, 1, base_anchors_.size() * height, width);
  // bbox_targets
  top[1]->Reshape(1, base_anchors_.size() * 4, height, width);
  // bbox_inside_weights
  top[2]->Reshape(1, base_anchors_.size() * 4, height, width);
  // bbox_outside_weights
  top[3]->Reshape(1, base_anchors_.size() * 4, height, width);
}

template <typename Dtype>
void AnchorTargetLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK(bottom[0]->shape(0) == 1) << "Only single item batches are supported";

  int height = bottom[0]->shape(2);
  int width = bottom[0]->shape(3);

  // labels
  top[0]->Reshape(1, 1, base_anchors_.size() * height, width);
  // bbox_targets
  top[1]->Reshape(1, base_anchors_.size() * 4, height, width);
  // bbox_inside_weights
  top[2]->Reshape(1, base_anchors_.size() * 4, height, width);
  // bbox_outside_weights
  top[3]->Reshape(1, base_anchors_.size() * 4, height, width);


  Dtype const * const im_info = bottom[2]->cpu_data();

  using Overlaps = typename AnchorTargetLayer<Dtype>::Overlaps;
  using Boxes = typename AnchorTargetLayer<Dtype>::Boxes;

  struct Shift
  {
    int x, y;
    Shift( int x_, int y_ ) : x(x_), y(y_) { }
  };

  // Create all anchors
  std::vector< Anchor > anchors;
  std::vector< Shift > anchors_shifts;
  std::vector< int > anchor_base_indices;

  for( int shift_y = 0; shift_y < height; ++shift_y )
  {
    for( int shift_x = 0; shift_x < width; ++shift_x )
    {
      for( int baseAnchorIx = 0; baseAnchorIx < base_anchors_.size(); ++baseAnchorIx )
      {
        Anchor anchor{ base_anchors_[baseAnchorIx][0]+shift_x*feat_stride_,
          base_anchors_[baseAnchorIx][1]+shift_y*feat_stride_,
          base_anchors_[baseAnchorIx][2]+shift_x*feat_stride_,
          base_anchors_[baseAnchorIx][3]+shift_y*feat_stride_ };

        if( anchor[0] >= -allowed_border_ && anchor[1] >= -allowed_border_
          && anchor[2] < im_info[1]+allowed_border_ && anchor[3] < im_info[0]+allowed_border_)
        {
          anchors.push_back( anchor );
          anchors_shifts.push_back( Shift(shift_x, shift_y) );
          anchor_base_indices.push_back( baseAnchorIx );
        }
      }
    }
  }

  // Load gt_boxes;
  Boxes gt_boxes;
  size_t num_gt_boxes = bottom[1]->shape(0);
  Dtype const * const data_gt_boxes = bottom[1]->cpu_data();

  for( size_t i = 0; i < num_gt_boxes; ++i )
    gt_boxes.push_back(
        Anchor{ data_gt_boxes[bottom[1]->shape(1)*i]
                , data_gt_boxes[bottom[1]->shape(1)*i + 1]
                , data_gt_boxes[bottom[1]->shape(1)*i + 2]
                , data_gt_boxes[bottom[1]->shape(1)*i + 3] } );

  // Compute overlaps between anchors and gt_boxes
  Overlaps overlaps = bbox_overlaps(anchors, gt_boxes);

  // Select maximally overlapping gt_box for each anchor
  std::vector<size_t> anchor_argmax_overlaps(anchors.size(), 0);
  std::vector<Overlap> anchor_max_overlaps(anchors.size(), 0);
  for( size_t i = 0; i < anchors.size(); ++i )
  {
    Overlap max = 0;
    for( size_t j = 0; j < gt_boxes.size(); ++j )
      if( overlaps[i][j] > max )
      {
        max = overlaps[i][j];
        anchor_argmax_overlaps[i] = j;
      }
    anchor_max_overlaps[i] = max;
  }

  // Select maximally overlapping anchor for each gt_box
  std::vector<size_t> gt_argmax_overlaps(gt_boxes.size(), 0);
  std::vector<Overlap> gt_max_overlaps(gt_boxes.size(), 0);
  for( size_t j = 0; j < gt_boxes.size(); ++j )
  {
    Overlap max = 0;
    for( size_t i = 0; i < anchors.size(); ++i )
      if( overlaps[i][j] > max )
      {
        max = overlaps[i][j];
        gt_argmax_overlaps[j] = i;
      }
    gt_max_overlaps[j] = max;
  }
  // NOTE: Figure out the meagning of the following part
  //   original python line: gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
  gt_argmax_overlaps.clear( );
  for( size_t i = 0; i < anchors.size(); ++i )
    for( size_t j = 0; j < gt_boxes.size(); ++j )
      if( std::find( gt_max_overlaps.begin(), gt_max_overlaps.end(), overlaps[i][j] ) != gt_max_overlaps.end() )
        gt_argmax_overlaps.push_back(i);


  // Compute labels
  Dtype * labels = top[0]->mutable_cpu_data();
  std::fill( labels, labels + base_anchors_.size()*height*width, Dtype(-1) );

  float RPN_POSITIVE_OVERLAP = 0.7;
  float RPN_NEGATIVE_OVERLAP = 0.3;
  bool RPN_CLOBBER_POSITIVES = false;
  float RPN_FG_FRACTION = 0.5;
  int RPN_BATCHSIZE = 256;
  std::vector<Dtype> RPN_BBOX_INSIDE_WEIGHTS{1.0, 1.0, 1.0, 1.0};
  Dtype RPN_POSITIVE_WEIGHT = -1.0;

  if( !RPN_CLOBBER_POSITIVES )
  {
    // assign bg labels first so that positive labels can clobber them
    for( size_t i = 0; i < anchors.size(); ++i )
    {
      if( anchor_max_overlaps[i] < RPN_NEGATIVE_OVERLAP )
      {
        Shift anchorShift = anchors_shifts[i];
        labels[ anchor_base_indices[i] * (width*height) + anchorShift.y*width + anchorShift.x ] = 0;
      }
    }
  }

  //fg label: for each gt, anchor with highest overlap
  for( size_t i = 0; i < gt_argmax_overlaps.size(); ++i )
  {
    int anchorIx = gt_argmax_overlaps[i];
    Shift anchorShift = anchors_shifts[anchorIx];
    labels[ anchor_base_indices[anchorIx] * (width*height) + anchorShift.y*width + anchorShift.x ] = 1;
  }

  // fg label: above threshold IOU
  for( size_t i = 0; i < anchors.size(); ++i )
  {
    if( anchor_max_overlaps[i] >= RPN_POSITIVE_OVERLAP )
    {
      Shift anchorShift = anchors_shifts[i];
      labels[ anchor_base_indices[i] * (width*height) + anchorShift.y*width + anchorShift.x ] = 1;
    }
  }

  // assign bg labels last so that negative labels can clobber positives
  if( RPN_CLOBBER_POSITIVES )
  {
    // assign bg labels first so that positive labels can clobber them
    for( size_t i = 0; i < anchors.size(); ++i )
    {
      if( anchor_max_overlaps[i] < RPN_NEGATIVE_OVERLAP )
      {
        Shift anchorShift = anchors_shifts[i];
        labels[ anchor_base_indices[i] * (width*height) + anchorShift.y*width + anchorShift.x ] = 0;
      }
    }
  }

  // subsample positive labels if we have too many
  int num_fg = RPN_FG_FRACTION * RPN_BATCHSIZE;
  std::vector<size_t> fg_inds; fg_inds.reserve( num_fg );
  for( int i = 0; i < base_anchors_.size()*width*height; ++i )
    if( labels[i] == 1 ) fg_inds.push_back(i);
  if( fg_inds.size() > num_fg )
  {
    std::random_shuffle( fg_inds.begin(), fg_inds.end() );
    std::for_each( fg_inds.begin()+num_fg, fg_inds.end(), [labels](size_t ix){labels[ix]=-1;} );
  }
  else
    num_fg = fg_inds.size();

  // subsample negative labels if we have too many
  int num_bg = RPN_BATCHSIZE - num_fg;
  std::vector<size_t> bg_inds; bg_inds.reserve( num_bg );
  for( int i = 0; i < base_anchors_.size()*width*height; ++i )
    if( labels[i] == 0 ) bg_inds.push_back(i);
  if( bg_inds.size() > num_bg )
  {
    std::random_shuffle( bg_inds.begin(), bg_inds.end() );
    std::for_each( bg_inds.begin()+num_bg, bg_inds.end(), [labels](size_t ix){labels[ix]=-1;} );
  }
  else
    num_bg = bg_inds.size();
  // At this point labels are ready :)

  // Computing bbox_targets
  Dtype * bbox_targets = top[1]->mutable_cpu_data( );
  std::fill( bbox_targets, bbox_targets + 4*base_anchors_.size()*height*width, Dtype(0) );

  for( size_t i = 0; i < anchors.size(); ++i )
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

    Shift anchorShift = anchors_shifts[i];
    bbox_targets[ anchor_base_indices[i]*4*width*height
          + anchorShift.y*width + anchorShift.x ] = targets_dx;
    bbox_targets[ anchor_base_indices[i]*4*width*height + width*height
          + anchorShift.y*width + anchorShift.x ] = targets_dy;
    bbox_targets[ anchor_base_indices[i]*4*width*height + 2*width*height
          + anchorShift.y*width + anchorShift.x ] = targets_dw;
    bbox_targets[ anchor_base_indices[i]*4*width*height + 3*width*height
          + anchorShift.y*width + anchorShift.x ] = targets_dh;
  }
  // At this point bbox_targets are ready :)


  // Computing bbox_inside_weights
  Dtype * bbox_inside_weights = top[2]->mutable_cpu_data( );
  std::fill( bbox_inside_weights, bbox_inside_weights + 4*base_anchors_.size()*height*width, Dtype(0) );
  for( size_t i = 0; i < anchors.size(); ++i )
  {
    Shift anchorShift = anchors_shifts[i];
    if( labels[ anchor_base_indices[i] * (width*height) + anchorShift.y*width + anchorShift.x ] == 1)
    {
      bbox_inside_weights[ anchor_base_indices[i]*4*width*height
            + anchorShift.y*width + anchorShift.x ] = RPN_BBOX_INSIDE_WEIGHTS[0];
      bbox_inside_weights[ anchor_base_indices[i]*4*width*height+ width*height
            + anchorShift.y*width + anchorShift.x ] = RPN_BBOX_INSIDE_WEIGHTS[1];
      bbox_inside_weights[ anchor_base_indices[i]*4*width*height+ 2*width*height
            + anchorShift.y*width + anchorShift.x ] = RPN_BBOX_INSIDE_WEIGHTS[2];
      bbox_inside_weights[ anchor_base_indices[i]*4*width*height+ 3*width*height
            + anchorShift.y*width + anchorShift.x ] = RPN_BBOX_INSIDE_WEIGHTS[3];
    }
  }
  // At this point bbox_inside_weights are ready :)


  // Computing bbox_outside_weights
  Dtype * bbox_outside_weights = top[3]->mutable_cpu_data( );
  std::fill( bbox_outside_weights, bbox_outside_weights + 4*base_anchors_.size()*height*width, Dtype(0) );

  int num_examples = num_fg + num_bg;
  Dtype positive_weight;
  Dtype negative_weight;
  if( RPN_POSITIVE_WEIGHT < 0 )
  {
    positive_weight = 1.0 / num_examples;
    negative_weight = 1.0 / num_examples;
  }
  else
  {
    CHECK((RPN_POSITIVE_WEIGHT > 0) && (RPN_POSITIVE_WEIGHT < 1));
    positive_weight = (RPN_POSITIVE_WEIGHT / num_fg);
    negative_weight = ((1.0 - RPN_POSITIVE_WEIGHT) / num_bg);
  }

  for( size_t i = 0; i < anchors.size(); ++i )
  {
    Shift anchorShift = anchors_shifts[i];
    Dtype label = labels[ anchor_base_indices[i] * (width*height) + anchorShift.y*width + anchorShift.x ];
    if( label == 1 || label == 0 )
    {
      Dtype weight = label==1?positive_weight:negative_weight;
      bbox_outside_weights[ anchor_base_indices[i]*4*width*height
            + anchorShift.y*width + anchorShift.x ] = weight;
      bbox_outside_weights[ anchor_base_indices[i]*4*width*height + width*height
            + anchorShift.y*width + anchorShift.x ] = weight;
      bbox_outside_weights[ anchor_base_indices[i]*4*width*height+ 2*width*height
            + anchorShift.y*width + anchorShift.x ] = weight;
      bbox_outside_weights[ anchor_base_indices[i]*4*width*height+ 3*width*height
            + anchorShift.y*width + anchorShift.x ] = weight;
    }
  }

}

template <typename Dtype>
typename AnchorTargetLayer<Dtype>::Overlaps
AnchorTargetLayer<Dtype>::bbox_overlaps(  typename AnchorTargetLayer<Dtype>::Boxes& boxes, typename AnchorTargetLayer<Dtype>::Boxes& query_boxes )
{
  using Boxes = typename AnchorTargetLayer<Dtype>::Boxes;
  using Overlaps = typename AnchorTargetLayer<Dtype>::Overlaps;

  typename Boxes::size_type N = boxes.size();
  typename Boxes::size_type K = query_boxes.size();

  Overlaps overlaps( N, std::vector<Overlap>(K, 0) );

  for( int k = 0; k < K; ++k )
  {
    Overlap box_area =
        (query_boxes[k][2] - query_boxes[k][0] + 1) *
        (query_boxes[k][3] - query_boxes[k][1] + 1);
    for( int n = 0; n < N; ++n )
    {
      Overlap iw =
          std::min(boxes[n][2], query_boxes[k][2]) -
          std::max(boxes[n][0], query_boxes[k][0]) + 1;
      if(iw > 0)
      {
        Overlap ih =
            std::min(boxes[n][3], query_boxes[k][3]) -
            std::max(boxes[n][1], query_boxes[k][1]) + 1;

        if(ih > 0)
        {
          Overlap ua = Overlap((boxes[n][2] - boxes[n][0] + 1)
            * (boxes[n][3] - boxes[n][1] + 1)) + box_area - iw * ih ;

          overlaps[n][k] = iw * ih / ua;
        }
      }
    }
  }

  return overlaps;
}

INSTANTIATE_CLASS(AnchorTargetLayer);
REGISTER_LAYER_CLASS(AnchorTarget);

}  // namespace caffe
