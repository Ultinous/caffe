#include <vector>
#include <algorithm>
#include <fstream>

#include "glog/logging.h"
#include <boost/concept_check.hpp>

#include "caffe/util/anchors.hpp"

#include "caffe/layers/proposal_layer.hpp"

namespace caffe{

template <typename Dtype>
void ProposalLayer<Dtype>::LayerSetUp( const vector< Blob< Dtype >* > &bottom,
                                const vector< Blob< Dtype >* > &top )
{
  ProposalParameter proposal_param = this->layer_param_.proposal_param();
  
  if(proposal_param.scales_size()!=0)
  {
    std::vector<int> anchor_scales;
    for(size_t i=0; i<proposal_param.scales_size(); ++i)
      anchor_scales.push_back(proposal_param.scales(i));
    m_base_anchors = generate_anchors(anchor_scales);
  }
  else
  {
    m_base_anchors = generate_anchors();
  }
  
//   m_base_anchors.Reshape(std::vector<int>({(int)anchors.size(), (int)anchors[0].size()}));
//   for(std::size_t i=0; i < anchors.size(); ++i)
//     for(std::size_t j=0; j< anchors[i].size(); ++j)
//       m_base_anchors.mutable_cpu_data()[m_base_anchors.offset(i,j)] = anchors[i][j];
  
  CHECK(bottom.size()==3 || bottom.size()==4);
  m_num_anchors = m_base_anchors.size();
  m_feat_stride = proposal_param.feat_stride();
  m_nms_thresh = proposal_param.nms_thresh();
  m_pre_nms_topN = proposal_param.pre_nms_topn();
  m_post_nms_topN = proposal_param.post_nms_topn();
  m_min_size = proposal_param.min_size();
  
  CHECK(top.size()==2 || top.size()==1);
  top[0]->Reshape(std::vector<int>({1,5}));
  if(top.size() == 2)
    top[1]->Reshape(std::vector<int>({1,1,1,1}));
}

template <typename Dtype>
void ProposalLayer<Dtype>::Reshape( const vector< Blob< Dtype >* > &bottom,
                             const vector< Blob< Dtype >* > &top )
{
  CHECK(bottom[0]->shape(0)==1); //TODO rewrite for batches
  
  int height = bottom[0]->shape(2) ;
  int width = bottom[0]->shape(3);
//   m_img_anchors = std::vector<Anchor>(); 
//   for(int i=0; i<height*width; ++i)
//   {
//   /**this is the shift generation matrix generations for anchors
//     *we do the shift directly on base anchors
//     *shifts.push_back(
//     *   std::vector<int>({(i%width)*m_feat_stride,
//     *                     (i/width)*m_feat_stride,
//     *                     (i%width)*m_feat_stride,
//     *                     (i/width)*m_feat_stride}));
//     *
//     *Then add every shift to every anchor. That will be the anchors on the pics.
//     */
//     for(int j=0; j < m_num_anchors ; ++j)
//     {
//       m_img_anchors.push_back(
//         std::vector<double>({m_base_anchors[j][0]+(i%width)*m_feat_stride,
//                         m_base_anchors[j][1]+(i/width)*m_feat_stride,
//                         m_base_anchors[j][2]+(i%width)*m_feat_stride,
//                         m_base_anchors[j][3]+(i/width)*m_feat_stride})
//       );
//     }
//   }

  m_anchors.Reshape(std::vector<int>( {height * width * m_num_anchors, 4}));  
  std::size_t ind = 0;
  for(int i=0; i<height*width; ++i)
    for(int j = 0; j < m_num_anchors; ++j)
    {
      m_anchors.mutable_cpu_data()[ind*4] = m_base_anchors[j][0]+(i%width)*m_feat_stride;
      m_anchors.mutable_cpu_data()[ind*4+1] = m_base_anchors[j][1]+(i/width)*m_feat_stride;
      m_anchors.mutable_cpu_data()[ind*4+2] = m_base_anchors[j][2]+(i%width)*m_feat_stride;
      m_anchors.mutable_cpu_data()[ind*4+3] = m_base_anchors[j][3]+(i/width)*m_feat_stride;
      ++ind;
    }
    
  m_proposals.ReshapeLike(m_anchors);
  
  std::size_t scores_num = bottom[0]->shape(2)*bottom[0]->shape(3)*bottom[0]->shape(1);
  m_scores.Reshape(std::vector<int>(1,scores_num-m_num_anchors));
  m_indexes.Reshape(std::vector<int>(1,scores_num-m_num_anchors));
  
  std::cout<<"Reshape end"<<std::endl;
  for(int i=0; i<scores_num-m_num_anchors;++i)
    m_indexes.mutable_cpu_data()[i]=i;
  
  top[0]->Reshape(std::vector<int>({static_cast<int>(m_post_nms_topN),5}));
}


template <typename Dtype>
void ProposalLayer<Dtype>::Forward_cpu( const vector< Blob< Dtype >* > &bottom,
                                 const vector< Blob< Dtype >* > &top )
{ 
  if(bottom.size()==4)
  {
    m_pre_nms_topN = bottom[3]->cpu_data()[bottom[3]->offset(0,0)];
    m_post_nms_topN = bottom[3]->cpu_data()[bottom[3]->offset(0,1)];
    m_nms_thresh =bottom[3]->cpu_data()[bottom[3]->offset(0,2)];
    m_min_size = bottom[3]->cpu_data()[bottom[3]->offset(0,3)];
  }
  
  //transponate the input blob (n,anchors_num*4,h,w) -> (n,h,w, anchors_num*4), then reshape (n,h*w*anchors_num,4) 
  //                                                                          where rows are ordered by (h, w, anchor_ind)
  const Dtype* bbox_deltas_blob =  bottom[1]->mutable_cpu_data();  
  Dtype* proposal_cpu = m_proposals.mutable_cpu_data();

  std::size_t ind_prop = 0; 
  for(int h=0; h<bottom[1]->shape(2); ++h)
    for(int w=0; w<bottom[1]->shape(3);++w)
      for(int ch=0; ch<bottom[1]->shape(1)/4;++ch)
      {
        //(h*bottom->shape(3)+w)*(bottom[1]->shape(1)/4)+ch;
        proposal_cpu[ind_prop*4] = bbox_deltas_blob[bottom[1]->offset(0,ch*4,h,w)];     //[0][ch*4][h][w],
        proposal_cpu[ind_prop*4+1] = bbox_deltas_blob[bottom[1]->offset(0,ch*4+1,h,w)];   //[0][ch*4+1][h][w],
        proposal_cpu[ind_prop*4+2] = bbox_deltas_blob[bottom[1]->offset(0,ch*4+2,h,w)];   //[0][ch*4+2][h][w],
        proposal_cpu[ind_prop*4+3] = bbox_deltas_blob[bottom[1]->offset(0,ch*4+3,h,w)];   //[0][ch*4+3][h][w]
        ++ind_prop;
      }
  //transponate the input blob (n,anchors_num,h,w) -> (n,h,w, anchors_num), then reshape (n,h*w*anchors_num,1) 
  //where rows are ordered by (h, w, anchor_ind)
  
  const Dtype* scores_blob = bottom[0]->mutable_cpu_data();
  Dtype* scores_cpu = m_scores.mutable_cpu_data();
  
  std::size_t scr_ind=0;//this is necessery too keep  
  for(int h=0; h<bottom[0]->shape(2); ++h)
    for(int w=0; w<bottom[0]->shape(3);++w)
      for(int ch=m_num_anchors; ch<bottom[0]->shape(1);++ch)
      {      
        scores_cpu[scr_ind++] = scores_blob[bottom[0]->offset(0,ch,h,w)];
      }

  bbox_transform_inv();

  clip_boxes(bottom[2]->cpu_data()[bottom[2]->offset(0,1)],bottom[2]->cpu_data()[bottom[2]->offset(0,0)]);
  
  Dtype min_size= m_min_size * bottom[2]->cpu_data()[bottom[2]->offset(0,2)];
  
//   std::vector<std::size_t> index_vec(m_proposals.size());
//   std::iota(index_vec.begin(), index_vec.end(), 0);
//   
//   index_vec.erase( std::remove_if(
//       index_vec.begin(),
//       index_vec.end(),
//       [&min_size, this](const std::size_t &i){ return m_proposals[i][2] - m_proposals[i][0]+1<min_size || m_proposals[i][3] - m_proposals[i][1]+1<min_size;}
//     ),
//     index_vec.end());

  
   std::vector<std::size_t> index_vec;
   for(int i=0; i<m_proposals.shape(0); ++i)
     if(proposal_cpu[i*4+2]-proposal_cpu[i*4]+1>=min_size && proposal_cpu[i*4+3]-proposal_cpu[i*4+1]+1>=min_size)
       index_vec.push_back(i);
  
  std::partial_sort(index_vec.begin(),index_vec.begin()+m_pre_nms_topN,index_vec.end(),
                    [scores_cpu](const std::size_t& a,const std::size_t& b){return scores_cpu[a]>scores_cpu[b];});

  if(index_vec.size()>m_pre_nms_topN)
    index_vec.resize(m_pre_nms_topN);

//   std::ofstream f("/home/jlocki/Desktop/cpp_full_pp.txt",std::ofstream::out | std::ofstream::app);
//   for(auto i=index_vec.begin(); i!=index_vec.end(); ++i)
//     f<<m_proposals[*i][0]<<"\t"<<m_proposals[*i][1]<<"\t"<<m_proposals[*i][2]<<"\t"<<m_proposals[*i][3]<<std::endl;
// 
//   

  index_vec=cpu_base_nms(index_vec);

  
  
  if(index_vec.size()>m_post_nms_topN)
    index_vec.resize(m_post_nms_topN);
  
  

  top[0]->Reshape(std::vector<int>({static_cast<int>(index_vec.size()),5}));
  for(int i=0; i<index_vec.size(); ++i){
    top[0]->mutable_cpu_data()[top[0]->offset(i,0)] = 0;
    top[0]->mutable_cpu_data()[top[0]->offset(i,1)] = proposal_cpu[index_vec[i]*4];
    top[0]->mutable_cpu_data()[top[0]->offset(i,2)] = proposal_cpu[index_vec[i]*4+1];
    top[0]->mutable_cpu_data()[top[0]->offset(i,3)] = proposal_cpu[index_vec[i]*4+2];
    top[0]->mutable_cpu_data()[top[0]->offset(i,4)] = proposal_cpu[index_vec[i]*4+3];
  }
  
  //TODO implement score out blob

}


template <typename Dtype>
std::vector<std::size_t> ProposalLayer<Dtype>::cpu_base_nms(
  const std::vector<std::size_t>& index_vec
)
{
  std::vector<std::size_t> keep;
  std::vector<double> areas;
  Dtype* proposals = m_proposals.mutable_cpu_data();
  
  keep.push_back(index_vec[0]);
  areas.push_back((proposals[keep[0]*4+2]-proposals[keep[0]*4]+1)*(proposals[keep[0]*4+3]-proposals[keep[0]*4+1]+1));
  
  for(size_t i=1; i<index_vec.size(); ++i)
  {
    int current=index_vec[i];
    double area_current=(proposals[current*4+2]-proposals[current*4]+1)*(proposals[current*4+3]-proposals[current*4+1]+1);
    
    size_t j=0;
    while(j<keep.size())
    {
      double inter_x1=std::max(proposals[keep[j]*4+0],proposals[current*4+0]);
      double inter_x2=std::min(proposals[keep[j]*4+2],proposals[current*4+2]);
      double inter_y1=std::max(proposals[keep[j]*4+1],proposals[current*4+1]);
      double inter_y2=std::min(proposals[keep[j]*4+3],proposals[current*4+3]);
      double w=std::max<double>(0, inter_x2-inter_x1+1);
      double h=std::max<double>(0, inter_y2-inter_y1+1);
      double inter = w*h;
      double overlap= inter / (areas[j]+area_current-inter);      
      if (overlap >= m_nms_thresh)
        break;
      ++j;
    }
    
    if(j==keep.size())
    {
      keep.push_back(current);
      areas.push_back(area_current);
    }
  }
  return keep;
}


template <typename Dtype>
void ProposalLayer<Dtype>::Backward_cpu( const vector< Blob< Dtype >* > &top,
                                  const vector< bool > &propagate_down,
                                  const vector< Blob< Dtype >* > &bottom )
{
  // This layer dont need backward 
}

template <typename Dtype>
void ProposalLayer<Dtype>::clip_boxes(const double& img_w, const double& img_h )
{
  Dtype* proposals_cpu = m_proposals.mutable_cpu_data();
  
  for(std::size_t i=0; i < m_proposals.shape(0); ++i)
  {
    proposals_cpu[i*4] = std::max<double>(std::min<double>(proposals_cpu[i*4],img_w-1),0);
    proposals_cpu[i*4+1] = std::max<double>(std::min<double>(proposals_cpu[i*4+1],img_h-1),0);
    proposals_cpu[i*4+2] = std::max<double>(std::min<double>(proposals_cpu[i*4+2],img_w-1),0);
    proposals_cpu[i*4+3] = std::max<double>(std::min<double>(proposals_cpu[i*4+3],img_h-1),0);
  }
}

template <typename Dtype>
void ProposalLayer<Dtype>::bbox_transform_inv()
{
  const Dtype* anchors = m_anchors.mutable_cpu_data();
  Dtype* proposals = m_proposals.mutable_cpu_data();
  
  for(std::size_t i = 0; i < m_proposals.shape(0); ++i)
  {
    double width = anchors[i*4+2] - anchors[i*4] + 1;
    double height = anchors[i*4+3] - anchors[i*4+1] + 1;;
    double ctr_x = anchors[i*4] + 0.5 * width;
    double ctr_y = anchors[i*4+1] + 0.5 * height;
    
    double pred_ctr_x = proposals[i*4] * width + ctr_x;
    double pred_ctr_y = proposals[i*4+1]* height + ctr_y;
    double pred_w = exp( proposals[i*4+2] ) * width;
    double pred_h = exp( proposals[i*4+3] ) * height;
    
    proposals[i*4] = pred_ctr_x - 0.5 * pred_w;
    proposals[i*4+1] = pred_ctr_y - 0.5 * pred_h;
    proposals[i*4+2] = pred_ctr_x + 0.5 * pred_w;
    proposals[i*4+3] = pred_ctr_y + 0.5 * pred_h;
  }

}

#ifdef CPU_ONLY
STUB_GPU(ProposalLayer);
#endif

INSTANTIATE_CLASS(ProposalLayer);
REGISTER_LAYER_CLASS(Proposal);
}//caffe