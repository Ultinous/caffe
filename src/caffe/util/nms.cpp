#include <vector>


#include "caffe/util/nms.hpp"



namespace caffe
{

template <typename Dtype>
Dtype host_IOU(Dtype const * const a, Dtype const * const b)
{
  using namespace std;
  Dtype area_a = (a[2]-a[0]+1) * (a[3]-a[1]+1);
  Dtype area_b = (b[2]-b[0]+1) * (b[3]-b[1]+1);

  Dtype inter_x1 = max(a[0], b[0]);
  Dtype inter_y1 = max(a[1], b[1]);
  Dtype inter_x2 = min(a[2], b[2]);
  Dtype inter_y2 = min(a[3], b[3]);
  Dtype inter = max((Dtype)0, inter_x2 - inter_x1 + 1) * max((Dtype)0, inter_y2 - inter_y1 + 1);

  return inter / (area_a + area_b - inter);
}


template <typename Dtype>
int nms_cpu(const int& boxes_num, int* indexes, const Dtype* scores,const Dtype* proposals , const Dtype& threshold)
{
    std::vector<bool> keep((unsigned long) boxes_num, false);

    for(int i=0; i<boxes_num; ++i)
    {
        int j=0;
        while(j<boxes_num)
        {
            if(keep[j])
            {
                int i_ind = indexes[i];
                int j_ind = indexes[j];
                if(host_IOU<Dtype>(proposals+i_ind*4, proposals+j_ind*4) >= threshold)
                    break;
            }
            ++j;
        }
        if(j==boxes_num)
            keep[i]=true;
    }

    int result = 0;
    for(int i=0; i<boxes_num; ++i)
    {
        if(keep[i])
          indexes[result++] = indexes[i];
    }

    return result;
}


}
