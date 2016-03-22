#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/ultinous/triplet_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {
namespace ultinous {

template <typename Dtype>
TripletDataLayer<Dtype>::~TripletDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void TripletDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  hardTriplets = this->layer_param_.triplet_data_param().hardtriplets();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";


  // Read the file with filenames and labels
  const string& sourceFile = this->layer_param_.image_data_param().source();

  LOG(INFO) << "Opening file " << sourceFile;
  std::ifstream infile(sourceFile.c_str());
  string filename;
  int label;

  infile >> filename >> label;
  infile.close();
 /* while (infile >> filename >> label)
  {
    lines_.push_back(std::make_pair(filename, label));

    if( !hardTriplets ) {
      LabelList::iterator elem = std::find(labelList.begin(), labelList.end(), label);
      int labelIx = 0;
      if( elem == labelList.end() )
      {
        labelList.push_back( label );
        labelIx = labelList.size()-1;
        imageIndexListPerLabel.push_back( ImageIndexList() );
      } else
      {
        labelIx = std::distance(labelList.begin(), elem);
      }
      imageIndexListPerLabel[labelIx].push_back( lines_.size( )-1 );
    }
  }*/

  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder+filename, new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << filename;

  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);

  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  top_shape[1] *= 3;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();

  // label
  vector<int> label_shape(2);
  label_shape[0] = batch_size;
  label_shape[1] = 3;
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }


  // Init Classification Model
  string featureMapId = this->layer_param_.triplet_data_param().featuremapid();
  uint32_t sampledClasses = this->layer_param_.triplet_data_param().sampledclasses();
  uint32_t sampledPictures = this->layer_param_.triplet_data_param().sampledpictures();
  Dtype margin = this->layer_param_.triplet_data_param().margin();

  read( sourceFile, imageClassificationModel );

  if( hardTriplets )
  {
    hardTripletGenerator = HardTripletGeneratorPtr( new HardTripletGenerator<Dtype>( sampledClasses
        , sampledPictures, margin
        , imageClassificationModel.getBasicModel()
        , featureMapId ) );
  }
}

// This function is called on prefetch thread
template <typename Dtype>
void TripletDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();


  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + imageClassificationModel.getImageName(0),
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << imageClassificationModel.getImageName(0);
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  top_shape[1] *= 3;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  for (int item_id = 0; item_id < batch_size; ++item_id) {

    std::vector< std::string > files(3);
    std::vector< int > labels(3);

    if( hardTriplets )
    {
      //typename HardTripletGenerator<Dtype>::Triplet triplet =
      hardTripletGenerator->nextTriplet();
      //for( int i = 0; i < 3; ++ i ) {
      //  std::cout << triplet[i] << std::endl;
       /* int index = triplet[i];

        files[i] = imageClassificationModel.getImageName(index);
        labels[i] = index;*/
      //}

    } //else {
        const ImageClassificationModel::BasicModel& model = imageClassificationModel.getBasicModel( );

        int labIx1, labIx2;
        labIx1 = rand() % model.size();
        do{ labIx2 = rand() % model.size(); } while(labIx1==labIx2);
        int imageIxA, imageIxP, imageIxN;
        imageIxA = rand() % model[labIx1].images.size();
        do{ imageIxP = rand() % model[labIx1].images.size(); } while(imageIxA==imageIxP);
        imageIxN = rand() % model[labIx2].images.size();

        imageIxA = model[labIx1].images[imageIxA];
        imageIxP = model[labIx1].images[imageIxP];
        imageIxN = model[labIx2].images[imageIxN];

        files[0] = imageClassificationModel.getImageName( imageIxA );
        files[1] = imageClassificationModel.getImageName( imageIxP );
        files[2] = imageClassificationModel.getImageName( imageIxN );

        labels[0] = imageIxA;
        labels[1] = imageIxP;
        labels[2] = imageIxN;
    //}

    // get a blob
    for( int i = 0; i < 3; ++ i ) {
      cv::Mat cv_img = ReadImageToCVMat(root_folder + files[i],
          new_height, new_width, is_color);
      CHECK(cv_img.data) << "Could not load " << files[i];

      // Apply transformations (mirror, crop...) to the image
      int offset = batch->data_.offset(item_id, 3*i);
      this->transformed_data_.set_cpu_data(prefetch_data + offset);
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));

      prefetch_label[3*item_id+i] = labels[i];
    }
  }
}

INSTANTIATE_CLASS(TripletDataLayer);
REGISTER_LAYER_CLASS(TripletData);

}  // namespace ultinous
}  // namespace caffe
#endif  // USE_OPENCV
