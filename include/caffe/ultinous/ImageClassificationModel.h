#pragma once

#include <string>
#include <vector>
#include <map>
#include <istream>
#include <fstream>
#include "caffe/util/rng.hpp"

namespace caffe
{
namespace ultinous
{

class ImageClassificationModel
{
public:
  typedef std::string ImageName; // e.g. filename
  typedef int ClassId;
  typedef size_t ImageIndex;
  typedef std::vector<ImageIndex> ImageIndexes;
public:
  class ClassModel
  {
  public:
    static const ClassId INVALID_CLASSID = ClassId(-1);
  public:
    ClassModel(ClassId _classId = INVALID_CLASSID)
      : classId(_classId)
    {}
  public:
    ClassId classId;
    ImageIndexes images;
  };
public:
  typedef std::vector<ImageName> ImageNames;
  typedef std::vector<ClassModel> BasicModel;
public:
  const BasicModel& getBasicModel() const { return m_basicModel; }
  const ImageName& getImageName(ImageIndex index) const { return m_imageNames.at(index); }
  const size_t getImageNum() const { return m_imageNames.size(); }
  const size_t getImageClass(ImageIndex index) const
  {
    for( int i = 0; i < m_basicModel.size(); ++i )
      for( int j = 0; j < m_basicModel[i].images.size(); ++j )
        if( index == m_basicModel[i].images[j] )
          return m_basicModel[i].classId;
    throw std::exception( );
  }
public:
  void add(const ImageName& imageName, ClassId classId)
  {
    ImageIndex imageIndex = m_imageNames.size();
    m_imageNames.push_back(imageName);
    m_basicModel[ensureClass(classId)].images.push_back(imageIndex);
  }
private:
  typedef size_t ClassIndex;
  typedef std::map<ClassId, ClassIndex> ClassIds;
private:
  ClassIndex ensureClass(ClassId classId)
  {
    ClassIds::const_iterator it = m_classIds.find(classId);
    if(it == m_classIds.end())
    {
      it = m_classIds.insert(ClassIds::value_type(classId, m_basicModel.size())).first;
      m_basicModel.push_back(ClassModel(classId));
    }
    return it->second;
  }
private:
  ImageNames m_imageNames;
  BasicModel m_basicModel;
  ClassIds m_classIds;
};

class ImageClassificationModelShuffle
{
public:
  typedef ImageClassificationModel::BasicModel BasicModel;
public:
  ImageClassificationModelShuffle(const BasicModel& basicModel)
    : m_shuffledModel(basicModel)
  {}
public:
  const BasicModel& shuffledModel() const { return m_shuffledModel; }
public:
  void shuffleModel()
  {
    shuffleClasses();
    shufflePictures();
  }
  void shuffleClasses()
  {
    shuffle(m_shuffledModel.begin(), m_shuffledModel.end());
  }
  void shufflePictures()
  {
    for(size_t i = 0; i<m_shuffledModel.size(); ++i)
    {
      shuffle(m_shuffledModel[i].images.begin(), m_shuffledModel[i].images.end());
    }
  }
private:
  BasicModel m_shuffledModel;
};

class ImageSampler
{
public:
  typedef ImageClassificationModel::ImageIndexes ImageIndexes;
  typedef ImageClassificationModel::BasicModel BasicModel;
  typedef ImageClassificationModel::ClassModel ClassModel;
  typedef BasicModel Sample;
private:
  typedef size_t ClassIndex;
  typedef std::vector<ClassIndex> ClassIndexes;
  typedef std::set<ImageClassificationModel::ClassId> UsedClasses;
public:
  ImageSampler(const BasicModel& basicModel)
    : m_model(basicModel)
    , m_nextIndexes(basicModel.size(), 0)
    , m_nextClass(0)
  {
    reset();
  }
public:
  static void initSample(size_t numOfSampleClasses, size_t numOfSampleImagesPerClass, Sample& sample)
  {
    typedef ImageClassificationModel::ClassModel ClassModel;
    ClassModel classModel;
    classModel.images.resize(numOfSampleImagesPerClass);
    sample = Sample(numOfSampleClasses, classModel);
  }
public:
  void sample(Sample& sample)
  {
    ClassIndexes classIndexes(sample.size());
    UsedClasses usedClasses;
    size_t found = sampleClasses(classIndexes, 0, usedClasses);
    for(size_t i = 0; i<found; ++i)
    {
      sampleFromClass(classIndexes[i], sample[i]);
    }
    if(found < sample.size())
    {
      reset();
      size_t found2 = sampleClasses(classIndexes, found, usedClasses);
      CHECK_EQ( found2, sample.size() );
      for(size_t i = found; i<sample.size(); ++i)
      {
        sampleFromClass(classIndexes[i], sample[i]);
      }
    }
  }
public:
  void sampleFromClass(ClassIndex classIndex, ClassModel& targetClassModel)
  {
    const ClassModel& sourceClassModel = m_model.shuffledModel()[classIndex];
    const ImageIndexes& sourceImages = sourceClassModel.images;
    ImageIndexes& targetImages = targetClassModel.images;
    targetClassModel.classId = sourceClassModel.classId;

    size_t sourceIdx = m_nextIndexes[classIndex];
    for( size_t targetIdx = 0; targetIdx < targetImages.size(); ++targetIdx )
    {
       targetImages[targetIdx] = sourceImages[sourceIdx];
       sourceIdx = (1+sourceIdx) % sourceImages.size();
    }
    m_nextIndexes[classIndex] += targetImages.size();
  }
public:
  size_t sampleClasses(ClassIndexes& classes, size_t used, UsedClasses& usedClasses)
  {
    if( used >= classes.size() ) return used;
    for( size_t i = 0; i < m_nextIndexes.size() && used < classes.size(); ++i )
    {
      const ClassModel& classModel = m_model.shuffledModel()[m_nextClass];
      if(usedClasses.find(classModel.classId) == usedClasses.end()
        && m_nextIndexes[m_nextClass]<classModel.images.size())
      {
        classes[used] = m_nextClass;
        usedClasses.insert(classModel.classId);
        ++used;
      }
      m_nextClass = (1+m_nextClass) % m_nextIndexes.size();
    }
    return used;
  }
public:
  void reset()
  {
    m_model.shuffleModel();
    m_nextIndexes = ImageIndexes(m_model.shuffledModel().size(), 0);
    m_nextClass = 0;
  }
  const ImageClassificationModelShuffle& getModel() const {return m_model;}
private:
  ImageClassificationModelShuffle m_model;
  ImageIndexes m_nextIndexes;
  size_t m_nextClass;
};

inline
void read(std::istream& is, ImageClassificationModel& model)
{
  ImageClassificationModel::ImageName pictureName;
  ImageClassificationModel::ClassId classId;
  while(is >> pictureName >> classId)
  {
    model.add(pictureName, classId);
  }
}

inline
void read(const std::string& fname, ImageClassificationModel& model)
{
  std::ifstream is(fname.c_str());
  read(is, model);
}

} // namespace ultinous
} // namespace caffe
