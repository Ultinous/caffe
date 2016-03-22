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
  void sample(Sample& s)
  {
    ClassIndexes classIndexes(m_model.shuffledModel().size());
    size_t found = sampleClasses(classIndexes, 0);
    for(size_t i = 0; i<found; ++i)
    {
      sampleFromClass(classIndexes[i], s[i]);
    }
    if(found < s.size())
    {
      reset();
    }
    sampleClasses(classIndexes, found); // assert == s.size()
    for(size_t i = found; i<s.size(); ++i)
    {
      sampleFromClass(classIndexes[i], s[i]);
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
    size_t targetIdx = 0;
    for(; targetIdx<targetImages.size() && sourceIdx < sourceImages.size(); ++targetIdx, ++sourceIdx)
    {
      targetImages[targetIdx] = sourceImages[sourceIdx];
    }
    while(targetIdx < targetImages.size())
    {
      for(sourceIdx = 0; targetIdx<targetImages.size() && sourceIdx < sourceImages.size(); ++targetIdx, ++sourceIdx)
      {
        targetImages[targetIdx] = sourceImages[sourceIdx];
      }
    }

    m_nextIndexes[classIndex] += targetImages.size();
  }
public:
  size_t sampleClasses(ClassIndexes& classes, size_t used)
  {
    typedef std::set<ClassIndex> UsedClasses;
    UsedClasses usedClasses;

    for(size_t i = 0; i<used; ++i)
    {
      usedClasses.insert(classes[i]);
    }

    size_t lastClass = m_nextClass;
    for(; m_nextClass<m_nextIndexes.size(); ++m_nextClass)
    {
      if(usedClasses.find(m_nextClass) != usedClasses.end())
        continue;
      if(m_nextIndexes[m_nextClass]<m_model.shuffledModel()[m_nextClass].images.size())
      {
        classes[used] = m_nextClass;
        usedClasses.insert(m_nextClass);
        ++used;
        if(used>=classes.size())
          return used;
      }
    }
    for(m_nextClass = 0; m_nextClass<lastClass; ++m_nextClass)
    {
      if(usedClasses.find(m_nextClass) != usedClasses.end())
        continue;
      if(m_nextIndexes[m_nextClass]<m_model.shuffledModel()[m_nextClass].images.size())
      {
        classes[used] = m_nextClass;
        usedClasses.insert(m_nextClass);
        ++used;
        if(used>=classes.size())
          return used;
      }
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
