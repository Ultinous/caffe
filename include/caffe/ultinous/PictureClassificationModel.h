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

class PictureClassificationModel
{
public:
  typedef std::string PictureName;
  typedef int ClassId;
  typedef size_t PictureIndex;
  typedef size_t ClassIndex;
public:
  typedef std::vector<PictureName> PictureNames;
  typedef std::vector<PictureIndex> PictureIndexes;
  typedef std::vector<ClassId> ClassIds;
  typedef std::vector<PictureIndexes> PicturesOfClasses;
  typedef std::map<ClassId, ClassIndex> ClassIndexing;
public:
  const PicturesOfClasses& getModel()
  {
    return m_picturesOfClasses;
  }
  ClassId getClass(ClassIndex index) const
  {
    return m_classIds[index];
  }
  const PictureName& getPicture(PictureIndex index) const
  {
    return m_pictureNames[index];
  }
public:
  void add(const PictureName& pictureName, ClassId classId)
  {
    PictureIndex pictureIndex = m_pictureNames.size();
    m_pictureNames.push_back(pictureName);
    m_classIds.push_back(classId);
    ClassIndexing::const_iterator it = m_classIndexing.find(classId);
    if(it == m_classIndexing.end())
    {
      ClassIndex classIndex = m_picturesOfClasses.size();
      m_picturesOfClasses.push_back(PictureIndexes());
      it = m_classIndexing.insert(ClassIndexing::value_type(classId, classIndex)).first;
    }
    m_picturesOfClasses[it->second].push_back(pictureIndex);
  }
private:
  PictureNames m_pictureNames;
  ClassIds m_classIds;
  ClassIndexing m_classIndexing;
  PicturesOfClasses m_picturesOfClasses;
};

class PictureClassificationModelShuffle
{
public:
  typedef PictureClassificationModel::PicturesOfClasses PicturesOfClasses;
  typedef std::vector<PictureClassificationModel::ClassIndex> ClassShuffle;
public:
  PictureClassificationModelShuffle(const PicturesOfClasses& pictures)
    : m_picturesOfClasses(pictures)
    , m_classShuffle(pictures.size(), 0)
  {
    for(size_t i = 0; i<m_classShuffle.size(); ++i)
      m_classShuffle[i] = i;
  }
public:
  const ClassShuffle& classShuffle() const { return m_classShuffle; }
  const PicturesOfClasses& picturesShuffles() const { return m_picturesOfClasses; }
public:
  void shuffleModel()
  {
    shuffleClasses();
    shufflePictures();
  }
  void shuffleClasses()
  {
    shuffle(m_classShuffle.begin(), m_classShuffle.end());
  }
  void shufflePictures()
  {
    for(size_t i = 0; i<m_picturesOfClasses.size(); ++i)
    {
      shuffle(m_picturesOfClasses[i].begin(), m_picturesOfClasses[i].end());
    }
  }
private:
  PicturesOfClasses m_picturesOfClasses;
  ClassShuffle m_classShuffle;
};

class PictureSampler
{
public:
  typedef PictureClassificationModel::PictureIndexes PictureIndexes;
  typedef PictureClassificationModel::PicturesOfClasses PicturesOfClasses;
  typedef std::vector<PictureClassificationModel::ClassIndex> ClassIndexes;
public:
  PictureSampler(const PicturesOfClasses& pictures)
    : m_model(pictures)
    , m_nextIndexes(pictures.size(), 0)
    , m_nextClass(0)
  {
    reset();
  }
public:
  class Sample
  {
  public:
    Sample(size_t classes, size_t pictures)
      : m_pictures(classes, PicturesOfClasses::value_type(pictures, 0))
      , m_classes(classes)
    {}
  public:
    PicturesOfClasses m_pictures;
    ClassIndexes m_classes;
  };
public:
  void sample(Sample& s)
  {
    size_t found = sampleClasses(s.m_classes, 0);
    for(size_t i = 0; i<found; ++i)
    {
      sampleIndexes(s.m_classes[i], s.m_pictures[i]);
    }
    if(found < s.m_classes.size())
    {
      reset();
    }
    sampleClasses(s.m_classes, found); // assert == s.m_classes.size()
    for(size_t i = found; i<s.m_classes.size(); ++i)
    {
      sampleIndexes(s.m_classes[i], s.m_pictures[i]);
    }
  }
public:
  void sampleIndexes(size_t classIndex, PictureIndexes& pictures)
  {
    size_t pictureIndex = m_nextIndexes[classIndex];
    m_nextIndexes[classIndex] += pictures.size();
    const PictureIndexes& basePictures = m_model.picturesShuffles[classIndex];
    size_t k = 0;
    for(; k<pictures.size() && pictureIndex < basePictures.size(); ++pictureIndex, ++k)
    {
      pictures[k] = basePictures[pictureIndex];
    }
    while(k < pictures.size())
    {
      for(pictureIndex = 0; k<pictures.size() && pictureIndex < basePictures.size(); ++pictureIndex, ++k)
      {
        pictures[k] = basePictures[pictureIndex];
      }
    }
  }
public:
  size_t sampleClasses(ClassIndexes& classes, size_t used)
  {
    typedef std::set<size_t> UsedClasses;
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
      if(m_nextIndexes[m_nextClass]<m_model.picturesShuffles()[m_nextClass].size())
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
      if(m_nextIndexes[m_nextClass]<m_model.picturesShuffles()[m_nextClass].size())
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
    m_nextIndexes = PictureIndexes(m_model.classShuffle().size(), 0);
    m_nextClass = 0;
  }
private:
  typedef PictureClassificationModel::PictureIndexes PictureIndexes;
private:
  PictureClassificationModelShuffle m_model;
  PictureIndexes m_nextIndexes;
  size_t m_nextClass;
};

inline
void read(std::istream& is, PictureClassificationModel& model)
{
  PictureClassificationModel::PictureName pictureName;
  PictureClassificationModel::ClassId classId;
  while(is >> pictureName >> classId)
  {
    model.add(pictureName, classId);
  }
}

inline
void read(const std::string& fname, PictureClassificationModel& model)
{
  std::ifstream is(fname.c_str());
  read(is, model);
}

} // namespace ultinous
} // namespace caffe
