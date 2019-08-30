#include "caffe/solver_factory.hpp"

namespace caffe
{

template<typename T>
typename SolverRegistry<T>::CreatorRegistry & SolverRegistry<T>::Registry()
{
  static SolverRegistry<T>::CreatorRegistry * g_registry_ = new SolverRegistry<T>::CreatorRegistry();
  return *g_registry_;
}

template SolverRegistry<float>::CreatorRegistry & SolverRegistry<float>::Registry();
template SolverRegistry<double>::CreatorRegistry & SolverRegistry<double>::Registry();

}
