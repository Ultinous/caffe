#include <boost/thread.hpp>
#include <exception>

#include "caffe/internal_threads.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

InternalThreads::~InternalThreads() {
  StopInternalThreads();
}

bool InternalThreads::must_stop() {
  for (const auto& thread : threads_)
    if (thread && thread->interruption_requested())
      return true;
  return false;
}

void InternalThreads::StartInternalThreads() {
  bool is_any_started = false;
  for (const auto& thread : threads_)
    if (thread && thread->joinable()) {
      is_any_started = true;
      break;
    }
  CHECK(!is_any_started) << "There are already started threads.";

  int device = 0;
#ifndef CPU_ONLY
  CUDA_CHECK(cudaGetDevice(&device));
  cudaStream_t str = Caffe::cuda_stream();
  Caffe::SetDevice(device);
  Caffe::setCudaStream(str);
#endif
  Caffe::Brew mode = Caffe::mode();
  int rand_seed = caffe_rng_rand();
  int solver_count = Caffe::solver_count();
  int solver_rank = Caffe::solver_rank();
  bool multiprocess = Caffe::multiprocess();
  Caffe::set_mode(mode);
  Caffe::set_random_seed(rand_seed);
  Caffe::set_solver_count(solver_count);
  Caffe::set_solver_rank(solver_rank);
  Caffe::set_multiprocess(multiprocess);

  try {
    threads_ = std::vector<shared_ptr<boost::thread>>(worker_count_);
    for (size_t thread_index=0; thread_index<threads_.size(); ++thread_index)
      threads_[thread_index].reset(new boost::thread(&InternalThreads::InternalThreadEntry, this, thread_index));
  } catch (std::exception& e) {
    LOG(FATAL) << "Thread exception: " << e.what();
  }
}

void InternalThreads::StopInternalThreads() {
  for (auto& thread : threads_)
    if (thread && thread->joinable()) {
      try {
        thread->join();
      } catch (boost::thread_interrupted&) {
      } catch (std::exception& e) {
        LOG(FATAL) << "Thread exception: " << e.what();
      }
    }
}

}  // namespace caffe
