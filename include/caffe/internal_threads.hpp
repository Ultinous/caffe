#ifndef CAFFE_INTERNAL_THREADS_HPP_
#define CAFFE_INTERNAL_THREADS_HPP_

#include "caffe/common.hpp"

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class thread; }

namespace caffe {

/**
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virtual function InternalThreadEntry.
 */
class InternalThreads {
 public:
  InternalThreads(int worker_count) : threads_(), worker_count_(worker_count) {}
  virtual ~InternalThreads();

  /**
   * Caffe's thread local state will be initialized using the current
   * thread values, e.g. device id, solver index etc. The random seed
   * is initialized using caffe_rng_rand.
   */
  void StartInternalThreads();

  /** Will not return until the internal thread has exited. */
  void StopInternalThreads();

 protected:
  /* Implement this method in your subclass
      with the code you want your thread to run. */
  virtual void InternalThreadEntry(size_t thread_index) {}

  /* Should be tested when running loops to exit when requested. */
  bool must_stop();

  std::vector<shared_ptr<boost::thread>> threads_;
  int worker_count_;
  std::mutex mutex;
};

}  // namespace caffe

#endif  // CAFFE_INTERNAL_THREAD_HPP_
