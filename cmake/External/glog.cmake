# glog depends on gflags
include("cmake/External/gflags.cmake")

if (NOT __GLOG_INCLUDED)
  set(__GLOG_INCLUDED TRUE)

  # try the system-wide glog first
  find_package(Glog)
  set(GLOG_EXTERNAL FALSE)

endif()

