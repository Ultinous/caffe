if (NOT __GFLAGS_INCLUDED) # guard against multiple includes
  set(__GFLAGS_INCLUDED TRUE)

  # use the system-wide gflags if present
  find_package(GFlags)
  set(GFLAGS_EXTERNAL FALSE)

endif()
