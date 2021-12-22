find_package(PkgConfig QUIET)

# Check to see if pkgconfig is installed.
pkg_check_modules(PC_IPOPT ipopt QUIET)

# Definitions
set(IPOPT_DEFINITIONS ${PC_IPOPT_CFLAGS_OTHER})

# Include directories
find_path(
  IPOPT_INCLUDE_DIRS
  NAMES IpIpoptNLP.hpp
  HINTS ${PC_IPOPT_INCLUDEDIR}
  PATHS "${CMAKE_INSTALL_PREFIX}/include")

# Libraries
find_library(
  IPOPT_LIBRARIES
  NAMES ipopt
  HINTS ${PC_IPOPT_LIBDIR})

# Version
set(IPOPT_VERSION ${PC_IPOPT_VERSION})

# Set (NAME)_FOUND if all the variables and the version are satisfied.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  IPOPT
  FAIL_MESSAGE DEFAULT_MSG
  REQUIRED_VARS IPOPT_INCLUDE_DIRS IPOPT_LIBRARIES
  VERSION_VAR IPOPT_VERSION)