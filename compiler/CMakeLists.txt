add_library(muda_compiler src/driver.cpp)
add_library(muda::compiler ALIAS muda_compiler)

target_include_directories(
  muda_compiler
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${MUDA_GENERATED_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(muda_compiler PUBLIC cxx_std_20)
target_compile_options(muda_compiler PRIVATE -Wall -Wextra -Wpedantic)

install(TARGETS muda_compiler ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
