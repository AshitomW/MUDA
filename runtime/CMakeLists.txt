add_library(
  muda_runtime
  src/allocation_table.mm
  src/buffer.mm
  src/buffer_pool.mm
  src/cuda_runtime.mm
  src/device.mm
  src/device_manager.mm
  src/driver_api.mm
  src/error_state.cpp
  src/event_manager.mm
  src/kernel.mm
  src/kernel_registry.mm
  src/launch_engine.mm
  src/memory_manager.mm
  src/module_manager.mm
  src/runtime_context.mm
  src/stream_manager.mm
  src/transfer_manager.mm
)
add_library(muda::runtime ALIAS muda_runtime)

target_include_directories(
  muda_runtime
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${MUDA_GENERATED_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(muda_runtime PUBLIC cxx_std_20)
target_compile_options(muda_runtime PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
target_link_libraries(
  muda_runtime
  PRIVATE
    muda_compiler
    muda_warnings
    "-framework Foundation"
    "-framework Metal"
)

add_library(
  cuda SHARED
  src/allocation_table.mm
  src/buffer.mm
  src/buffer_pool.mm
  src/cuda_runtime.mm
  src/device.mm
  src/device_manager.mm
  src/driver_api.mm
  src/error_state.cpp
  src/event_manager.mm
  src/kernel.mm
  src/kernel_registry.mm
  src/launch_engine.mm
  src/memory_manager.mm
  src/module_manager.mm
  src/runtime_context.mm
  src/stream_manager.mm
  src/transfer_manager.mm
)
target_include_directories(
  cuda
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${MUDA_GENERATED_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(cuda PUBLIC cxx_std_20)
target_compile_options(cuda PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
target_link_libraries(
  cuda
  PRIVATE
    muda_compiler
    muda_warnings
    "-framework Foundation"
    "-framework Metal"
)
set_target_properties(cuda PROPERTIES OUTPUT_NAME "cuda")

install(TARGETS muda_runtime cuda ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${MUDA_GENERATED_DIR}/muda/version.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/muda)
