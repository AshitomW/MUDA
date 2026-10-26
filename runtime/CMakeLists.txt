add_library(muda_runtime src/device.mm)
add_library(muda::runtime ALIAS muda_runtime)

target_include_directories(
  muda_runtime
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${MUDA_GENERATED_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(muda_runtime PUBLIC cxx_std_20)
target_compile_options(muda_runtime PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(muda_runtime PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
target_link_libraries(muda_runtime PRIVATE "-framework Foundation" "-framework Metal")

install(TARGETS muda_runtime ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${MUDA_GENERATED_DIR}/muda/version.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/muda)
