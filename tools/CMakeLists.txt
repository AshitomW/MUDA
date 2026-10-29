add_executable(mudac mudac/main.cpp)
target_link_libraries(mudac PRIVATE muda::compiler muda_warnings)

add_executable(muda muda/main.cpp)
target_link_libraries(muda PRIVATE muda::compiler muda::runtime muda_warnings)

install(TARGETS mudac muda RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
