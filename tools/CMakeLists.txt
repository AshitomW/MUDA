add_executable(mudac mudac/main.cpp)
target_link_libraries(mudac PRIVATE muda::compiler)
target_compile_options(mudac PRIVATE -Wall -Wextra -Wpedantic)

add_executable(muda muda/main.cpp)
target_link_libraries(muda PRIVATE muda::runtime)
target_compile_options(muda PRIVATE -Wall -Wextra -Wpedantic)

install(TARGETS mudac muda RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
