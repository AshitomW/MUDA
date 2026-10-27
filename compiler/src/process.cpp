/**
 * @file process.cpp
 * @brief Runs external toolchain processes without a shell.
 */

#include "process.hpp"

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <string>
#include <vector>

namespace muda::compiler::detail {

int RunProcess(const std::vector<std::string>& arguments, bool silent) {
  if (arguments.empty()) {
    return -1;
  }

  std::vector<char*> argv;
  argv.reserve(arguments.size() + 1);
  for (const std::string& argument : arguments) {
    argv.push_back(const_cast<char*>(argument.c_str()));
  }
  argv.push_back(nullptr);

  const pid_t child = fork();
  if (child == -1) {
    return -1;
  }
  if (child == 0) {
    if (silent) {
      const int null_fd = open("/dev/null", O_WRONLY);
      if (null_fd != -1) {
        dup2(null_fd, STDOUT_FILENO);
        dup2(null_fd, STDERR_FILENO);
        close(null_fd);
      }
    }
    execvp(argv.front(), argv.data());
    _exit(127);
  }

  int status = 0;
  while (waitpid(child, &status, 0) == -1) {
    if (errno != EINTR) {
      return -1;
    }
  }

  if (!WIFEXITED(status)) {
    return -1;
  }
  return WEXITSTATUS(status);
}

}
