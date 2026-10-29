/**
 * @file process.cpp
 * @brief Runs external toolchain processes.
 */

#include "process.hpp"

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <string>
#include <vector>

extern char** environ;

namespace muda::compiler::detail {

ProcessResult RunProcess(std::span<const std::string_view> arguments,
                         ProcessOutputMode output_mode) {
  if (arguments.empty() || arguments.front().empty()) {
    return {-1};
  }

  std::vector<std::string> owned_args;
  owned_args.reserve(arguments.size());
  for (const std::string_view arg : arguments) {
    owned_args.emplace_back(arg);
  }

  std::vector<char*> argv;
  argv.reserve(owned_args.size() + 1);
  for (std::string& arg : owned_args) {
    argv.push_back(arg.data());
  }
  argv.push_back(nullptr);

  posix_spawn_file_actions_t file_actions;
  if (posix_spawn_file_actions_init(&file_actions) != 0) {
    return {-1};
  }

  int spawn_result = 0;
  pid_t child = -1;

  if (output_mode == ProcessOutputMode::kSilent) {
    spawn_result =
        posix_spawn_file_actions_addopen(&file_actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
    if (spawn_result == 0) {
      spawn_result =
          posix_spawn_file_actions_addopen(&file_actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    }
  }

  if (spawn_result == 0) {
    spawn_result = posix_spawnp(&child, argv.front(), &file_actions, nullptr, argv.data(), environ);
  }

  (void)posix_spawn_file_actions_destroy(&file_actions);

  if (spawn_result != 0 || child == -1) {
    return {-1};
  }

  int status = 0;
  while (waitpid(child, &status, 0) == -1) {
    if (errno != EINTR) {
      return {-1};
    }
  }

  if (!WIFEXITED(status)) {
    return {-1};
  }
  return {WEXITSTATUS(status)};
}

}  // namespace muda::compiler::detail
