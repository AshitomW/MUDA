/**
 * @file metal_bridge.hpp
 * @brief Shared Objective-C++ bridging utilities for UTF-8 strings and NSError formatting.
 */

#pragma once

#import <Foundation/Foundation.h>

#include <string>
#include <string_view>

namespace muda::runtime::bridge {

[[nodiscard]] inline std::string ToStdString(NSString* str) {
  if (str == nil) {
    return {};
  }
  const char* utf8 = str.UTF8String;
  return utf8 != nullptr ? std::string(utf8) : std::string{};
}

[[nodiscard]] inline NSString* ToNSString(const char* utf8) noexcept {
  if (utf8 == nullptr || utf8[0] == '\0') {
    return nil;
  }
  return [NSString stringWithUTF8String:utf8];
}

[[nodiscard]] inline std::string FormatNSError(NSError* error,
                                               std::string_view fallback = "unknown Metal error") {
  if (error == nil) {
    return std::string(fallback);
  }
  const std::string description = ToStdString(error.localizedDescription);
  return !description.empty() ? description : std::string(fallback);
}

}  // namespace muda::runtime::bridge
