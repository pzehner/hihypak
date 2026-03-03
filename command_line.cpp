#include <iostream>
#include <stdexcept>
#include <string>

#include "command_line.hpp"

namespace command_line {

bool isHelpRequested(int const argc, char const *const argv[]) {
  for (std::size_t argn = 0; argn < argc; argn++) {
    std::string arg = argv[argn];
    if (arg == "-h" or arg == "--help") {
      return true;
    }
  }

  return false;
}

int getInt(int const argc, char const *const argv[], int const argn,
           int const def) {
  if (argn >= argc) {
    return def;
  }

  std::string arg = argv[argn];
  try {
    return std::stoi(arg);
  } catch (const std::invalid_argument &error) {
    std::cerr << "Invalid value '" << arg << "'" << std::endl;
    std::cerr << "Call the program with '-h' for help" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

} // namespace command_line
