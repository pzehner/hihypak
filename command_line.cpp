#include <string>

#include "command_line.hpp"

namespace command_line {

int getInt(int const argc, char const * const argv[], int const argn, int const def) {
    if (argn >= argc) {
        return def;
    }

    std::string arg = argv[argn];
    return std::stoi(arg);
}

}
