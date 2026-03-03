#pragma once

namespace command_line {

bool isHelpRequested(int const argc, char const *const argv[]);

int getInt(int const argc, char const *const argv[], int const argn,
           int const def = 0);

} // namespace command_line
