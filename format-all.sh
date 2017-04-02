#!/bin/sh

find editor lib player track \( -name '*.cpp' -or -name '*.h' -or -name '*.cu' \) -print -exec clang-format --style=file -i {} \;

# clang-format breaks <<< >>> CUDA kernel calls
find editor lib player track -name '*.cu' -print -exec sed -i 's/> > >/>>>/g' {} \;
