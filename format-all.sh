#!/bin/sh

find src \( -name '*.cpp' -or -name '*.h' -or -name '*.cu' \) -print -exec clang-format --style=file -i {} \;

# clang-format breaks <<< >>> CUDA kernel calls
find src -name '*.cu' -print -exec sed -i 's/> > >/>>>/g' {} \;
