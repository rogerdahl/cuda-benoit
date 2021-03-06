#include "config.h"
#include "cuda_timers.h"
#include "tracks.h"

void Initialize(
    u32 num_resources, cudaGraphicsResource** resources, StaticTracks* tracks);

void Shutdown();

void FractalCalc(bool mouse_button_left, bool mouse_button_right);
