#pragma once

struct Configuration {
  s32 device_;
  bool vsync_;
  u32 vsync_interval_;
  bool fullscreen_;
  u32 screen_w_;
  u32 screen_h_;
  u32 transform_ss_x_;
  u32 transform_ss_y_;
  double zoom_step_;
  u32 fractal_box_ss_;
  u32 boxes_per_frame_;
  u32 bailout_;
  bool grayscale_;
  bool timers_;
  bool single_precision_;
};

bool read_config(Configuration& cfg);
