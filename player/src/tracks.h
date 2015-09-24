#pragma once

// Static structure for storing tracks.
//
// This entire structure is copied into device constant memory. On all CUDA
// capable devices up to 2.1, constant memory size is 64K.

const u32 MAX_SPATIAL_KEYS(12);
const u32 MAX_TEMPORAL_KEYS(12);
const u32 MAX_TRACKS(30);

struct StaticFractalSpec {
  double center_r_;
  double center_i_;
  u32 bailout_;
  double zoom_begin_;
  double zoom_end_;
};

struct StaticSpatialPaletteKey {
  float pos_;
  uchar4 color_;
};

struct StaticTemporalPaletteKey {
  float pos_;
  StaticSpatialPaletteKey spatial_palette_keys_[MAX_SPATIAL_KEYS + 2 /* virtual keys */];
};

struct StaticTemporalPalette {
  StaticTemporalPaletteKey temporal_palette_keys_[MAX_TEMPORAL_KEYS + 2 /* virtual keys */];
};

struct StaticTrack {
  StaticTemporalPalette temporal_palette_;
  StaticFractalSpec fractal_spec_;
};

struct StaticTracks {
  StaticTrack tracks_[MAX_TRACKS];
  u32 count_;
  u32 shared_bailout_;
};

namespace boost {
  namespace filesystem {
    class path;
  };
};

StaticTracks* ReadTracks(const boost::filesystem::path& track_dir_path);
void TrackToStatic(const boost::filesystem::path&, StaticTrack&);
