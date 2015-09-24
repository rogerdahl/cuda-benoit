#include "pch.h"

#include "track.h"
#include "tracks.h"
#include "config.h"

// The track document used in the editor is based on STL and Boost. To use the
// track in the CUDA kernels, it is converted to a hierarchy of structs. In the
// conversion, some optimizations are done to make it simple to use the
// information directly from CUDA kernels.
//
// NVIDIA has put effort into making sure that the structs are packed the same
// way in device and host memory, so that they can be copied around as needed.

extern Configuration g_cfg;

using namespace std;
using namespace boost;
using namespace boost::filesystem;

StaticTracks* ReadTracks(const path& track_dir_path) {
  StaticTracks* tracks = new StaticTracks;
  memset(tracks, 0, sizeof(StaticTracks));

  u32 track_idx(0);
  for (directory_iterator itr(track_dir_path); itr != directory_iterator(); ++itr) {
    if (tracks->count_ == MAX_TRACKS) {
      cout << format("Warning: Reached limit of %d tracks and skipped remaining tracks.") % MAX_TRACKS << endl;
      break;
    }
    if (is_regular_file(*itr) && itr->path().extension() == ".benoit") {
      try {
        TrackToStatic(*itr, tracks->tracks_[track_idx++]);
      }
      catch(std::exception&) {
        continue;
      }
      ++tracks->count_;
    }
  }

  // If a bailout value has been specified in the configuration file, use it as
  // the shared bailout value.
  if (g_cfg.bailout_ != -1) {
    tracks->shared_bailout_ = g_cfg.bailout_;
    return tracks;
  }

  // A bailout value was not specified in the configuration file. Set the shared
  // bailout value to the lowest among the tracks and issue a warning if all
  // tracks were not saved with the same bailout value.

  // Find the lowest bailout.
  u32 lowest_bailout(0xffffffff);
  u32 highest_bailout(0);
  for (u32 i(0); i < tracks->count_; ++i) {
    StaticTrack& track(tracks->tracks_[i]);
    if (track.fractal_spec_.bailout_ < lowest_bailout) {
      lowest_bailout = track.fractal_spec_.bailout_;
    }
    if (track.fractal_spec_.bailout_ > highest_bailout) {
      highest_bailout = track.fractal_spec_.bailout_;
    }
  }
  if (lowest_bailout != highest_bailout) {
    cout << format("Warning: Changed the bailout value of one or more tracks to %d.") % lowest_bailout << endl;
    cout << "Colors will not be rendered correctly on the tracks that were changed." << endl;
    cout << "Fix this by editing tracks to use the same bailout value." << endl;
  }

  tracks->shared_bailout_ = lowest_bailout;

  return tracks;
}

void TrackToStatic(const path& track_path, StaticTrack& track_struct) {
  Track track;
  try {
    track.Load(track_path);
  }
  catch (std::exception& e) {
    cout << "Track skipped because of error: " << track_path.filename() << endl;
    cout << e.what() << endl;
    throw;
  }
  //
  // TemporalPalette.
  //
  TemporalPalette temporal_palette(track.GetTemporalPalette());
  TemporalKeys sorted_temporal_keys(temporal_palette.GetTemporalKeys());
	sorted_temporal_keys.sort(TemporalPosLessThan());
	// Add "virtual" sliders at ends.
	if (sorted_temporal_keys.front().GetPos() > 0.0) {
		TemporalKey temporal_key(0.0, sorted_temporal_keys.front().GetSpatialKeys());
		sorted_temporal_keys.push_front(temporal_key);
	}
	if (sorted_temporal_keys.back().GetPos() < 1.0) {
		TemporalKey p(1.0, sorted_temporal_keys.back().GetSpatialKeys());
		sorted_temporal_keys.push_back(p);
	}
  int spatial_idx(0);
	for (TemporalKeys::iterator iter = sorted_temporal_keys.begin(); iter != sorted_temporal_keys.end(); ++iter) {
    // Get spatial palette for this temporal palette key.
	  // Sort palette.
	  SpatialKeys sorted_spatial_keys(iter->GetSpatialKeys());
	  sorted_spatial_keys.sort(PosLessThan());
	  // Add "virtual" sliders at ends.
	  if (sorted_spatial_keys.front().GetPos() != 0.0) {
		  Color& color(sorted_spatial_keys.front().GetColor());
		  SpatialKey spatial_key(0.0, color);
		  sorted_spatial_keys.push_front(spatial_key);
	  }
	  if (sorted_spatial_keys.back().GetPos() != 1.0) {
		  Color& color(sorted_spatial_keys.back().GetColor());
		  SpatialKey spatial_key(1.0, color);
		  sorted_spatial_keys.push_back(spatial_key);
	  }
    track_struct.temporal_palette_.temporal_palette_keys_[spatial_idx].pos_ = iter->GetPos();
    int color_idx(0);
  	for (SpatialKeys::iterator iter2 = sorted_spatial_keys.begin(); iter2 != sorted_spatial_keys.end(); ++iter2) {
      StaticSpatialPaletteKey& color = track_struct.temporal_palette_.temporal_palette_keys_[spatial_idx].spatial_palette_keys_[color_idx++];
      color.pos_ = iter2->GetPos();
      color.color_ = make_uchar4(iter2->GetColor().red_, iter2->GetColor().green_, iter2->GetColor().blue_, 0);
    }
    ++spatial_idx;
	}
  //
  // FractalSpec.
  //
  FractalSpec fractal_spec(track.GetFractalSpec());
  track_struct.fractal_spec_.center_r_ = fractal_spec.center_r_;
  track_struct.fractal_spec_.center_i_ = fractal_spec.center_i_;
  track_struct.fractal_spec_.bailout_ = fractal_spec.bailout_;
  track_struct.fractal_spec_.zoom_begin_ = fractal_spec.zoom_begin_;
  track_struct.fractal_spec_.zoom_end_ = fractal_spec.zoom_end_;
}
