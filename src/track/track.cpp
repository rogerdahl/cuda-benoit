#include "pch.h"

// App.
//#include "../editor/utils.h"
#include "track.h"
#include "util.h"

using namespace std;
using namespace boost;
using namespace boost::filesystem;

// ---------------------------------------------------------------------------
// Fractal parameters.
// ---------------------------------------------------------------------------

FractalSpec::FractalSpec()
{
  Init();
}

FractalSpec::~FractalSpec()
{
}

void FractalSpec::Init()
{
  center_r_ = -0.5;
  center_i_ = 0.0;
  // The default bailout value.
  bailout_ = 2000;
  zoom_begin_ = 1.5;
  zoom_end_ = 1.5;
}

// ---------------------------------------------------------------------------
// Track.
// ---------------------------------------------------------------------------

Track::Track()
{
  Init();
}

Track::~Track()
{
}

void Track::Init()
{
  fractal_spec_.Init();
  temporal_palette_.Init();
}

FractalSpec& Track::GetFractalSpec()
{
  return fractal_spec_;
}

TemporalPalette& Track::GetTemporalPalette()
{
  return temporal_palette_;
}

void Track::Load(const wpath& archive_path)
{
  boost::filesystem::ifstream ifs(archive_path, ios::binary);
  boost::archive::xml_iarchive ia(ifs);
  ia >> boost::serialization::make_nvp("track", *this);
}

void Track::Save(const wpath& archive_path)
{
  boost::filesystem::ofstream ofs(archive_path, ios::binary);
  boost::archive::xml_oarchive oa(ofs);
  oa << boost::serialization::make_nvp("track", const_cast<const Track&>(*this));
}

// ---------------------------------------------------------------------------
// Color.
// ---------------------------------------------------------------------------

Color::Color()
{
}

Color::Color(const u8 red, const u8 green, const u8 blue)
  : red_(red), green_(green), blue_(blue)
{
}

Color::~Color()
{
}

// ---------------------------------------------------------------------------
// SpatialKey.
// ---------------------------------------------------------------------------

SpatialKey::SpatialKey()
{
}

SpatialKey::SpatialKey(const Pos pos, const Color& color)
  : pos_(pos), color_(color)
{
}

SpatialKey::~SpatialKey()
{
}

void SpatialKey::SetPos(const Pos pos)
{
  pos_ = clamp(pos);
}

Pos SpatialKey::GetPos()
{
  return pos_;
}

void SpatialKey::SetColor(const Color& color)
{
  color_ = color;
}

Color& SpatialKey::GetColor()
{
  return color_;
}

// ---------------------------------------------------------------------------
// TemporalKey.
// ---------------------------------------------------------------------------

TemporalKey::TemporalKey()
{
}

TemporalKey::TemporalKey(const Pos pos, const SpatialKeys& spatial_keys)
  : pos_(pos), spatial_keys_(spatial_keys)
{
}

TemporalKey::~TemporalKey()
{
}

void TemporalKey::SetPos(const Pos pos)
{
  pos_ = clamp(pos);
}

Pos TemporalKey::GetPos()
{
  return pos_;
}

void TemporalKey::SetSpatialKeys(const SpatialKeys& spatial_keys)
{
  spatial_keys_ = spatial_keys;
}

SpatialKeys& TemporalKey::GetSpatialKeys()
{
  return spatial_keys_;
}

// ---------------------------------------------------------------------------
// TemporalPalette.
// ---------------------------------------------------------------------------

TemporalPalette::TemporalPalette()
{
  Init();
}

TemporalPalette::~TemporalPalette()
{
}

void TemporalPalette::Init()
{
  // Clear any existing keys.
  temporal_keys_.clear();
  // Create initial sliders.
  SpatialKey spatial_key(0.0, Color(0, 0, 0));
  SpatialKeys spatial_keys;
  spatial_keys.push_front(spatial_key);
  // We set this to a fraction of 1.0 because the initial fractal comes out
  // better that way (because it has few pixels that approach the bailout
  // value).
  spatial_key.SetPos(0.1);
  spatial_key.SetColor(Color(255, 255, 255));
  spatial_keys.push_front(spatial_key);
  TemporalKey temporal_key(0.5, spatial_keys);
  temporal_keys_.push_front(temporal_key);
}

void TemporalPalette::SetTopColor(const Color& color)
{
  GetTopSpatialKey().SetColor(color);
}

Color& TemporalPalette::GetTopColor()
{
  return GetTopSpatialKey().GetColor();
}

SpatialKey& TemporalPalette::GetTopSpatialKey()
{
  return GetTopTemporalKey().spatial_keys_.front();
}

void TemporalPalette::SetTopTemporalKey(const TemporalKey& temporal_key)
{
  temporal_keys_.push_front(temporal_key);
}

TemporalKey& TemporalPalette::GetTopTemporalKey()
{
  return temporal_keys_.front();
}

TemporalKeys& TemporalPalette::GetTemporalKeys()
{
  return temporal_keys_;
}

ColorArray TemporalPalette::GetColorArray(
    const SpatialKeys& spatial_keys, const u32 num_colors)
{
  ColorArray color_array;
  color_array.reserve(num_colors);
  // Sort palette.
  SpatialKeys sorted_spatial_keys(spatial_keys);
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
  // Set iters to the two first sliders.
  SpatialKeys::iterator iter(sorted_spatial_keys.begin());
  SpatialKeys::iterator next(iter);
  ++next;
  // For each palette value we want to generate.
  for (u32 x(0); x < num_colors; ++x) {
    double pos((double)x / num_colors);
    // If pos is outside of values for these sliders, select next pair.
    while (pos < iter->GetPos() || pos > next->GetPos()) {
      iter++;
      next++;
    }
    // Find delta position.
    double delta(
        (double)(pos - iter->GetPos()) / (next->GetPos() - iter->GetPos()));
    // Find color at position and store in array.
    color_array.push_back(Lerp(iter->GetColor(), next->GetColor(), delta));
  }
  return color_array;
}

ColorArray TemporalPalette::GetColorArray(const Pos pos, const u32 num_colors)
{
  assert(pos >= 0.0);
  assert(pos <= 1.0);
  // Create sorted temporal_keys_.
  TemporalKeys sorted_temporal_keys(GetTemporalKeys());
  sorted_temporal_keys.sort(TemporalPosLessThan());
  // Add "virtual" sliders at ends.
  if (sorted_temporal_keys.front().GetPos() > 0.0) {
    TemporalKey temporal_key(0.0, sorted_temporal_keys.front().GetSpatialKeys());
    sorted_temporal_keys.push_front(temporal_key);
  }
  if (sorted_temporal_keys.back().GetPos() < 1.0) {
    TemporalKey temporal_key(1.0, sorted_temporal_keys.back().GetSpatialKeys());
    sorted_temporal_keys.push_back(temporal_key);
  }
  // Find the two relevant temporal_keys sliders.
  TemporalKeys::iterator iter = sorted_temporal_keys.begin();
  TemporalKeys::iterator next(iter);
  ++next;
  while (pos < iter->GetPos() || pos > next->GetPos()) {
    iter++;
    next++;
  }
  // Grab palettes for these two palette_sliders.
  ColorArray arr1(GetColorArray(iter->GetSpatialKeys(), num_colors));
  ColorArray arr2(GetColorArray(next->GetSpatialKeys(), num_colors));
  // Find delta position.
  double delta(
      (double)(pos - iter->GetPos()) / (next->GetPos() - iter->GetPos()));
  ColorArray color_array;
  color_array.reserve(num_colors);
  // For each palette value we want to generate.
  for (u32 i(0); i < num_colors; ++i) {
    // Find color at position and store in array.
    color_array.push_back(Lerp(arr1[i], arr2[i], delta));
  }
  return color_array;
}

Color TemporalPalette::GetColor(const Pos pos)
{
  SpatialKeys& spatial_keys(GetTopTemporalKey().GetSpatialKeys());
  SpatialKeys sorted_spatial_keys(spatial_keys);
  sorted_spatial_keys.sort(PosLessThan());
  // Add "virtual" sliders at ends.
  if (sorted_spatial_keys.front().GetPos() > 0.0) {
    Color& color(sorted_spatial_keys.front().GetColor());
    SpatialKey spatial_key(0.0, color);
    sorted_spatial_keys.push_front(spatial_key);
  }
  if (sorted_spatial_keys.back().GetPos() < 1.0) {
    Color& color(sorted_spatial_keys.back().GetColor());
    SpatialKey spatial_key(1.0, color);
    sorted_spatial_keys.push_back(spatial_key);
  }
  SpatialKeys::iterator iter, next;
  for (iter = sorted_spatial_keys.begin();; ++iter) {
    next = iter;
    ++next;
    if (pos >= iter->GetPos() && pos <= next->GetPos())
      break;
  }
  // Calculate color at position.
  double delta(
      (double)(pos - iter->GetPos()) / (next->GetPos() - iter->GetPos()));
  return Lerp(iter->GetColor(), next->GetColor(), delta);
}

void TemporalPalette::CreateTemporalKey(const Pos pos)
{
  SpatialKey spatial_key(0.0, Color(0, 0, 0));
  SpatialKeys spatial_keys;
  spatial_keys.push_front(spatial_key);
  spatial_key.SetPos(1.0);
  spatial_key.SetColor(Color(255, 255, 255));
  spatial_keys.push_front(spatial_key);
  TemporalKey temporal_key(pos, spatial_keys);
  temporal_keys_.push_front(temporal_key);
}

void TemporalPalette::CreateTemporalKey(
    const Pos pos, const SpatialKeys& spatial_keys)
{
  temporal_keys_.push_front(TemporalKey(pos, spatial_keys));
}
