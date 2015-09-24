#pragma once

/*
A palette document:

- Color: An RGB color.
- Pos: A position for a Color, SpatialKey or TemporalKey (between 0 - 1).
- SpatialKey: Color + Pos.
- SpatialKeys: List of SpatialKey (define palette for point in zoom).
- TemporalKey: SpatialKeys + Pos.
- TemporalKeys: List of TemporalKey (define palette for entire zoom).
*/

// boost::serialization
// #define BOOST_LIB_DIAGNOSTIC
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/list.hpp>

// boost::filesystem
#define BOOST_FILESYSTEM_VERSION 3
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem/exception.hpp>

// STL
#include <list>
#include <vector>

// app
#include "int_types.h"

// ---------------------------------------------------------------------------
// Fractal parameters.
// ---------------------------------------------------------------------------

class FractalSpec {
public:
  FractalSpec();
  ~FractalSpec();

  double center_r_;
  double center_i_;
  u32 bailout_;
  double zoom_begin_;
  double zoom_end_;

  void Init();

private:
  // serialization
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & boost::serialization::make_nvp("center_r", center_r_);
	  ar & boost::serialization::make_nvp("center_i", center_i_);
	  ar & boost::serialization::make_nvp("bailout", bailout_);
    ar & boost::serialization::make_nvp("zoom_begin", zoom_begin_);
		ar & boost::serialization::make_nvp("zoom_end", zoom_end_);
  }
};


// ---------------------------------------------------------------------------
// Temporal Palette.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Color.
// ---------------------------------------------------------------------------

class Color {
public:
	Color();
	Color(const u8, const u8, const u8);
	~Color();
	u8 red_, green_, blue_;
private:
	// serialization
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & boost::serialization::make_nvp("red", red_);
		ar & boost::serialization::make_nvp("green", green_);
		ar & boost::serialization::make_nvp("blue", blue_);
	}
};

typedef std::vector<Color> ColorArray;

// ---------------------------------------------------------------------------
// SpatialKey.
// ---------------------------------------------------------------------------

typedef double Pos;

class SpatialKey {
  friend class TemporalPaletteCtrl;
  friend class PosLessThan;
public:
  SpatialKey();
  SpatialKey(const Pos pos, const Color& color);
  ~SpatialKey();
  void SetPos(const Pos pos);
  Pos GetPos();
  void SetColor(const Color& color);
  Color& GetColor();
private:
  Pos pos_;
  Color color_;

  // Serialization.
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & boost::serialization::make_nvp("pos", pos_);
    ar & boost::serialization::make_nvp("color", color_);
  }
};

class PosLessThan : std::binary_function<SpatialKey, SpatialKey, bool> {
public:
  bool operator()(const SpatialKey& m1, const SpatialKey& m2) const {
    return m1.pos_ < m2.pos_;
  }
};

typedef std::list<SpatialKey> SpatialKeys;

// ---------------------------------------------------------------------------
// TemporalKey.
// ---------------------------------------------------------------------------

class TemporalKey {
  friend class TemporalPalette;
  friend class TemporalPosLessThan;
public:
  TemporalKey();
  TemporalKey(const Pos pos, const SpatialKeys& spatial_keys);
  ~TemporalKey();
  void SetPos(const Pos pos);
  Pos GetPos();
  void SetSpatialKeys(const SpatialKeys& spatial_keys);
  SpatialKeys& GetSpatialKeys();
private:
  Pos pos_;
  SpatialKeys spatial_keys_;

  // Serialization.
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & boost::serialization::make_nvp("pos", pos_);
    ar & boost::serialization::make_nvp("spatial_keys", spatial_keys_);
  }
};

class TemporalPosLessThan : std::binary_function<TemporalKey, TemporalKey, bool> {
public:
  bool operator()(const TemporalKey& m1, const TemporalKey& m2) const {
    return m1.pos_ < m2.pos_;
  }
};

typedef std::list<TemporalKey> TemporalKeys;

// ---------------------------------------------------------------------------
// TemporalPalette.
// ---------------------------------------------------------------------------

class TemporalPalette {
public:
  TemporalPalette();
  ~TemporalPalette();
  void Init();
  ColorArray GetColorArray(const SpatialKeys& spatial_keys, const u32 num_colors);
	ColorArray GetColorArray(const Pos pos, const u32 num_colors);
	Color GetColor(const Pos pos);
	void SetTopColor(const Color& color);
	Color& GetTopColor();
	SpatialKey& GetTopSpatialKey();
	void SetTopTemporalKey(const TemporalKey& temporal_key);
	TemporalKey& GetTopTemporalKey();
	TemporalKeys& GetTemporalKeys();
  void CreateTemporalKey(const Pos);
  void CreateTemporalKey(const Pos, const SpatialKeys& spatial_keys);
private:
  TemporalKeys temporal_keys_;

	// Serialization.
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
    ar & boost::serialization::make_nvp("temporal_keys", temporal_keys_);
	}
};

// ---------------------------------------------------------------------------
// Track.
// ---------------------------------------------------------------------------


class Track {
  FractalSpec fractal_spec_;
  TemporalPalette temporal_palette_;

  // Serialization.
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & boost::serialization::make_nvp("fractal_spec", fractal_spec_);
    ar & boost::serialization::make_nvp("temporal_palette", temporal_palette_);
  }
public:
  Track();
  ~Track();
  void Init();
  FractalSpec& GetFractalSpec();
  TemporalPalette& GetTemporalPalette();

  void Load(const boost::filesystem::wpath& archive_path);
  void Save(const boost::filesystem::wpath& archive_path);
};
