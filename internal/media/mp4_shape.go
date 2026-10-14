package media

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// The ISO base media file format's box layout (ISO/IEC 14496-12, sections 4.2,
// 8.3.2, 8.4.2, 8.8.3, 8.8.7 and 8.8.8), as far as a clip's shape reads it.
// The format fixes each value, so none is reconsidered.
const (
	mp4BoxHeader      = 8  // a box's 32-bit size and its type
	mp4LargeBoxHeader = 16 // a box's size marker, type and 64-bit size
	mp4LargeSize      = 1  // the size that says a 64-bit size follows the type
	mp4TypeOffset     = 4  // where a box's type starts
	mp4Field          = 4  // a 32-bit field, and a full box's version and flags
	mp4WideField      = 8  // a 64-bit field
	mp4FlagBits       = 24 // the low bits of a full box's first field that hold its flags
	mp4FixedPoint     = 16 // the fraction bits of a track header's 16.16 width and height

	tfhdBaseDataOffset     = 0x01  // tfhd carries a 64-bit base data offset
	tfhdDescriptionIndex   = 0x02  // tfhd carries a sample description index
	tfhdDefaultDuration    = 0x08  // tfhd carries a default sample duration
	trunDataOffset         = 0x01  // trun carries a data offset
	trunFirstSampleFlags   = 0x04  // trun carries the first sample's flags
	trunSampleDuration     = 0x100 // each trun sample carries its duration
	trunSampleSize         = 0x200 // each trun sample carries its size
	trunSampleFlags        = 0x400 // each trun sample carries its flags
	trunSampleCompositions = 0x800 // each trun sample carries its composition offset
)

// MP4Shape is what a fragmented MP4 says of its one track: its frame count,
// its display size and its frame rate, read from the file rather than a JSON
// copy.
type MP4Shape struct {
	Frames int
	Height int
	Width  int
	FPS    int
}

// DecodeMP4Shape reads the shape of a fragmented MP4 of one track, the form a
// clip is streamed in (FFmpeg's frag_keyframe and empty_moov).
//   - Width and Height = the track header's display size
//   - Frames = the samples every fragment's runs hold
//   - FPS = Frames over the samples' summed durations in the media's timescale, to the nearest whole frame a second
//   - durations come from the run, else the fragment header, else the movie's track defaults
//   - an MP4 of other than one track, of no fragment samples, or whose boxes overrun the file is refused.
func DecodeMP4Shape(data []byte) (MP4Shape, error) {
	var walk mp4Walk
	if err := walk.boxes(data); err != nil {
		return MP4Shape{}, err
	}
	if walk.tracks != 1 || walk.frames == 0 || walk.duration == 0 || walk.timescale == 0 {
		return MP4Shape{}, fmt.Errorf("media: an MP4 of %d tracks and %d fragment samples lasting %d ticks of %d a second, want one track of fragments that last",
			walk.tracks, walk.frames, walk.duration, walk.timescale)
	}
	shape := MP4Shape{
		Frames: int(walk.frames), Width: int(walk.width >> mp4FixedPoint), Height: int(walk.height >> mp4FixedPoint),
		FPS: int(math.Round(float64(walk.frames) * float64(walk.timescale) / float64(walk.duration))),
	}
	if shape.Width <= 0 || shape.Height <= 0 || shape.FPS <= 0 {
		return MP4Shape{}, fmt.Errorf("media: an MP4 track of %dx%d at %d frames a second", shape.Width, shape.Height, shape.FPS)
	}
	return shape, nil
}

// mp4Walk gathers a clip's shape from its boxes in file order: the track
// defaults precede the fragments, and a fragment's header precedes its runs.
type mp4Walk struct {
	tracks                          int
	width, height, timescale        uint32
	trackDuration, fragmentDuration uint32
	frames, duration                uint64
}

var errMP4Overrun = errors.New("media: an MP4 box overruns what holds it")

// boxes walks the boxes data holds, one after another.
func (w *mp4Walk) boxes(data []byte) error {
	for len(data) > 0 {
		if len(data) < mp4BoxHeader {
			return errMP4Overrun
		}
		size, header := uint64(binary.BigEndian.Uint32(data)), uint64(mp4BoxHeader)
		switch size {
		case 0:
			size = uint64(len(data))
		case mp4LargeSize:
			if len(data) < mp4LargeBoxHeader {
				return errMP4Overrun
			}
			size, header = binary.BigEndian.Uint64(data[mp4BoxHeader:]), mp4LargeBoxHeader
		}
		if size < header || size > uint64(len(data)) {
			return errMP4Overrun
		}
		if err := w.box(string(data[mp4TypeOffset:mp4BoxHeader]), data[header:size]); err != nil {
			return err
		}
		data = data[size:]
	}
	return nil
}

// box reads one box: a container is walked, and a box that holds part of the
// shape gives it.
func (w *mp4Walk) box(kind string, body []byte) error {
	switch kind {
	case "trak":
		w.tracks++
		return w.boxes(body)
	case "moov", "mdia", "mvex", "moof", "traf":
		return w.boxes(body)
	case "tkhd":
		if len(body) < mp4WideField {
			return errMP4Overrun
		}
		w.width, w.height = binary.BigEndian.Uint32(body[len(body)-mp4WideField:]), binary.BigEndian.Uint32(body[len(body)-mp4Field:])
	case "mdhd":
		times := mp4Field
		if len(body) > 0 && body[0] == 1 {
			times = mp4WideField
		}
		return mp4Read(body, mp4Field+times+times, &w.timescale)
	case "trex":
		return mp4Read(body, mp4Field+mp4Field+mp4Field, &w.trackDuration)
	case "tfhd":
		flags, offset := mp4Flags(body), mp4Field+mp4Field
		if flags&tfhdBaseDataOffset != 0 {
			offset += mp4WideField
		}
		if flags&tfhdDescriptionIndex != 0 {
			offset += mp4Field
		}
		w.fragmentDuration = w.trackDuration
		if flags&tfhdDefaultDuration != 0 {
			return mp4Read(body, offset, &w.fragmentDuration)
		}
	case "trun":
		return w.run(body)
	}
	return nil
}

// run counts a track run's samples and their durations.
func (w *mp4Walk) run(body []byte) error {
	var count uint32
	if err := mp4Read(body, mp4Field, &count); err != nil {
		return err
	}
	flags, offset := mp4Flags(body), mp4Field+mp4Field
	for _, present := range []uint32{trunDataOffset, trunFirstSampleFlags} {
		if flags&present != 0 {
			offset += mp4Field
		}
	}
	stride := 0
	for _, present := range []uint32{trunSampleDuration, trunSampleSize, trunSampleFlags, trunSampleCompositions} {
		if flags&present != 0 {
			stride += mp4Field
		}
	}
	if uint64(offset)+uint64(count)*uint64(stride) > uint64(len(body)) {
		return errMP4Overrun
	}
	w.frames += uint64(count)
	if flags&trunSampleDuration == 0 {
		w.duration += uint64(count) * uint64(w.fragmentDuration)
		return nil
	}
	for sample := range int(count) {
		w.duration += uint64(binary.BigEndian.Uint32(body[offset+sample*stride:]))
	}
	return nil
}

// mp4Flags is a full box's flags.
func mp4Flags(body []byte) uint32 {
	if len(body) < mp4Field {
		return 0
	}
	return binary.BigEndian.Uint32(body) & (1<<mp4FlagBits - 1)
}

// mp4Read reads the 32-bit field at offset into value, or refuses a box too
// short to hold it.
func mp4Read(body []byte, offset int, value *uint32) error {
	if len(body) < offset+mp4Field {
		return errMP4Overrun
	}
	*value = binary.BigEndian.Uint32(body[offset:])
	return nil
}
