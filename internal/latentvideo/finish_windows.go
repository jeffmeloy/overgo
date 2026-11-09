//go:build windows

package latentvideo

import (
	"cmp"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/nvvfx"
	"overgo/internal/media"
)

// ClipFinish is how a clip's frames are finished on the card before they are encoded, by NVIDIA's RTX Video Super
// Resolution from the runtime OVERGO_NVVFX names (internal/cuda/nvvfx): denoised, then deblurred, then upscaled, each
// only where the request asks, in the order NVIDIA's own upscaler script runs them (upscale_video.py _PIPELINE_ORDER).
type ClipFinish struct {
	// Upscale multiplies the clip's sides by 2, 3 or 4; none where unset.
	Upscale int `json:"upscale,omitzero"`
	// UpscaleLevel is the upscaling's level, high where unset.
	UpscaleLevel UpscaleLevel `json:"upscale_level,omitzero"`
	// Denoise and Deblur are a strength from 1 to 4; none where unset.
	Denoise int `json:"denoise,omitzero"`
	Deblur  int `json:"deblur,omitzero"`
}

// The bounds of a finish, and the effect's levels it maps to (nvidia-vfx QualityLevel): denoising at 8 to 11 and
// deblurring at 12 to 15, by strength from 1 to 4.
const (
	smallestUpscale  = 2
	largestUpscale   = 4
	strongestFinish  = 4
	denoiseLevelBase = 7
	deblurLevelBase  = 11
)

// UpscaleLevel names a level of RTX Video Super Resolution's upscaling: bicubic, the four trained on compressed video
// (low to ultra), and the four for clean sources (clean-low to clean-ultra), which skip artifact suppression.
type UpscaleLevel string

// The effect's upscaling levels (nvidia-vfx QualityLevel): bicubic, then low to ultra for compressed video, and the same
// four for clean sources cleanLevelOffset above them (clean-low is 16).
const (
	bicubicLevel uint32 = iota
	lowLevel
	mediumLevel
	highLevel
	ultraLevel
	cleanLevelOffset = 15
)

// upscaleLevels are the levels an UpscaleLevel names, by the effect's QualityLevel.
var upscaleLevels = map[UpscaleLevel]uint32{
	"bicubic": bicubicLevel, "low": lowLevel, "medium": mediumLevel, "high": highLevel, "ultra": ultraLevel,
	"clean-low": lowLevel + cleanLevelOffset, "clean-medium": mediumLevel + cleanLevelOffset,
	"clean-high": highLevel + cleanLevelOffset, "clean-ultra": ultraLevel + cleanLevelOffset,
}

// defaultUpscaleLevel is the level an upscale runs at where the request names none: nvidia-vfx's own default, until
// rtx-vsr/measured reads which level a person should get first.
const defaultUpscaleLevel UpscaleLevel = "high"

// Choices are the levels a page offers.
func (UpscaleLevel) Choices() []string {
	names := make([]string, 0, len(upscaleLevels))
	for name := range upscaleLevels {
		names = append(names, string(name))
	}
	slices.SortFunc(names, func(a, b string) int {
		return int(upscaleLevels[UpscaleLevel(a)]) - int(upscaleLevels[UpscaleLevel(b)])
	})
	return names
}

// Default is the level a page shows where the request names none.
func (UpscaleLevel) Default() string { return string(defaultUpscaleLevel) }

// IsZero says the request asks for no finish.
func (finish ClipFinish) IsZero() bool {
	return finish.Upscale == 0 && finish.Denoise == 0 && finish.Deblur == 0 && finish.UpscaleLevel == ""
}

// Validate refuses a finish the effect cannot run.
func (finish ClipFinish) Validate() error {
	_, named := upscaleLevels[finish.UpscaleLevel]
	switch {
	case finish.Upscale != 0 && (finish.Upscale < smallestUpscale || finish.Upscale > largestUpscale):
		return fmt.Errorf("latent video: upscale is %d, and a clip is upscaled by %d to %d", finish.Upscale, smallestUpscale, largestUpscale)
	case finish.UpscaleLevel != "" && !named:
		return fmt.Errorf("latent video: upscale_level %q is none of %v", finish.UpscaleLevel, UpscaleLevel("").Choices())
	case finish.UpscaleLevel != "" && finish.Upscale == 0:
		return errors.New("latent video: upscale_level names a level of an upscale the request does not ask for")
	case finish.Denoise < 0 || finish.Denoise > strongestFinish || finish.Deblur < 0 || finish.Deblur > strongestFinish:
		return fmt.Errorf("latent video: denoise and deblur are a strength from 1 to %d", strongestFinish)
	}
	return nil
}

// levels are the effect's levels the finish runs, in order, each with the scale it writes at.
func (finish ClipFinish) levels() []finishStage {
	var stages []finishStage
	if finish.Denoise > 0 {
		stages = append(stages, finishStage{level: uint32(denoiseLevelBase + finish.Denoise), scale: 1})
	}
	if finish.Deblur > 0 {
		stages = append(stages, finishStage{level: uint32(deblurLevelBase + finish.Deblur), scale: 1})
	}
	if finish.Upscale > 0 {
		level := upscaleLevels[cmp.Or(finish.UpscaleLevel, defaultUpscaleLevel)]
		stages = append(stages, finishStage{level: level, scale: finish.Upscale})
	}
	return stages
}

// finishStage is one run of the effect: its level and the scale it writes at.
type finishStage struct {
	level uint32
	scale int
}

// runtimeIdentity is the identity of the runtime's libraries, read once a process: 792 MB that do not change while it
// runs.
var runtimeIdentity = sync.OnceValues(func() ([]artifact.ID, error) {
	runtime, err := nvvfx.Open()
	if err != nil {
		return nil, err
	}
	var identity []artifact.ID
	for _, library := range runtime.Libraries() {
		id, err := artifact.IdentifyFile(artifact.KindFile, library)
		if err != nil {
			return nil, err
		}
		identity = append(identity, id)
	}
	return identity, nil
})

// finishingEncoder finishes each frame on the card and hands it to the encoder after it at the finished size: the
// frame mapped to [0, 1], uploaded, run through each stage of the finish on a device worker of its own, and read back.
type finishingEncoder struct {
	next   clipEncoder
	finish ClipFinish
	pixels PixelRange
	worker *device.Worker
	stages []*nvvfx.Upscaler
	// frames are the device frames the stages read and write: the input, then each stage's output.
	frames         []nvvfx.Image
	upload, result []byte
	finished       []float32
}

// newFinishingEncoder finishes frames in the pixel range given before next, which takes them in [0, 1].
func newFinishingEncoder(next clipEncoder, finish ClipFinish, pixels PixelRange) (*finishingEncoder, error) {
	if err := finish.Validate(); err != nil {
		return nil, err
	}
	worker, err := device.New(0)
	if err != nil {
		return nil, err
	}
	return &finishingEncoder{next: next, finish: finish, pixels: pixels, worker: worker}, nil
}

// build builds the stages for frames of height by width, on the encoder's device worker.
func (f *finishingEncoder) build(state *device.State, height, width int) error {
	runtime, err := nvvfx.Open()
	if err != nil {
		return err
	}
	frame := nvvfx.Image{Width: width, Height: height}
	if frame.Pixels, err = state.Driver.MemAlloc(frame.Bytes()); err != nil {
		return err
	}
	f.frames = append(f.frames, frame)
	for _, stage := range f.finish.levels() {
		target := nvvfx.Image{Width: frame.Width * stage.scale, Height: frame.Height * stage.scale}
		if target.Pixels, err = state.Driver.MemAlloc(target.Bytes()); err != nil {
			return err
		}
		f.frames = append(f.frames, target)
		upscaler, err := runtime.NewUpscaler(state.Driver, state.Stream, stage.level, frame, target)
		if err != nil {
			return err
		}
		f.stages = append(f.stages, upscaler)
		frame = target
	}
	f.upload, f.result = make([]byte, f.frames[0].Bytes()), make([]byte, frame.Bytes())
	f.finished = make([]float32, len(f.result)/float32Bytes)
	return nil
}

// float32Bytes is the bytes of one planar float32 value, and signedUnitSpan the width of [-1, 1], which a signed frame
// is divided by, once shifted, to lie in [0, 1].
const (
	float32Bytes   = 4
	signedUnitSpan = 2
)

// add finishes the frame and hands it on.
func (f *finishingEncoder) add(ctx context.Context, index int, frame []float32, height, width int) error {
	err := f.worker.Do(ctx, func(state *device.State) error {
		if f.frames == nil {
			if err := f.build(state, height, width); err != nil {
				return err
			}
		}
		if len(frame) != len(f.upload)/float32Bytes || height != f.frames[0].Height || width != f.frames[0].Width {
			return fmt.Errorf("latent video: a %dx%d frame of %d values, where the clip's first was %dx%d", width, height, len(frame),
				f.frames[0].Width, f.frames[0].Height)
		}
		for index, value := range frame {
			if f.pixels == SignedUnitPixels {
				value = (value + 1) / signedUnitSpan
			}
			binary.LittleEndian.PutUint32(f.upload[index*float32Bytes:], math.Float32bits(min(max(value, 0), 1)))
		}
		if err := state.Driver.MemcpyHtoD(f.frames[0].Pixels, f.upload); err != nil {
			return err
		}
		for stage, upscaler := range f.stages {
			if err := upscaler.Frame(f.frames[stage].Pixels, f.frames[stage+1].Pixels); err != nil {
				return err
			}
		}
		if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
			return err
		}
		return state.Driver.MemcpyDtoH(f.result, f.frames[len(f.frames)-1].Pixels)
	})
	if err != nil {
		return err
	}
	for index := range f.finished {
		f.finished[index] = math.Float32frombits(binary.LittleEndian.Uint32(f.result[index*float32Bytes:]))
	}
	last := f.frames[len(f.frames)-1]
	return f.next.add(ctx, index, f.finished, last.Height, last.Width)
}

// Finish finishes the clip after it and records the finish beside its encoding.
func (f *finishingEncoder) Finish() (EncodedVideo, error) {
	video, err := f.next.Finish()
	if err != nil {
		return video, errors.Join(err, f.release())
	}
	identity, err := runtimeIdentity()
	if err != nil {
		return video, errors.Join(err, f.release())
	}
	record := &media.Finish{Scale: 1, Runtime: identity}
	for _, stage := range f.finish.levels() {
		record.Levels = append(record.Levels, stage.level)
		record.Scale *= stage.scale
	}
	video.Encoding.Finish = record
	return video, f.release()
}

// Close closes the encoder after it and frees the finish's device memory.
func (f *finishingEncoder) Close() error {
	return errors.Join(f.next.Close(), f.release())
}

// release frees the stages and frames on the device worker, and stops it.
func (f *finishingEncoder) release() error {
	if f.worker == nil {
		return nil
	}
	err := f.worker.Do(context.Background(), func(state *device.State) error {
		var failed error
		for _, upscaler := range f.stages {
			failed = errors.Join(failed, upscaler.Close())
		}
		for _, frame := range f.frames {
			failed = errors.Join(failed, state.Driver.MemFree(frame.Pixels))
		}
		return failed
	})
	f.stages, f.frames = nil, nil
	err = errors.Join(err, f.worker.Close())
	f.worker = nil
	return err
}
