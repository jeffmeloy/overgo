package audiodsp

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/recipecontract"
	"overgo/internal/scratch"
)

// FrontendConfig declares numerical behavior, without model-family defaults.
// FrameSpan controls admission; WindowOffset places the window within both
// the source frame and the zero-padded Fourier input. Padding is zero or reflect.
// Window is periodic-hann, symmetric-hann, or rectangular. Output is frame-major.
type FrontendConfig struct {
	SampleRate   int                               `json:"sample_rate"`
	Geometry     recipecontract.AudioFrameGeometry `json:"geometry"`
	FFTLength    int                               `json:"fft_length"`
	FrameSpan    int                               `json:"frame_span"`
	WindowOffset int                               `json:"window_offset"`
	PadLeft      int                               `json:"pad_left"`
	PadRight     int                               `json:"pad_right"`
	Padding      string                            `json:"padding"`
	Window       string                            `json:"window"`
	Mel          MelConfig                         `json:"mel"`
	Log          LogConfig                         `json:"log"`
	Normalize    *NormalizeConfig                  `json:"normalize,omitzero"`
	ResampleTaps []float64                         `json:"resample_taps,omitempty"`
}

// MelConfig selects HTK or Slaney frequency mapping and optional area scaling.
// Frequency bounds must be explicit; FFT-bin frequencies use sampleRate*k/N.
type MelConfig struct {
	Scale         string  `json:"scale"`
	MinFrequency  float64 `json:"min_frequency"`
	MaxFrequency  float64 `json:"max_frequency"`
	AreaNormalize bool    `json:"area_normalize"`
}

// LogConfig selects magnitude or power, a natural or base-ten logarithm, and
// an additive or clamped positive guard. DynamicRange is in logarithm units.
// Scale and Bias apply after optional maximum-relative clipping.
type LogConfig struct {
	Power        bool     `json:"power"`
	Base         string   `json:"base"`
	GuardMode    string   `json:"guard_mode"`
	Guard        float64  `json:"guard"`
	DynamicRange *float64 `json:"dynamic_range,omitzero"`
	Scale        float64  `json:"scale"`
	Bias         float64  `json:"bias"`
}

// NormalizeConfig declares per-feature or all-feature standardization.
// Correction is the variance degrees-of-freedom subtraction (zero or one).
// Epsilon is added to the standard deviation, not to variance.
type NormalizeConfig struct {
	Mode       string  `json:"mode"`
	Correction int     `json:"correction"`
	Epsilon    float64 `json:"epsilon"`
}

// Frontend is immutable after construction and may be shared across callers.
// It reuses the projector's float64 direct-DFT reference, not a fast FFT.
// Audio.cpp 3497b7cc44753e2c141d8fe60ac42cec433e3281 supplies padding,
// window, filterbank and normalization semantics; see licenses/audio-notices.txt.
type Frontend struct {
	config                           FrontendConfig
	window, cosine, sine, filterbank []float64
	bins, windowSize, hop, bands     int
	memoryBytes, tableBytes          uint64
}

// Workspace owns mutable per-call storage. Its zero value is ready to use.
// Do not share it concurrently or between plans. Returned slices remain valid
// only until the next call with this workspace. Input chunks are never retained.
type Workspace struct {
	owner                              *Frontend
	window, real, imaginary, magnitude []float64
	accum, envelope                    []float64
	features, waveform, resampled      []float32
	joined                             []float32
}

// NewFrontend validates declared behavior and the byte budget before building
// tables. The budget covers numeric plan/workspace backing arrays, excluding
// caller-owned inputs, slice headers and Go allocator overhead. Each workspace
// independently consumes that budget; callers own aggregate concurrency limits.
func NewFrontend(config FrontendConfig, memoryBytes uint64) (*Frontend, error) {
	if err := config.Geometry.Validate(); err != nil {
		return nil, err
	}
	windowSize, windowOK := checked.Int(config.Geometry.WindowSamples)
	hop, hopOK := checked.Int(config.Geometry.HopSamples)
	bands, bandsOK := checked.Int(uint64(config.Geometry.FeatureBins))
	if !windowOK || !hopOK || !bandsOK || bands > math.MaxInt-2 || !checked.PositiveInts(config.SampleRate, config.FFTLength, config.FrameSpan) ||
		config.FFTLength < 2 || config.Geometry.FeatureBins == 0 ||
		config.WindowOffset < 0 || config.PadLeft < 0 || config.PadRight < 0 ||
		windowSize > config.FFTLength || config.WindowOffset > config.FFTLength-windowSize ||
		windowSize > config.FrameSpan || config.WindowOffset > config.FrameSpan-windowSize {
		return nil, errors.New("audio frontend: invalid frame geometry")
	}
	if config.Padding != "zero" && config.Padding != "reflect" ||
		config.Window != "periodic-hann" && config.Window != "symmetric-hann" && config.Window != "rectangular" {
		return nil, errors.New("audio frontend: unsupported padding or window")
	}
	if config.Mel.Scale != "htk" && config.Mel.Scale != "slaney" ||
		!checked.NonNegativeFinite64(config.Mel.MinFrequency) || !checked.PositiveFinite64(config.Mel.MaxFrequency) ||
		config.Mel.MinFrequency >= config.Mel.MaxFrequency || config.Mel.MaxFrequency > float64(config.SampleRate)/2 {
		return nil, errors.New("audio frontend: invalid mel declaration")
	}
	log := config.Log
	if log.Base != "natural" && log.Base != "ten" || log.GuardMode != "add" && log.GuardMode != "clamp" ||
		!checked.PositiveFinite64(log.Guard) || !checked.Finite64(log.Scale) || !checked.Finite64(log.Bias) ||
		log.DynamicRange != nil && !checked.NonNegativeFinite64(*log.DynamicRange) {
		return nil, errors.New("audio frontend: invalid logarithm declaration")
	}
	if norm := config.Normalize; norm != nil {
		if norm.Mode != "per-feature" && norm.Mode != "all" || norm.Correction < 0 || norm.Correction > 1 || !checked.PositiveFinite64(norm.Epsilon) {
			return nil, errors.New("audio frontend: invalid normalization declaration")
		}
		copy := *norm
		config.Normalize = &copy
	}
	if log.DynamicRange != nil {
		value := *log.DynamicRange
		config.Log.DynamicRange = &value
	}
	if len(config.ResampleTaps) != 0 && len(config.ResampleTaps)%2 != 1 {
		return nil, errors.New("audio frontend: FIR taps must have odd length")
	}
	for _, tap := range config.ResampleTaps {
		if !checked.Finite64(tap) {
			return nil, errors.New("audio frontend: non-finite FIR tap")
		}
	}
	p := &Frontend{config: config, bins: config.FFTLength/2 + 1, windowSize: windowSize, hop: hop,
		bands: bands, memoryBytes: memoryBytes}
	basis, ok := checked.MulInt(p.bins, windowSize)
	bank, bankOK := checked.MulInt(p.bins, p.bands)
	if !ok || !bankOK {
		return nil, errors.New("audio frontend: table extent overflows")
	}
	if err := p.reserve(0, []int{windowSize, basis, basis, bank, p.bands + 2, len(config.ResampleTaps)}, nil); err != nil {
		return nil, err
	}
	p.config.ResampleTaps = slices.Clone(config.ResampleTaps)
	p.window = make([]float64, windowSize)
	for index := range p.window {
		p.window[index] = 1
		if config.Window != "rectangular" && windowSize > 1 {
			denominator := windowSize
			if config.Window == "symmetric-hann" {
				denominator--
			}
			p.window[index] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(index)/float64(denominator))
		}
	}
	p.cosine, p.sine = make([]float64, basis), make([]float64, basis)
	for bin := range p.bins {
		angle := -2 * math.Pi * float64(bin) / float64(config.FFTLength)
		for index := range windowSize {
			phase := angle * float64(config.WindowOffset+index)
			p.cosine[bin*windowSize+index], p.sine[bin*windowSize+index] = math.Cos(phase), math.Sin(phase)
		}
	}
	var err error
	p.filterbank, err = p.melFilterbank()
	if err != nil {
		return nil, err
	}
	for _, count := range []int{windowSize, basis, basis, bank, len(config.ResampleTaps)} {
		term := uint64(count) * uint64(float64Bytes)
		if p.tableBytes > math.MaxUint64-term {
			return nil, errors.New("audio frontend: table byte count overflows")
		}
		p.tableBytes += term
	}
	return p, nil
}

const (
	float64Bytes = 8
	float32Bytes = 4
)

func (p *Frontend) reserve(base uint64, doubles, singles []int) error {
	for group, counts := range [][]int{doubles, singles} {
		width := uint64(float64Bytes)
		if group != 0 {
			width = float32Bytes
		}
		for _, count := range counts {
			if count < 0 || uint64(count) > uint64(math.MaxInt)/width || base > p.memoryBytes || uint64(count) > (p.memoryBytes-base)/width {
				return errors.New("audio frontend: numeric storage exceeds byte budget or addressability")
			}
			base += uint64(count) * width
		}
	}
	return nil
}

func (p *Frontend) workspace(w *Workspace, featureCount, sampleCount int, inverse bool) error {
	if w == nil || w.owner != nil && w.owner != p {
		return errors.New("audio frontend: missing or differently owned workspace")
	}
	doubles := []int{max(cap(w.window), p.windowSize), max(cap(w.real), p.bins), max(cap(w.imaginary), p.bins), max(cap(w.magnitude), p.bins), cap(w.accum), cap(w.envelope)}
	singles := []int{max(cap(w.features), featureCount), cap(w.waveform), cap(w.joined), cap(w.resampled)}
	if inverse {
		doubles[4], doubles[5], singles[1] = max(cap(w.accum), sampleCount), max(cap(w.envelope), sampleCount), max(cap(w.waveform), sampleCount)
	}
	if err := p.reserve(p.tableBytes, doubles, singles); err != nil {
		return err
	}
	w.owner = p
	w.window = scratch.Resize(w.window, p.windowSize)
	w.real = scratch.Resize(w.real, p.bins)
	w.imaginary = scratch.Resize(w.imaginary, p.bins)
	w.magnitude = scratch.Resize(w.magnitude, p.bins)
	w.features = scratch.Resize(w.features, featureCount)
	if inverse {
		w.accum = scratch.Resize(w.accum, sampleCount)
		w.envelope = scratch.Resize(w.envelope, sampleCount)
		w.waveform = scratch.Resize(w.waveform, sampleCount)
		clear(w.accum)
		clear(w.envelope)
	}
	return nil
}

// Process computes frame-major log-mel features from a complete ordered set of
// mono chunks, without concatenating them unless explicit FIR resampling is
// required. This is offline chunk assembly, not a live streaming session.
// Frame count and values do not depend on input chunk boundaries. The result
// aliases w; errors return no result, but may overwrite scratch from a prior call.
func (p *Frontend) Process(ctx context.Context, chunks [][]float32, sampleRate int, w *Workspace) ([]float32, int, error) {
	source, frames, err := p.prepare(ctx, chunks, sampleRate, w)
	if err != nil {
		return nil, 0, err
	}
	count, ok := checked.MulInt(frames, p.bands)
	if !ok {
		return nil, 0, errors.New("audio frontend: feature extent overflows")
	}
	if err := p.workspace(w, count, source.count, false); err != nil {
		return nil, 0, err
	}
	for frame := range frames {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		p.spectrum(source, frame, w)
		for bin := range p.bins {
			power := w.real[bin]*w.real[bin] + w.imaginary[bin]*w.imaginary[bin]
			if p.config.Log.Power {
				w.magnitude[bin] = power
			} else {
				w.magnitude[bin] = math.Sqrt(power)
			}
		}
		for band := range p.bands {
			var sum float64
			for bin, magnitude := range w.magnitude {
				sum += magnitude * p.filterbank[bin*p.bands+band]
			}
			if p.config.Log.GuardMode == "add" {
				sum += p.config.Log.Guard
			} else {
				sum = max(sum, p.config.Log.Guard)
			}
			value := math.Log(sum)
			if p.config.Log.Base == "ten" {
				value = math.Log10(sum)
			}
			w.features[frame*p.bands+band] = float32(value)
		}
	}
	if err := p.transform(ctx, w.features, frames); err != nil {
		return nil, 0, err
	}
	return w.features, frames, nil
}

// Reconstruct applies the same STFT followed by its standard one-sided inverse
// and squared-window overlap normalization. It rejects uncovered samples rather
// than silently replacing their zero envelope. It does not undo mel/log features.
func (p *Frontend) Reconstruct(ctx context.Context, chunks [][]float32, sampleRate int, w *Workspace) ([]float32, error) {
	source, frames, err := p.prepare(ctx, chunks, sampleRate, w)
	if err != nil {
		return nil, err
	}
	if err := p.workspace(w, 0, source.count, true); err != nil {
		return nil, err
	}
	for frame := range frames {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		p.spectrum(source, frame, w)
		for index, window := range p.window {
			sample := frame*p.hop - p.config.PadLeft + p.config.WindowOffset + index
			if sample < 0 || sample >= source.count {
				continue
			}
			var sum float64
			for bin := range p.bins {
				weight := 2.0
				if bin == 0 || p.config.FFTLength%2 == 0 && bin == p.bins-1 {
					weight = 1
				}
				basis := bin*p.windowSize + index
				sum += weight * (w.real[bin]*p.cosine[basis] + w.imaginary[bin]*p.sine[basis])
			}
			w.accum[sample] += sum / float64(p.config.FFTLength) * window
			w.envelope[sample] += window * window
		}
	}
	for sample := range w.waveform {
		if w.envelope[sample] <= 0 {
			return nil, fmt.Errorf("audio frontend: inverse sample %d has no window coverage", sample)
		}
		value := float32(w.accum[sample] / w.envelope[sample])
		if !checked.Finite32(value) {
			return nil, errors.New("audio frontend: non-finite inverse output")
		}
		w.waveform[sample] = value
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return w.waveform, nil
}
