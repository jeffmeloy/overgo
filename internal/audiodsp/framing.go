package audiodsp

import (
	"context"
	"errors"
	"math"

	"overgo/internal/checked"
	"overgo/internal/media"
	"overgo/internal/scratch"
)

// chunkSource borrows complete offline input. A monotone cursor avoids a
// per-sample search on contiguous chunks, including reflected frame boundaries.
type chunkSource struct {
	chunks               [][]float32
	count, chunk, offset int
}

func (s *chunkSource) at(index int) float64 {
	for index < s.offset {
		s.chunk--
		s.offset -= len(s.chunks[s.chunk])
	}
	for index >= s.offset+len(s.chunks[s.chunk]) {
		s.offset += len(s.chunks[s.chunk])
		s.chunk++
	}
	return float64(s.chunks[s.chunk][index-s.offset])
}

func (p *Frontend) prepare(ctx context.Context, chunks [][]float32, sampleRate int, w *Workspace) (chunkSource, int, error) {
	if p == nil || ctx == nil || w == nil || w.owner != nil && w.owner != p {
		return chunkSource{}, 0, errors.New("audio frontend: incomplete execution")
	}
	if err := ctx.Err(); err != nil {
		return chunkSource{}, 0, err
	}
	source := chunkSource{chunks: chunks}
	for _, chunk := range chunks {
		for _, scratch := range [][]float32{w.features, w.waveform, w.joined, w.resampled, w.sequence} {
			if checked.SlicesOverlap(chunk, scratch[:cap(scratch)]) {
				return chunkSource{}, 0, errors.New("audio frontend: input aliases mutable workspace")
			}
		}
		count, ok := checked.AddInt(source.count, len(chunk))
		if !ok {
			return chunkSource{}, 0, errors.New("audio frontend: input extent overflows")
		}
		source.count = count
		for _, value := range chunk {
			if !checked.Finite32(value) {
				return chunkSource{}, 0, errors.New("audio frontend: non-finite input")
			}
		}
		if err := ctx.Err(); err != nil {
			return chunkSource{}, 0, err
		}
	}
	if source.count == 0 || sampleRate <= 0 {
		return chunkSource{}, 0, errors.New("audio frontend: empty audio or invalid sample rate")
	}
	if sampleRate != p.config.SampleRate {
		if len(p.config.ResampleTaps) == 0 {
			return chunkSource{}, 0, errors.New("audio frontend: sample rate differs and no FIR taps were declared")
		}
		divisor := sampleRate
		remainder := p.config.SampleRate
		for remainder != 0 {
			divisor, remainder = remainder, divisor%remainder
		}
		up, down := p.config.SampleRate/divisor, sampleRate/divisor
		product, ok := checked.MulInt(source.count, up)
		if !ok || product > math.MaxInt-(down-1) {
			return chunkSource{}, 0, errors.New("audio frontend: resampling extent overflows")
		}
		outCount := (product + down - 1) / down
		// The resampler also computes an upsampled center plus its half-filter.
		lastCenter, ok := checked.MulInt(outCount-1, down)
		if !ok || lastCenter > math.MaxInt-(len(p.config.ResampleTaps)-1)/2 {
			return chunkSource{}, 0, errors.New("audio frontend: FIR index overflows")
		}
		if err := p.reserve(p.tableBytes, []int{cap(w.window), cap(w.real), cap(w.imaginary), cap(w.magnitude), cap(w.accum), cap(w.envelope), cap(w.mel)},
			[]int{cap(w.features), cap(w.waveform), max(cap(w.joined), source.count), max(cap(w.resampled), outCount), cap(w.sequence), cap(w.padding)}); err != nil {
			return chunkSource{}, 0, err
		}
		w.joined = scratch.Resize(w.joined, source.count)
		var position int
		for _, chunk := range chunks {
			position += copy(w.joined[position:], chunk)
		}
		var err error
		w.resampled, err = media.ResamplePoly(ctx, w.resampled, w.joined, up, down, p.config.ResampleTaps)
		if err != nil {
			return chunkSource{}, 0, err
		}
		if err := ctx.Err(); err != nil {
			return chunkSource{}, 0, err
		}
		for _, value := range w.resampled {
			if !checked.Finite32(value) {
				return chunkSource{}, 0, errors.New("audio frontend: non-finite resampled input")
			}
		}
		w.resampledChunks[0] = w.resampled
		source = chunkSource{chunks: w.resampledChunks[:], count: len(w.resampled)}
	}
	if p.config.Padding == "reflect" && (p.config.PadLeft >= source.count || p.config.PadRight >= source.count) {
		return chunkSource{}, 0, errors.New("audio frontend: reflection padding must be shorter than the signal")
	}
	padded, ok := checked.AddInt(source.count, p.config.PadLeft, p.config.PadRight)
	if !ok || padded < p.config.FrameSpan {
		return chunkSource{}, 0, errors.New("audio frontend: no complete frame or padded extent overflows")
	}
	geometry := p.config.Geometry
	geometry.WindowSamples = uint64(p.config.FrameSpan)
	count, err := geometry.FrameCount(uint64(padded))
	if err != nil {
		return chunkSource{}, 0, err
	}
	frames, ok := checked.Int(count)
	if !ok {
		return chunkSource{}, 0, errors.New("audio frontend: frame count overflows")
	}
	return source, frames, nil
}

func (p *Frontend) spectrum(source *chunkSource, frame int, w *Workspace) {
	start := frame*p.hop - p.config.PadLeft + p.config.WindowOffset
	for index := range p.window {
		position := start + index
		var value float64
		if position >= 0 && position < source.count {
			value = source.at(position)
		} else if p.config.Padding == "reflect" {
			if position < 0 {
				position = -position
			} else {
				position = source.count - 1 - (position - (source.count - 1))
			}
			value = source.at(position)
		}
		if coefficient := p.config.WaveformPreemphasis; coefficient != nil && position > 0 && position < source.count {
			// Match waveform arithmetic before zero/reflect padding. A padding
			// zero must not acquire the previous signal sample's contribution.
			value -= *coefficient * source.at(position-1)
		}
		w.window[index] = value
	}
	if condition := p.config.Condition; condition != nil {
		var mean float64
		for index := range w.window {
			w.window[index] *= condition.Gain
			mean += w.window[index]
		}
		if condition.RemoveMean {
			mean /= float64(p.windowSize)
			for index := range w.window {
				w.window[index] -= mean
			}
		}
		for index := len(w.window) - 1; index >= 0; index-- {
			w.window[index] -= condition.Preemphasis * w.window[max(0, index-1)]
		}
	}
	for index, window := range p.window {
		w.window[index] *= window
	}
	for bin := range p.bins {
		var real, imaginary float64
		basis := bin * p.windowSize
		for index, value := range w.window {
			real += value * p.cosine[basis+index]
			imaginary += value * p.sine[basis+index]
		}
		w.real[bin], w.imaginary[bin] = real, imaginary
	}
}
