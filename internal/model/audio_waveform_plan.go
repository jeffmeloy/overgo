package model

import "overgo/internal/tensor"

// AudioWaveformPlan: semantic-token waveform contract.
type AudioWaveformPlan struct {
	SampleRate, FFTSize, HopSize, PadSize int
	MagnitudeLimit                        float32 // Post-exp spectral clamp.
}

func (p AudioWaveformPlan) Valid() bool {
	return p.SampleRate > 0 && p.FFTSize > 0 && p.HopSize > 0 && p.PadSize >= 0 &&
		tensor.PairedExtent*p.PadSize < p.FFTSize && p.MagnitudeLimit > 0
}

func (p AudioWaveformPlan) Bins() int {
	return p.FFTSize/tensor.PairedExtent + tensor.SingletonExtent
}

func (p AudioWaveformPlan) FrameWidth() int { return tensor.PairedExtent * p.Bins() }
