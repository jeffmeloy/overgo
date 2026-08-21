package model

const (
	realSpectrumSymmetry = 2
	realSpectrumEndpoint = 1
)

// AudioWaveformPlan: semantic-token waveform contract.
type AudioWaveformPlan struct {
	SampleRate, FFTSize, HopSize, PadSize int
	MagnitudeLimit                        float32 // Post-exp spectral clamp.
}

func (p AudioWaveformPlan) Valid() bool {
	return p.SampleRate > 0 && p.FFTSize > 0 && p.HopSize > 0 && p.PadSize >= 0 &&
		realSpectrumSymmetry*p.PadSize < p.FFTSize && p.MagnitudeLimit > 0
}

func (p AudioWaveformPlan) Bins() int {
	return p.FFTSize/realSpectrumSymmetry + realSpectrumEndpoint
}

func (p AudioWaveformPlan) FrameWidth() int { return realSpectrumSymmetry * p.Bins() }
