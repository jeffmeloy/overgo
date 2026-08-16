package model

const (
	audioTokenSampleRate     = 24_000
	audioTokenFFTSize        = 1_280
	audioTokenHopSize        = 320
	audioTokenPadSize        = 480
	audioTokenMagnitudeLimit = 100
)

// AudioWaveformPlan: derived semantic-token waveform contract.
// MagnitudeLimit clamps per-bin spectral magnitudes after exponentiation.
type AudioWaveformPlan struct {
	SampleRate, FFTSize, HopSize, PadSize int
	MagnitudeLimit                        float32
}

func compileAudioWaveformPlan(profile ArchitectureProfile) AudioWaveformPlan {
	if profile.Forward.Operation != ForwardOperationAudioTokens {
		return AudioWaveformPlan{}
	}
	return AudioWaveformPlan{
		SampleRate: audioTokenSampleRate,
		FFTSize:    audioTokenFFTSize, HopSize: audioTokenHopSize, PadSize: audioTokenPadSize,
		MagnitudeLimit: audioTokenMagnitudeLimit,
	}
}

func (p AudioWaveformPlan) Valid() bool {
	return p.SampleRate > 0 && p.FFTSize > 0 && p.HopSize > 0 && p.PadSize >= 0 &&
		2*p.PadSize < p.FFTSize && p.MagnitudeLimit > 0
}

func (p AudioWaveformPlan) Bins() int       { return p.FFTSize/2 + 1 }
func (p AudioWaveformPlan) FrameWidth() int { return 2 * p.Bins() }
