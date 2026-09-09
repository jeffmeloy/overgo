package speechactivity

import (
	"context"
	"errors"

	"overgo/internal/checked"
)

// OfflineConfig declares whole-recording hysteresis, gap merging, symmetric
// padding and valley splitting. These are deliberately distinct from live
// boundaries: offline confirmation can revise preceding frame decisions.
type OfflineConfig struct {
	Smoothing    int     `json:"smoothing"`
	Threshold    float64 `json:"threshold"`
	MinSpeech    int     `json:"min_speech"`
	MaxSpeech    int     `json:"max_speech"`
	MinSilence   int     `json:"min_silence"`
	MergeSilence int     `json:"merge_silence"`
	ExtendSpeech int     `json:"extend_speech"`
}

// OfflineDecisions computes one decision per complete frame. Splits use exact
// frame indices, never rounded seconds. Extension clips to the recording even
// when its kernel exceeds the recording length. Numeric scratch is two bool
// arrays; input probabilities are borrowed and remain unchanged.
func OfflineDecisions(ctx context.Context, probabilities []float32, config OfflineConfig, memoryBytes uint64) ([]bool, error) {
	if ctx == nil || config.Smoothing <= 0 || !checked.UnitInterval64(config.Threshold) || config.MinSpeech < 0 || config.MinSilence < 0 ||
		config.MaxSpeech <= 0 || config.MergeSilence < 0 || config.ExtendSpeech < 0 || uint64(len(probabilities)) > memoryBytes/2 {
		return nil, errors.New("speech boundary: invalid offline policy or memory budget")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	for _, value := range probabilities {
		if !checked.UnitInterval64(float64(value)) {
			return nil, errors.New("speech boundary: probability outside unit interval")
		}
	}
	n := len(probabilities)
	binary, decisions := make([]bool, n), make([]bool, n)
	for frame := range probabilities {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		start := max(0, frame-config.Smoothing+1)
		var sum float64
		for _, value := range probabilities[start : frame+1] {
			sum += float64(value) / float64(frame-start+1)
		}
		binary[frame] = sum >= config.Threshold
	}
	phase, start, silence := quiet, 0, 0
	for frame, speech := range binary {
		if config.MinSpeech == 0 && config.MinSilence == 0 {
			decisions[frame] = speech
			continue
		}
		switch phase {
		case quiet:
			if speech {
				phase, start = possibleSpeech, frame
			}
		case possibleSpeech:
			if !speech {
				phase = quiet
			} else if frame-start >= config.MinSpeech {
				phase = speaking
				for i := start; i < frame; i++ {
					decisions[i] = true
				}
			}
		case speaking:
			if !speech {
				phase, silence = possibleSilence, frame
			}
		case possibleSilence:
			if speech {
				phase = speaking
			} else if frame-silence >= config.MinSilence {
				phase = quiet
			}
		}
		decisions[frame] = phase == speaking || phase == possibleSilence
	}
	copy(binary, decisions)
	for frame := 1; frame < n; frame++ {
		if !decisions[frame-1] && decisions[frame] {
			for i := max(0, frame-config.Smoothing); i < frame; i++ {
				binary[i] = true
			}
		}
	}
	copy(decisions, binary)
	gap := -1
	for frame := 1; frame < n; frame++ {
		if binary[frame-1] && !binary[frame] {
			gap = frame
		}
		if !binary[frame-1] && binary[frame] && gap >= 0 {
			if frame-gap < config.MergeSilence {
				for i := gap; i < frame; i++ {
					decisions[i] = true
				}
			}
			gap = -1
		}
	}
	// Sliding occupancy implements symmetric dilation in linear time without
	// allocating a convolution kernel or expanding the output beyond n frames.
	left, right, active := 0, 0, 0
	for frame := range binary {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := frame + min(config.ExtendSpeech, n-1-frame) + 1
		for right < end {
			if decisions[right] {
				active++
			}
			right++
		}
		begin := frame - min(frame, config.ExtendSpeech)
		for left < begin {
			if decisions[left] {
				active--
			}
			left++
		}
		binary[frame] = active > 0
	}
	copy(decisions, binary)
	for start := 0; start < n; {
		if !binary[start] {
			start++
			continue
		}
		end := start
		for end < n && binary[end] {
			end++
		}
		for end-start > config.MaxSpeech {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			windowEnd := start + config.MaxSpeech
			at := start + config.MaxSpeech/2
			for i := at + 1; i < windowEnd; i++ {
				if probabilities[i] < probabilities[at] {
					at = i
				}
			}
			decisions[at] = false
			start = at + 1
		}
		start = end
	}
	return decisions, ctx.Err()
}
