package speechsynth

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"overgo/internal/safetensors"
	"overgo/internal/tensor"
)

// VoiceState holds one voice's per-layer attention rows. The artifact
// ships each voice as a precomputed transformer attention state -- the
// upstream voice-cloning encoder is amputated, so these rows are the
// only voice representation that exists; loading one seeds the decode
// state and generation continues at the voice's frame offset, exactly
// as the reference decode path does.
type VoiceState struct {
	Frames int
	K, V   [][]float32 // per layer, [Frames * DModel], [row][heads][headDim]
}

// LoadVoiceState reads one exported voice state and validates it
// against the model geometry.
func LoadVoiceState(path string, dims Dims) (*VoiceState, error) {
	source, err := safetensors.OpenFile(path)
	if err != nil {
		return nil, err
	}
	defer source.Close()
	voice := &VoiceState{K: make([][]float32, dims.Layers), V: make([][]float32, dims.Layers)}
	for layer := range dims.Layers {
		cacheTensor, ok := source.Tensors[fmt.Sprintf("transformer.layers.%d.self_attn/cache", layer)]
		if !ok {
			return nil, fmt.Errorf("speechsynth: voice lacks layer %d attention cache", layer)
		}
		shape := cacheTensor.Shape
		if len(shape) != 5 || shape[0] != 2 || shape[1] != 1 ||
			int(shape[3]) != dims.Heads || int(shape[4]) != dims.HeadDim {
			return nil, fmt.Errorf("speechsynth: voice layer %d cache shape %v differs from model geometry", layer, shape)
		}
		frames := int(shape[2])
		offsetTensor, ok := source.Tensors[fmt.Sprintf("transformer.layers.%d.self_attn/offset", layer)]
		if !ok || offsetTensor.DType != "I64" {
			return nil, fmt.Errorf("speechsynth: voice lacks layer %d frame offset", layer)
		}
		var raw [8]byte
		if _, err := io.ReadFull(offsetTensor.Reader(), raw[:]); err != nil {
			return nil, err
		}
		offset := int(int64(binary.LittleEndian.Uint64(raw[:])))
		if offset <= tensor.FirstOffset || offset > frames {
			return nil, fmt.Errorf("speechsynth: voice layer %d offset %d exceeds %d cached frames", layer, offset, frames)
		}
		values, err := safetensors.ReadF32(cacheTensor)
		if err != nil {
			return nil, err
		}
		width := frames * dims.DModel
		if layer == tensor.FirstOffset {
			voice.Frames = offset
		} else if voice.Frames != offset {
			return nil, fmt.Errorf("speechsynth: voice layer %d offset %d differs from %d", layer, offset, voice.Frames)
		}
		voice.K[layer] = values[: offset*dims.DModel : width]
		voice.V[layer] = values[width : width+offset*dims.DModel : 2*width]
	}
	return voice, nil
}

// VoiceDecodeState seeds a decode state with the voice's attention
// rows; the next appended row continues at the voice's frame offset
// with the rotary positions the cache was built under.
func (m *Model) VoiceDecodeState(voice *VoiceState, capacity int) (*DecodeState, error) {
	if voice == nil || voice.Frames <= tensor.FirstOffset || len(voice.K) != m.Dims.Layers || len(voice.V) != m.Dims.Layers {
		return nil, fmt.Errorf("speechsynth: incomplete voice state")
	}
	st := m.NewDecodeState(voice.Frames + capacity)
	for layer := range m.Dims.Layers {
		if len(voice.K[layer]) != voice.Frames*m.Dims.DModel || len(voice.V[layer]) != voice.Frames*m.Dims.DModel {
			return nil, fmt.Errorf("speechsynth: voice layer %d rows differ from frame count", layer)
		}
		st.k[layer] = append(st.k[layer], voice.K[layer]...)
		st.v[layer] = append(st.v[layer], voice.V[layer]...)
	}
	st.rows = voice.Frames
	return st, nil
}

// ResolveVoicePath finds a named voice under the model directory's
// exported embeddings, newest export generation first.
func ResolveVoicePath(directory, name string) (string, error) {
	trimmed := strings.TrimSpace(name)
	if trimmed == "" || trimmed != filepath.Base(trimmed) || strings.ContainsAny(trimmed, `/\.`) {
		return "", fmt.Errorf("speechsynth: invalid voice name %q", name)
	}
	entries, err := os.ReadDir(directory)
	if err != nil {
		return "", err
	}
	var generations []string
	for _, entry := range entries {
		if entry.IsDir() && strings.HasPrefix(entry.Name(), "embeddings") {
			generations = append(generations, entry.Name())
		}
	}
	sort.Sort(sort.Reverse(sort.StringSlice(generations)))
	for _, generation := range generations {
		candidate := filepath.Join(directory, generation, trimmed+".safetensors")
		if _, err := os.Stat(candidate); err == nil {
			return candidate, nil
		}
	}
	return "", fmt.Errorf("speechsynth: voice %q is not exported by the artifact", name)
}
