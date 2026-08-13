package routedlm

import "fmt"

type PrefixKV struct {
	Key, Value []float32
}

// PrefixState owns every layer's reusable conditioning KV.
type PrefixState struct {
	Rows, KVHeads, HeadDim, ImageTime int
	Layers                            []PrefixKV
}

func NewPrefixState(rows, kvHeads, headDim, imageTime, layers int) (*PrefixState, error) {
	if rows <= 0 || kvHeads <= 0 || headDim <= 0 || imageTime <= 0 || layers <= 0 {
		return nil, fmt.Errorf("routed lm prefix state: invalid geometry")
	}
	return &PrefixState{
		Rows: rows, KVHeads: kvHeads, HeadDim: headDim, ImageTime: imageTime,
		Layers: make([]PrefixKV, layers),
	}, nil
}

func (s *PrefixState) SetLayer(layer int, key, value []float32) error {
	if s == nil || layer < 0 || layer >= len(s.Layers) {
		return fmt.Errorf("routed lm prefix state: invalid layer")
	}
	want := s.Rows * s.KVHeads * s.HeadDim
	if len(key) != want || len(value) != want {
		return fmt.Errorf("routed lm prefix state: layer %d elements=%d/%d want=%d", layer, len(key), len(value), want)
	}
	s.Layers[layer] = PrefixKV{Key: key, Value: value}
	return nil
}

func (s *PrefixState) Complete() bool {
	if s == nil || len(s.Layers) == 0 {
		return false
	}
	want := s.Rows * s.KVHeads * s.HeadDim
	for _, layer := range s.Layers {
		if len(layer.Key) != want || len(layer.Value) != want {
			return false
		}
	}
	return true
}
