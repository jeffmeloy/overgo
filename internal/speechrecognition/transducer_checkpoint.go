package speechrecognition

import (
	"errors"
	"math"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
)

// transducerCheckpoint contains only completed numerical state. Its enclosing
// service checkpoint binds source, model, recipe, tokenizer and frontend IDs.
// Binary32 buffers preserve every bit without decimal JSON expansion.
type transducerCheckpoint struct {
	Frames    int      `json:"frames"`
	Steps     int      `json:"steps"`
	LastToken int      `json:"last_token"`
	Final     bool     `json:"final"`
	Values    [][]byte `json:"values"`
}

func (t *Transducer) checkpointLayout() ([]int, uint64, error) {
	hidden, ok := checked.MulInt(t.hidden, len(t.recurrent))
	if !ok {
		return nil, 0, errors.New("transducer checkpoint: recurrent extent overflows")
	}
	counts := []int{hidden, hidden, t.decoderProjection.out}
	_, _, cacheBytes, err := t.streamGeometry(func(count int) { counts = append(counts, count) })
	if err != nil {
		return nil, 0, err
	}
	var bytes uint64
	for _, count := range counts {
		term, ok := checked.Mul64(uint64(count), binaryschema.Uint32Bytes)
		if !ok {
			return nil, 0, errors.New("transducer checkpoint: buffer byte extent overflows")
		}
		bytes, ok = checked.Add64(bytes, term)
		if !ok || bytes > uint64(math.MaxInt) || bytes > (t.memoryBytes-t.weightBytes)/2 {
			return nil, 0, errors.New("transducer checkpoint: encoded and decoded state exceed byte budget")
		}
	}
	return counts, cacheBytes, nil
}

func transducerStateBuffers(w *TransducerWorkspace) []*[]float32 {
	buffers := []*[]float32{&w.hidden, &w.cell, &w.decoder}
	for i := range w.stream.subsampling.tails {
		buffers = append(buffers, &w.stream.subsampling.tails[i])
	}
	for i := range w.stream.encoder.attention {
		history := &w.stream.encoder.attention[i]
		buffers = append(buffers, &history.key, &history.value, &w.stream.encoder.convolution[i])
	}
	return buffers
}

func (t *Transducer) checkpoint(w *TransducerWorkspace) (transducerCheckpoint, error) {
	if w == nil || w.owner != t || w.stream == nil || w.stream.failed || !w.initialized {
		return transducerCheckpoint{}, errors.New("transducer checkpoint: no completed streaming state")
	}
	counts, _, err := t.checkpointLayout()
	if err != nil {
		return transducerCheckpoint{}, err
	}
	state := transducerCheckpoint{Frames: w.stream.encoder.frames, Steps: w.steps, LastToken: w.lastToken, Final: w.stream.final}
	for i, buffer := range transducerStateBuffers(w) {
		if len(*buffer) != counts[i] {
			return transducerCheckpoint{}, errors.New("transducer checkpoint: live state shape differs")
		}
		for _, value := range *buffer {
			if !checked.Finite32(value) {
				return transducerCheckpoint{}, errors.New("transducer checkpoint: non-finite live state")
			}
		}
		state.Values = append(state.Values, binaryschema.LittleEndian.Float32s(*buffer))
	}
	return state, nil
}

func (t *Transducer) restore(state transducerCheckpoint, w *TransducerWorkspace) error {
	if w == nil || w.owner != nil || w.stream != nil || state.Final || state.Frames <= 0 ||
		state.Frames%t.encoder.blocks[0].attention.block != 0 || state.Steps < state.Frames ||
		state.LastToken < 0 || state.LastToken >= t.vocabulary {
		return errors.New("transducer checkpoint: invalid restart progress or destination")
	}
	if limit, ok := checked.MulInt(state.Frames, t.binding.MaxSymbolsPerFrame); ok && state.Steps > limit {
		return errors.New("transducer checkpoint: emissions exceed declared frame limit")
	}
	counts, cacheBytes, err := t.checkpointLayout()
	if err != nil {
		return err
	}
	if len(state.Values) != len(counts) {
		return errors.New("transducer checkpoint: buffer count differs")
	}
	// Validate all payload sizes and numbers before allocating any decoded state.
	for i, data := range state.Values {
		if len(data) != counts[i]*binaryschema.Uint32Bytes {
			return errors.New("transducer checkpoint: buffer extent differs")
		}
		for offset := 0; offset < len(data); offset += binaryschema.Uint32Bytes {
			if !checked.Finite32(binaryschema.LittleEndian.Float32(data[offset:])) {
				return errors.New("transducer checkpoint: non-finite restored state")
			}
		}
	}
	restored := TransducerWorkspace{owner: t, initialized: true, lastToken: state.LastToken, steps: state.Steps,
		stream: &transducerStream{bytes: cacheBytes,
			subsampling: subsamplingState{started: true, tails: make([][]float32, len(t.subsampling))},
			encoder:     &encoderStream{frames: state.Frames, attention: make([]attentionHistory, len(t.encoder.blocks)), convolution: make([][]float32, len(t.encoder.blocks))}}}
	for i, buffer := range transducerStateBuffers(&restored) {
		*buffer = make([]float32, counts[i])
		for index := range *buffer {
			(*buffer)[index] = binaryschema.LittleEndian.Float32(state.Values[i][index*binaryschema.Uint32Bytes:])
		}
	}
	for i := range restored.stream.encoder.attention {
		restored.stream.encoder.attention[i].frames = min(state.Frames, t.encoder.blocks[i].attention.leftContext)
	}
	*w = restored
	return nil
}
