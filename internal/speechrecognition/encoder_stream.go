package speechrecognition

type attentionHistory struct {
	key, value []float32
	frames     int
}

// encoderStream is private to the transducer's validated chunk lifecycle.
// Only bounded left context is retained; no prior encoder outputs are replayed.
type encoderStream struct {
	attention   []attentionHistory
	convolution [][]float32
	frames      int
}

func (e *Encoder) newStream() *encoderStream {
	state := &encoderStream{attention: make([]attentionHistory, len(e.blocks)), convolution: make([][]float32, len(e.blocks))}
	for i, b := range e.blocks {
		state.attention[i].key = make([]float32, b.attention.leftContext*b.attention.query.out)
		state.attention[i].value = make([]float32, len(state.attention[i].key))
		state.convolution[i] = make([]float32, (b.convolution.kernelSize-1)*b.convolution.channels)
	}
	return state
}
