package speechrecognition

import (
	"context"
	"errors"
	"math"
	"strconv"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/safetensors"
	"overgo/internal/scratch"
)

// RecurrentBinding declares one input/forget/candidate/output-gated LSTM layer.
type RecurrentBinding struct {
	Input     AffineBinding `json:"input"`
	Recurrent AffineBinding `json:"recurrent"`
}

// TransducerBinding supplies acoustic subsampling, categorical conditioning,
// recurrent prediction and additive ReLU joint-network roles. No model family
// or file-name convention is inferred by the executor.
type TransducerBinding struct {
	Bands              int                         `json:"bands"`
	Subsampling        []SpatialConvolutionBinding `json:"subsampling"`
	ConditionIn        AffineBinding               `json:"condition_in"`
	ConditionOut       AffineBinding               `json:"condition_out"`
	ConditionSlots     int                         `json:"condition_slots"`
	ConditionIndex     int                         `json:"condition_index"`
	Embedding          string                      `json:"embedding"`
	Recurrent          []RecurrentBinding          `json:"recurrent"`
	DecoderProjection  AffineBinding               `json:"decoder_projection"`
	Joint              AffineBinding               `json:"joint"`
	Blank              int                         `json:"blank"`
	MaxSymbolsPerFrame int                         `json:"max_symbols_per_frame"`
}

type recurrentLayer struct{ input, recurrent linear }

// Transducer owns immutable weights for a greedy recurrent acoustic decoder.
// The acoustic blocks and output projection use the same Encoder as CTC.
type Transducer struct {
	encoder                                             *Encoder
	binding                                             TransducerBinding
	subsampling                                         []spatialConvolution
	conditionIn, conditionOut, decoderProjection, joint linear
	embedding                                           []float32
	recurrent                                           []recurrentLayer
	hidden, vocabulary                                  int
	weightBytes, memoryBytes                            uint64
}

// TransducerWorkspace owns exclusive mutable inference storage. Its zero value
// is usable; returned sequences borrow it until its next call.
type TransducerWorkspace struct {
	owner                                                                                       *Transducer
	encoder                                                                                     Workspace
	condition, expanded, conditioned, hidden, cell, inputGates, recurrentGates, decoder, logits []float32
	jointInput                                                                                  []float32
	tokens, durations                                                                           []int
	initialized                                                                                 bool
	lastToken                                                                                   int
	steps                                                                                       int
	stream                                                                                      *transducerStream
}

// TransducerResult contains all greedy emissions, including the initial blank.
// Durations are encoder-frame advances (zero or one), not word durations.
type TransducerResult struct {
	Tokens, Durations []int
	Frames            int
}

// DecoderTrace exposes borrowed prediction, recurrent state and joint logits
// for one emitted token. Step is cumulative within the clip or live stream.
type DecoderTrace struct {
	Step, Frame, Token               int
	Prediction, Hidden, Cell, Logits []float32
}

// TransducerObserver selects numerical boundaries without retaining a trace.
// Callbacks must not mutate values or reenter the workspace.
type TransducerObserver struct {
	Encoder func(Trace) error
	Decoder func(DecoderTrace) error
}

// LoadTransducer validates tensor geometry and shares common acoustic operators.
// The byte limit covers numeric weights and execution state, not process RSS.
func LoadTransducer(ctx context.Context, source *safetensors.Source, encoder Declaration, binding TransducerBinding, memoryBytes uint64) (*Transducer, error) {
	if ctx == nil || source == nil || memoryBytes == 0 || binding.Bands <= 0 || len(binding.Subsampling) == 0 ||
		len(binding.Recurrent) == 0 || binding.ConditionSlots <= 0 || binding.ConditionIndex < 0 || binding.ConditionIndex >= binding.ConditionSlots ||
		binding.Blank < 0 || binding.MaxSymbolsPerFrame <= 0 || encoder.FeedbackAfter != 0 {
		return nil, errors.New("transducer: invalid declaration")
	}
	e, err := LoadEncoder(ctx, source, encoder, memoryBytes)
	if err != nil {
		return nil, err
	}
	l := loader{source: source, values: make(map[string][]float32), budget: memoryBytes - e.weightBytes}
	t := &Transducer{encoder: e, binding: binding, memoryBytes: memoryBytes}
	t.subsampling, err = l.subsampling(ctx, binding.Subsampling, binding.Bands, e.input.in)
	if err != nil {
		return nil, err
	}
	input, ok := checked.AddInt(e.input.out, binding.ConditionSlots)
	if !ok {
		return nil, errors.New("transducer: condition extent overflows")
	}
	if t.conditionIn, err = l.linear(ctx, binding.ConditionIn, input); err != nil {
		return nil, err
	}
	if t.conditionOut, err = l.linear(ctx, binding.ConditionOut, t.conditionIn.out); err != nil {
		return nil, err
	}
	if t.conditionOut.out != e.output.in {
		return nil, errors.New("transducer: conditioned encoder width differs")
	}
	shape, err := l.shape(ctx, binding.Embedding, 2)
	if err != nil {
		return nil, err
	}
	t.vocabulary, t.hidden = shape[0], shape[1]
	if binding.Blank >= t.vocabulary {
		return nil, errors.New("transducer: blank outside vocabulary")
	}
	if t.embedding, err = l.tensor(ctx, binding.Embedding, shape...); err != nil {
		return nil, err
	}
	for _, b := range binding.Recurrent {
		var layer recurrentLayer
		if layer.input, err = l.linear(ctx, b.Input, t.hidden); err != nil {
			return nil, err
		}
		if layer.recurrent, err = l.linear(ctx, b.Recurrent, t.hidden); err != nil {
			return nil, err
		}
		gates, ok := checked.MulInt(t.hidden, 4) // Four gates are the declared LSTM operation.
		if !ok || layer.input.out != gates || layer.recurrent.out != gates {
			return nil, errors.New("transducer: recurrent gate extent differs")
		}
		t.recurrent = append(t.recurrent, layer)
	}
	if t.decoderProjection, err = l.linear(ctx, binding.DecoderProjection, t.hidden); err != nil {
		return nil, err
	}
	if t.joint, err = l.linear(ctx, binding.Joint, e.output.out); err != nil {
		return nil, err
	}
	if t.decoderProjection.out != e.output.out || t.joint.out != t.vocabulary {
		return nil, errors.New("transducer: joint geometry differs")
	}
	t.weightBytes = e.weightBytes + l.bytes
	// Do not let the shared encoder spend the decoder's retained weight budget.
	e.memoryBytes -= l.bytes
	return t, nil
}

func (w *TransducerWorkspace) buffers() []*[]float32 {
	return []*[]float32{&w.condition, &w.expanded, &w.conditioned, &w.hidden, &w.cell, &w.inputGates, &w.recurrentGates, &w.decoder, &w.logits, &w.jointInput}
}

func (t *Transducer) prepare(w *TransducerWorkspace, frames int, liveBytes uint64) error {
	widths := []int{t.conditionIn.in, t.conditionIn.out, t.conditionOut.out, t.hidden, t.hidden, t.recurrent[0].input.out, t.recurrent[0].recurrent.out, t.decoderProjection.out, t.vocabulary, t.joint.in}
	counts := make([]int, len(widths))
	bytes, valid := checked.Add64(t.weightBytes, liveBytes)
	if !valid || bytes > t.memoryBytes {
		return errors.New("transducer: live input exceeds byte budget")
	}
	capacity, ok := checked.MulInt(frames, t.binding.MaxSymbolsPerFrame)
	if !ok || capacity == math.MaxInt {
		return errors.New("transducer: emission extent overflows")
	}
	capacity++ // The explicit initial blank precedes predicted emissions.
	for _, buffer := range [][]int{w.tokens, w.durations} {
		retained := max(cap(buffer), capacity)
		if uint64(retained) > (t.memoryBytes-bytes)/(strconv.IntSize/8) {
			return errors.New("transducer: emission storage exceeds byte budget")
		}
		bytes += uint64(retained) * (strconv.IntSize / 8)
	}
	for i, buffer := range w.buffers() {
		rows := 1
		if buffer == &w.condition || buffer == &w.expanded || buffer == &w.conditioned {
			rows = frames
		}
		if buffer == &w.hidden || buffer == &w.cell {
			rows = len(t.recurrent)
		}
		count, ok := checked.MulInt(rows, widths[i])
		if !ok || uint64(max(count, cap(*buffer))) > (t.memoryBytes-bytes)/binaryschema.Uint32Bytes {
			return errors.New("transducer: workspace exceeds byte budget")
		}
		counts[i] = count
		bytes += uint64(max(count, cap(*buffer))) * binaryschema.Uint32Bytes
	}
	for _, buffer := range w.encoder.buffers() {
		if uint64(cap(*buffer)) > (t.memoryBytes-bytes)/binaryschema.Uint32Bytes {
			return errors.New("transducer: retained encoder exceeds byte budget")
		}
		bytes += uint64(cap(*buffer)) * binaryschema.Uint32Bytes
	}
	for i, buffer := range w.buffers() {
		*buffer = scratch.Resize(*buffer, counts[i])
	}
	w.tokens = scratch.Resize(w.tokens, capacity)[:0]
	w.durations = scratch.Resize(w.durations, capacity)[:0]
	w.owner = t
	return nil
}

// Recognize executes a complete clip. It does not describe chunk transport as
// native streaming: each call explicitly resets the recurrent decoder.
func (t *Transducer) Recognize(ctx context.Context, features []float32, frames int, w *TransducerWorkspace, observe *TransducerObserver) (TransducerResult, error) {
	if w != nil && w.stream != nil {
		return TransducerResult{}, errors.New("transducer: streaming workspace cannot execute an offline clip")
	}
	return t.recognize(ctx, features, frames, w, observe)
}

func (t *Transducer) recognize(ctx context.Context, features []float32, frames int, w *TransducerWorkspace, observe *TransducerObserver) (TransducerResult, error) {
	if t == nil || t.encoder == nil || len(t.recurrent) == 0 || ctx == nil || w == nil || frames <= 0 || w.owner != nil && w.owner != t {
		return TransducerResult{}, errors.New("transducer: invalid invocation")
	}
	if err := ctx.Err(); err != nil {
		return TransducerResult{}, err
	}
	for _, x := range features {
		if !checked.Finite32(x) {
			return TransducerResult{}, errors.New("transducer: non-finite features")
		}
	}
	remaining := t.memoryBytes - t.weightBytes
	var acoustic *encoderStream
	var spatial *subsamplingState
	if w.stream != nil {
		if w.stream.bytes > remaining {
			return TransducerResult{}, errors.New("transducer: stream cache exceeds byte budget")
		}
		remaining -= w.stream.bytes
		acoustic, spatial = w.stream.encoder, &w.stream.subsampling
	}
	for _, buffer := range [][]int{w.tokens, w.durations} {
		if uint64(cap(buffer)) > remaining/(strconv.IntSize/8) {
			return TransducerResult{}, errors.New("transducer: retained emissions exceed byte budget")
		}
		remaining -= uint64(cap(buffer)) * (strconv.IntSize / 8)
	}
	for _, buffers := range [][]*[]float32{w.buffers(), w.encoder.buffers()} {
		for _, buffer := range buffers {
			if uint64(cap(*buffer)) > remaining/binaryschema.Uint32Bytes {
				return TransducerResult{}, errors.New("transducer: retained state exceeds byte budget")
			}
			if checked.SlicesOverlap(features, (*buffer)[:cap(*buffer)]) {
				return TransducerResult{}, errors.New("transducer: feature/workspace alias")
			}
			remaining -= uint64(cap(*buffer)) * binaryschema.Uint32Bytes
		}
	}
	x, rows, _, err := subsample(ctx, t.subsampling, features, frames, t.binding.Bands, remaining, spatial)
	if err != nil {
		return TransducerResult{}, err
	}
	liveBytes := uint64(len(x)) * binaryschema.Uint32Bytes
	if w.stream != nil {
		liveBytes += w.stream.bytes
	}
	if err := t.prepare(w, rows, liveBytes); err != nil {
		return TransducerResult{}, err
	}
	// Reserve all transducer scratch plus the still-live subsampling output
	// while the common encoder prepares its own workspace.
	reserved := liveBytes
	for _, buffer := range w.buffers() {
		reserved += uint64(cap(*buffer)) * binaryschema.Uint32Bytes
	}
	for _, buffer := range [][]int{w.tokens, w.durations} {
		reserved += uint64(cap(buffer)) * (strconv.IntSize / 8)
	}
	e := t.encoder
	w.encoder.reservedBytes = reserved
	var encoderObserver func(Trace) error
	if observe != nil {
		encoderObserver = observe.Encoder
	}
	frameOffset := 0
	if acoustic != nil {
		frameOffset = acoustic.frames
	}
	hidden, rows, err := e.encode(ctx, x, rows, &w.encoder, encoderObserver, acoustic)
	if err != nil {
		return TransducerResult{}, err
	}
	for row := range rows {
		condition := w.condition[row*t.conditionIn.in : (row+1)*t.conditionIn.in]
		copy(condition, hidden[row*e.input.out:(row+1)*e.input.out])
		clear(condition[e.input.out:])
		condition[e.input.out+t.binding.ConditionIndex] = 1
	}
	t.conditionIn.forward(w.expanded, w.condition, rows)
	for i := range w.expanded {
		w.expanded[i] = max(0, w.expanded[i])
	}
	t.conditionOut.forward(w.conditioned, w.expanded, rows)
	projected, err := e.Project(ctx, w.conditioned, rows, &w.encoder)
	if err != nil {
		return TransducerResult{}, err
	}
	if w.stream == nil || !w.initialized {
		clear(w.hidden)
		clear(w.cell)
		w.initialized = false
		w.lastToken = t.binding.Blank
		w.steps = 0
	}
	w.tokens, w.durations = w.tokens[:0], w.durations[:0]
	if !w.initialized {
		w.tokens = append(w.tokens, t.binding.Blank)
		w.durations = append(w.durations, 0)
	}
	for frame := range rows {
		for symbols := 0; symbols < t.binding.MaxSymbolsPerFrame; symbols++ {
			if err := ctx.Err(); err != nil {
				return TransducerResult{}, err
			}
			t.predict(w.lastToken, w)
			for i := range w.decoder {
				w.jointInput[i] = max(0, projected[frame*t.joint.in+i]+w.decoder[i])
			}
			t.joint.forward(w.logits, w.jointInput, 1)
			token := 0
			for i, value := range w.logits {
				if !checked.Finite32(value) {
					return TransducerResult{}, errors.New("transducer: non-finite joint output")
				}
				if value > w.logits[token] {
					token = i
				}
			}
			advance := token == t.binding.Blank || symbols+1 == t.binding.MaxSymbolsPerFrame
			if observe != nil && observe.Decoder != nil {
				if err := observe.Decoder(DecoderTrace{Step: w.steps, Frame: frameOffset + frame, Token: token, Prediction: w.decoder, Hidden: w.hidden, Cell: w.cell, Logits: w.logits}); err != nil {
					return TransducerResult{}, err
				}
			}
			w.steps++
			w.tokens = append(w.tokens, token)
			w.lastToken = token
			duration := 0
			if advance {
				duration = 1
			}
			w.durations = append(w.durations, duration)
			if advance {
				break
			}
		}
	}
	return TransducerResult{w.tokens, w.durations, rows}, ctx.Err()
}

func (t *Transducer) predict(token int, w *TransducerWorkspace) {
	if w.initialized && token == t.binding.Blank {
		return
	}
	x := t.embedding[token*t.hidden : (token+1)*t.hidden]
	for index, layer := range t.recurrent {
		h := w.hidden[index*t.hidden : (index+1)*t.hidden]
		c := w.cell[index*t.hidden : (index+1)*t.hidden]
		layer.input.forward(w.inputGates, x, 1)
		layer.recurrent.forward(w.recurrentGates, h, 1)
		for i := range w.inputGates {
			w.inputGates[i] += w.recurrentGates[i]
		}
		for i := range h {
			input := sigmoid(w.inputGates[i])
			forget := sigmoid(w.inputGates[t.hidden+i])
			candidate := float32(math.Tanh(float64(w.inputGates[2*t.hidden+i])))
			output := sigmoid(w.inputGates[3*t.hidden+i])
			c[i] = forget*c[i] + input*candidate
			h[i] = output * float32(math.Tanh(float64(c[i])))
		}
		x = h
	}
	t.decoderProjection.forward(w.decoder, x, 1)
	w.initialized = true
}

func sigmoid(value float32) float32 { return float32(1 / (1 + math.Exp(-float64(value)))) }
