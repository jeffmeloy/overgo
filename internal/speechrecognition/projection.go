package speechrecognition

import (
	"context"
	"errors"

	"overgo/internal/adaptertrain"
	"overgo/internal/artifact"
	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/trainingprogram"
)

// Project applies the optional input adapter and frozen output projection to unpadded hidden features.
// Encode prepares the workspace; returned logits borrow its storage. Keeping
// this boundary separate lets adapter training reuse encoded features without
// computing an unused base projection or retaining an encoder backward tape.
func (e *Encoder) Project(ctx context.Context, hidden []float32, frames int, w *Workspace) ([]float32, error) {
	if ctx == nil || e == nil || w == nil || w.owner != e || frames <= 0 {
		return nil, errors.New("encoder: invalid projection execution")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	features, featuresOK := checked.MulInt(frames, e.output.in)
	outputs, outputsOK := checked.MulInt(frames, e.output.out)
	if !featuresOK || !outputsOK || len(hidden) != features || outputs > len(w.logits) ||
		checked.SlicesOverlap(hidden, w.logits) {
		return nil, errors.New("encoder: projection shape, storage or overlap differs")
	}
	for _, value := range hidden {
		if !checked.Finite32(value) {
			return nil, errors.New("encoder: non-finite projection input")
		}
	}
	logits := w.logits[:outputs]
	if len(e.inputProjection) == 0 {
		e.output.forward(logits, hidden, frames)
	} else {
		if len(w.branch) < features || checked.SlicesOverlap(hidden, w.branch) {
			return nil, errors.New("encoder: adapted projection scratch or overlap differs")
		}
		hostmath.LinearInputProjection(logits, w.branch[:features], hidden, e.inputProjection,
			e.output.weight, e.output.bias, frames, e.output.in, e.output.out)
	}
	for _, value := range logits {
		if !checked.Finite32(value) {
			return nil, errors.New("encoder: non-finite projection output")
		}
	}
	return logits, ctx.Err()
}

// WithOutputProjection loads one immutable adapter without copying the base.
// Only Project uses it; tied intermediate output/feedback weights stay frozen.
// The new encoder requires its own workspace and includes adapter numeric bytes
// in admission. Existing branch scratch holds projected final features.
func (e *Encoder) WithOutputProjection(ctx context.Context, directory string, binding adaptertrain.InputProjectionBinding, weights artifact.ID) (*Encoder, error) {
	if ctx == nil || e == nil || len(e.blocks) == 0 || len(e.inputProjection) != 0 || e.weightBytes > e.memoryBytes {
		return nil, errors.New("encoder: invalid output projection binding")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	projection, err := adaptertrain.LoadInputProjection(directory, binding, weights, e.output.in, e.memoryBytes-e.weightBytes)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	adapted := *e
	adapted.inputProjection = projection
	adapted.weightBytes += uint64(len(projection)) * binaryschema.Uint32Bytes
	return &adapted, nil
}

// NewOutputAdapter binds the shared CPU input-projection/CTC component to this
// encoder's immutable output weights. maxFrames bounds encoder input frames;
// maxTargets bounds target tokens. The workspace owns the adapter reservation
// for its lifetime and cannot be rebound to another adapter or encoder.
// Memory admission includes frozen weights, retained encoder capacities, the
// input projection, all loss/gradient buffers and the shared optimizer state.
func (e *Encoder) NewOutputAdapter(ctx context.Context, w *Workspace, maxFrames, maxTargets, blank int, policy trainingprogram.OptimizerPolicy) (*adaptertrain.LinearCTC, error) {
	if ctx == nil || e == nil || len(e.blocks) == 0 || len(e.inputProjection) != 0 || w == nil || w.owner != nil && w.owner != e || w.adapter != nil || maxFrames <= 0 {
		return nil, errors.New("encoder: invalid output-adapter binding")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	outputFrames := e.outputFrames(maxFrames)
	parameters, ok := checked.MulInt(e.output.in, e.output.in)
	if !ok || outputFrames <= 0 || maxTargets < 0 || maxTargets > outputFrames || blank < 0 || blank >= e.output.out {
		return nil, errors.New("encoder: invalid output-adapter geometry")
	}
	config, err := policy.Config(parameters)
	if err != nil {
		return nil, err
	}
	used := e.weightBytes
	for _, buffer := range w.buffers() {
		bytes, ok := checked.Bytes(uint64(cap(*buffer)), binaryschema.Uint32Bytes)
		if !ok {
			return nil, errors.New("encoder: retained capacity overflows")
		}
		used, ok = checked.Add64(used, bytes)
		if !ok {
			return nil, errors.New("encoder: retained storage overflows")
		}
	}
	if used > e.memoryBytes {
		return nil, errors.New("encoder: retained storage exceeds adapter budget")
	}
	adapter, err := adaptertrain.NewLinearCTC(adaptertrain.LinearCTCSpec{
		OutputWeight: e.output.weight, OutputBias: e.output.bias, Width: e.output.in, Vocabulary: e.output.out,
		MaxFrames: outputFrames, MaxTargets: maxTargets, Blank: blank, MemoryBytes: e.memoryBytes - used, Optimizer: config,
	})
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	candidate := *w
	candidate.adapter = adapter
	if err := e.workspace(&candidate, maxFrames); err != nil {
		return nil, err
	}
	*w = candidate
	return adapter, nil
}

func (e *Encoder) outputFrames(inputFrames int) int {
	for _, block := range e.blocks {
		inputFrames /= block.convolution.stride
	}
	return inputFrames
}
