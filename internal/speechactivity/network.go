// Package speechactivity composes declared frame classifiers and speech-boundary
// policies over common audio and numerical operations.
package speechactivity

import (
	"context"
	"errors"
	"fmt"
	"math"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/pytorchzip"
	"overgo/internal/scratch"
)

// Affine binds a matrix and optional bias by artifact tensor name.
type Affine struct {
	Weight string `json:"weight"`
	Bias   string `json:"bias,omitzero"`
}

// MemoryBlock declares two projections and a depthwise finite-memory filter.
// Expansion uses ReLU; projection activation and the outer residual are explicit.
type MemoryBlock struct {
	Expand         Affine `json:"expand"`
	Project        Affine `json:"project"`
	ProjectReLU    bool   `json:"project_relu"`
	Residual       bool   `json:"residual"`
	Past           string `json:"past"`
	Future         string `json:"future,omitzero"`
	PastDilation   int    `json:"past_dilation"`
	FutureDilation int    `json:"future_dilation"`
}

// Declaration composes memory blocks, ReLU dense layers and a sigmoid output.
// Tensor names and topology belong to this declaration, never a family switch.
type Declaration struct {
	Blocks []MemoryBlock `json:"blocks"`
	Dense  []Affine      `json:"dense"`
	Output Affine        `json:"output"`
}

type affine struct {
	weight, bias []float32
	in, out      int
}
type memoryBlock struct {
	expand, project                       affine
	past, future                          []float32
	pastDilation, futureDilation, context int
	projectReLU, residual                 bool
}

// Network owns immutable weights. Every concurrent execution needs a Workspace.
type Network struct {
	blocks                   []memoryBlock
	dense                    []affine
	output                   affine
	inputWidth, scratchWidth int
	weightBytes, memoryBytes uint64
	causal                   bool
}

// InputWidth is the number of declared input features per frame.
func (n *Network) InputWidth() int { return n.inputWidth }

// Causal reports whether every block has no future taps.
func (n *Network) Causal() bool { return n.causal }

type weightLoader struct {
	reader        *pytorchzip.Reader
	metas         []pytorchzip.TensorMeta
	values        map[string][]float32
	bytes, budget uint64
}

func (l *weightLoader) tensor(ctx context.Context, name string, rank int) ([]float32, []int, error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	bindings, err := pytorchzip.CompileBindings(l.metas, []string{name})
	if err != nil {
		return nil, nil, err
	}
	shape, err := pytorchzip.HostShape(bindings[0].Meta, rank)
	if err != nil {
		return nil, nil, err
	}
	if values, found := l.values[name]; found {
		return values, shape, nil
	}
	count := bindings[0].Meta.Numel
	if count < 0 || uint64(count) > (l.budget-l.bytes)/binaryschema.Uint32Bytes {
		return nil, nil, errors.New("speech activity: weights exceed numeric budget")
	}
	values, err := l.reader.ReadBinding(bindings[0])
	if err != nil {
		return nil, nil, err
	}
	for _, value := range values {
		if !checked.Finite32(value) {
			return nil, nil, fmt.Errorf("speech activity: non-finite tensor %q", name)
		}
	}
	l.bytes += uint64(len(values)) * binaryschema.Uint32Bytes
	l.values[name] = values
	return values, shape, nil
}

func (l *weightLoader) affine(ctx context.Context, binding Affine, input int) (affine, error) {
	weight, shape, err := l.tensor(ctx, binding.Weight, 2)
	if err != nil {
		return affine{}, err
	}
	if input != 0 && shape[1] != input {
		return affine{}, errors.New("speech activity: affine input widths differ")
	}
	result := affine{weight: weight, in: shape[1], out: shape[0]}
	if binding.Bias != "" {
		bias, dims, err := l.tensor(ctx, binding.Bias, 1)
		if err != nil {
			return affine{}, err
		}
		if dims[0] != result.out {
			return affine{}, errors.New("speech activity: affine bias width differs")
		}
		result.bias = bias
	}
	return result, nil
}

// LoadNetwork binds a validated declaration through the bounded PyTorch reader.
// No checkpoint code is executed. All tensor names must be consumed, and numeric
// weight/workspace storage must fit memoryBytes; caller PCM and Go headers are excluded.
func LoadNetwork(ctx context.Context, checkpoint string, declaration Declaration, memoryBytes uint64) (*Network, error) {
	if ctx == nil || len(declaration.Blocks) == 0 || len(declaration.Dense) == 0 || memoryBytes == 0 {
		return nil, errors.New("speech activity: incomplete network declaration")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	catalog, err := pytorchzip.ReadCatalog(checkpoint)
	if err != nil {
		return nil, err
	}
	reader, err := pytorchzip.Open(checkpoint)
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	loader := weightLoader{reader: reader, metas: catalog.Tensors, values: map[string][]float32{}, budget: memoryBytes}
	n := &Network{memoryBytes: memoryBytes, causal: true}
	width := 0
	for _, binding := range declaration.Blocks {
		expand, err := loader.affine(ctx, binding.Expand, width)
		if err != nil {
			return nil, err
		}
		if n.inputWidth == 0 {
			n.inputWidth = expand.in
		}
		project, err := loader.affine(ctx, binding.Project, expand.out)
		if err != nil {
			return nil, err
		}
		past, shape, err := loader.tensor(ctx, binding.Past, 3)
		if err != nil {
			return nil, err
		}
		if shape[0] != project.out || shape[1] != 1 || binding.PastDilation <= 0 || binding.FutureDilation < 0 || binding.Residual && expand.in != project.out {
			return nil, errors.New("speech activity: invalid memory or residual geometry")
		}
		contextFrames, ok := checked.MulInt(shape[2]-1, binding.PastDilation)
		if !ok {
			return nil, errors.New("speech activity: memory context overflows")
		}
		block := memoryBlock{expand: expand, project: project, past: past, pastDilation: binding.PastDilation,
			futureDilation: binding.FutureDilation, context: contextFrames, projectReLU: binding.ProjectReLU, residual: binding.Residual}
		if binding.Future != "" {
			future, shape, err := loader.tensor(ctx, binding.Future, 3)
			if err != nil {
				return nil, err
			}
			_, spanOK := checked.MulInt(shape[2], binding.FutureDilation)
			if shape[0] != project.out || shape[1] != 1 || binding.FutureDilation <= 0 || !spanOK {
				return nil, errors.New("speech activity: invalid future geometry")
			}
			block.future, n.causal = future, false
		}
		n.blocks = append(n.blocks, block)
		n.scratchWidth = max(n.scratchWidth, expand.in, expand.out, project.out)
		width = project.out
	}
	for _, binding := range declaration.Dense {
		layer, err := loader.affine(ctx, binding, width)
		if err != nil {
			return nil, err
		}
		n.dense = append(n.dense, layer)
		n.scratchWidth = max(n.scratchWidth, layer.out)
		width = layer.out
	}
	n.output, err = loader.affine(ctx, declaration.Output, width)
	if err != nil {
		return nil, err
	}
	if n.output.out != 1 || len(loader.values) != len(catalog.Tensors) {
		return nil, errors.New("speech activity: output must be binary and every tensor must be declared")
	}
	n.weightBytes = loader.bytes
	return n, nil
}

// Workspace owns reusable execution buffers and returned state, not input features.
// Its zero value is usable. Returned slices expire on its next execution.
type Workspace struct {
	owner                                    *Network
	expanded, projected, left, right, output []float32
	state                                    [][]float32
}

// Trace exposes a read-only borrowed operation result during evaluation.
// Observers must not reenter evaluation with the same workspace.
type Trace struct {
	Stage                string
	Index, Frames, Width int
	Values               []float32
}

func (n *Network) prepare(frames int, w *Workspace) error {
	count, ok := checked.MulInt(frames, n.scratchWidth)
	if !ok || frames <= 0 || w == nil || w.owner != nil && w.owner != n {
		return errors.New("speech activity: invalid workspace or frame extent")
	}
	bytes := n.weightBytes
	reserve := func(count int) bool {
		if count < 0 || bytes > n.memoryBytes || uint64(count) > (n.memoryBytes-bytes)/binaryschema.Uint32Bytes {
			return false
		}
		bytes += uint64(count) * binaryschema.Uint32Bytes
		return true
	}
	for _, size := range []int{max(count, cap(w.expanded)), max(count, cap(w.projected)), max(count, cap(w.left)), max(count, cap(w.right)), max(frames, cap(w.output))} {
		if !reserve(size) {
			return errors.New("speech activity: workspace exceeds numeric budget")
		}
	}
	for _, block := range n.blocks {
		size, ok := checked.MulInt(block.context, block.project.out)
		if !ok || !reserve(size) {
			return errors.New("speech activity: cache exceeds numeric budget")
		}
	}
	w.owner = n
	w.expanded, w.projected = scratch.Resize(w.expanded, count), scratch.Resize(w.projected, count)
	w.left, w.right, w.output = scratch.Resize(w.left, count), scratch.Resize(w.right, count), scratch.Resize(w.output, frames)
	if w.state == nil {
		w.state = make([][]float32, len(n.blocks))
		for index, block := range n.blocks {
			w.state[index] = make([]float32, 0, block.context*block.project.out)
		}
	}
	return nil
}

func project(dst, input []float32, layer affine, frames int, relu bool) {
	hostmath.LinearF64(dst, input, layer.weight, layer.bias, frames, layer.in, layer.out)
	if relu {
		for index := range dst {
			dst[index] = max(0, dst[index])
		}
	}
}

// Evaluate executes complete frames and returns sigmoid probabilities plus
// bounded frame-major caches. A nonempty previous state requires a causal
// network. The caller must copy returned state before retaining it across calls.
// Initial missing context is zero; errors return no usable output or state.
func (n *Network) Evaluate(ctx context.Context, features []float32, frames int, previous [][]float32, w *Workspace, observe func(Trace) error) ([]float32, [][]float32, error) {
	if n == nil || ctx == nil || w == nil || len(n.blocks) == 0 || n.output.out != 1 {
		return nil, nil, errors.New("speech activity: incomplete execution")
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	count, ok := checked.MulInt(frames, n.inputWidth)
	if !ok || len(features) != count || len(previous) != 0 && (!n.causal || len(previous) != len(n.blocks)) {
		return nil, nil, errors.New("speech activity: invalid input or restart geometry")
	}
	for _, value := range features {
		if !checked.Finite32(value) {
			return nil, nil, errors.New("speech activity: non-finite features")
		}
	}
	for index, values := range previous {
		block := n.blocks[index]
		if len(values)%block.project.out != 0 || len(values)/block.project.out > block.context {
			return nil, nil, errors.New("speech activity: invalid restart cache extent")
		}
		for _, value := range values {
			if !checked.Finite32(value) {
				return nil, nil, errors.New("speech activity: non-finite input or state")
			}
		}
		for _, earlier := range previous[:index] {
			if checked.SlicesOverlap(values, earlier) {
				return nil, nil, errors.New("speech activity: restart caches overlap")
			}
		}
		for other, owned := range w.state {
			if other != index && checked.SlicesOverlap(values, owned[:cap(owned)]) {
				return nil, nil, errors.New("speech activity: restart cache has wrong workspace owner")
			}
		}
	}
	for _, state := range w.state {
		if checked.SlicesOverlap(features, state[:cap(state)]) {
			return nil, nil, errors.New("speech activity: input overlaps returned state")
		}
	}
	for _, buffer := range [][]float32{w.expanded, w.projected, w.left, w.right, w.output} {
		if checked.SlicesOverlap(features, buffer[:cap(buffer)]) {
			return nil, nil, errors.New("speech activity: input overlaps workspace")
		}
		for _, cache := range previous {
			if checked.SlicesOverlap(cache, buffer[:cap(buffer)]) {
				return nil, nil, errors.New("speech activity: restart cache overlaps working buffers")
			}
		}
	}
	if err := n.prepare(frames, w); err != nil {
		return nil, nil, err
	}
	trace := func(stage string, index, width int, values []float32) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		for _, value := range values {
			if !checked.Finite32(value) {
				return errors.New("speech activity: non-finite intermediate output")
			}
		}
		if observe != nil {
			return observe(Trace{Stage: stage, Index: index, Frames: frames, Width: width, Values: values})
		}
		return nil
	}
	input := features
	left, right := w.left, w.right
	for index, block := range n.blocks {
		expanded, projected := w.expanded[:frames*block.expand.out], w.projected[:frames*block.project.out]
		project(expanded, input, block.expand, frames, true)
		if err := trace("expand", index, block.expand.out, expanded); err != nil {
			return nil, nil, err
		}
		project(projected, expanded, block.project, frames, block.projectReLU)
		if err := trace("project", index, block.project.out, projected); err != nil {
			return nil, nil, err
		}
		var cache []float32
		if len(previous) != 0 {
			cache = previous[index]
		}
		output := left[:len(projected)]
		if err := hostmath.DepthwiseMemoryF64(output, projected, block.past, block.future, cache, frames, block.project.out, block.pastDilation, block.futureDilation); err != nil {
			return nil, nil, err
		}
		if block.residual {
			for i := range output {
				output[i] += input[i]
			}
		}
		if err := trace("memory", index, block.project.out, output); err != nil {
			return nil, nil, err
		}
		keepFrames := block.context
		if frames < block.context {
			keepFrames = frames + min(len(cache)/block.project.out, block.context-frames)
		}
		keep := keepFrames * block.project.out
		oldCount := max(0, keep-len(projected))
		// Capacity was reserved once from declared context. Overlapping copy also
		// supports the last borrowed state without allocating or retaining a prefix.
		w.state[index] = w.state[index][:keep]
		copy(w.state[index][:oldCount], cache[len(cache)-oldCount:])
		copy(w.state[index][oldCount:], projected[max(0, len(projected)-keep):])
		input, left, right = output, right, left
	}
	for index, layer := range n.dense {
		output := left[:frames*layer.out]
		project(output, input, layer, frames, true)
		if err := trace("dense", index, layer.out, output); err != nil {
			return nil, nil, err
		}
		input, left, right = output, right, left
	}
	project(w.output, input, n.output, frames, false)
	if err := trace("logits", 0, 1, w.output); err != nil {
		return nil, nil, err
	}
	for index, value := range w.output {
		w.output[index] = float32(1 / (1 + math.Exp(-float64(value))))
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	return w.output, w.state, nil
}
