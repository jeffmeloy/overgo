//go:build windows

// Package mosstrain trains low-rank adapters of a MOSS-Image transformer on the
// device. The base weights stay frozen and resident in F32; the factors of the
// adapters are host masters in one optimizer pack, uploaded whole after each
// update. A step keeps only each joint block's input through the forward, and
// the reverse sweep runs one block again at a time for what its gradient needs,
// so a step's memory beyond the weights is one block's and the checkpoints.
// The prompt encoder, the text refinement, the timestep embedding and the
// modulation stay frozen, as the source's guide asks of the text path.
package mosstrain

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"unsafe"

	"overgo/internal/checked"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/devicemath"
	"overgo/internal/latentimage"
	"overgo/internal/media"
	"overgo/internal/optimizer"
	"overgo/internal/peft"
)

const (
	// normEpsilon belongs to the pinned _RMSNorm.
	normEpsilon = 1e-6
	// timestepBase belongs to the pinned float64 timestep formula.
	timestepBase = 10000
	// float32Bytes is the storage of one activation.
	float32Bytes = 4
	// factorsPerModule is an adapter module's down and up.
	factorsPerModule = 2
	// stepVectorCount is the fields of stepVectors.
	stepVectorCount = 8
	// headModulationParts is the output head's shift and scale.
	headModulationParts = 2
	// rowBlockThreads is the thread block a row kernel runs in.
	rowBlockThreads = 256
	// rotaryPair is the width one rotation turns.
	rotaryPair = 2
	// squareSlope is the derivative of the mean squared error's square.
	squareSlope = 2
	// blockHiddenVectors and blockFeedVectors bound the token-long activations of
	// one block's forward and reverse, at hidden and at feed width; they are
	// counted from what Accumulate holds at once and rounded up.
	blockHiddenVectors, blockFeedVectors = 60, 12
	// blockTokenProducts and blockModuleProducts bound the rank-wide adapter
	// products of one block, per token and per projection's width.
	blockTokenProducts, blockModuleProducts = 6, 12
	// attentionScratchHeads is the two scratch matrices of the head in progress.
	attentionScratchHeads = 2
)

// Joint block modulation order, as the pinned _Block.forward unbinds it.
const (
	attentionShift = iota
	attentionScale
	attentionGate
	feedShift
	feedScale
	feedGate
	modulationFields
)

// Tensors gives the model's named tensors in F32.
type Tensors func(name string) ([]float32, error)

type jointWeights struct {
	norm1, norm2, qNorm, kNorm driver.DevicePtr
	projected                  [projections]driver.DevicePtr
}

// factorPair is one adapted projection's factors in the device slabs.
type factorPair struct {
	down, up         driver.DevicePtr
	gradDown, gradUp driver.DevicePtr
}

// Trainer holds one transformer on the device and one adapter of it.
type Trainer struct {
	spec    latentimage.TransformerSpec
	options Options
	scale   float32

	worker  *device.Worker
	session *devicemath.ResidentOpsSession

	blocks                                                                               []jointWeights
	videoWeight, videoBias, finalNorm, finalLinear                                       driver.DevicePtr
	time0Weight, time0Bias, time2Weight, time2Bias                                       []float32
	blockModulationWeight, blockModulationBias, headModulationWeight, headModulationBias []float32

	pack                 *optimizer.TensorPack
	stepper              optimizer.Stepper
	config               optimizer.Config
	step                 int
	factors              [][projections]factorPair
	weightSlab, gradSlab driver.DevicePtr

	reserved                                Geometry
	geometry                                Geometry
	checkpoints, rope, zero, one            driver.DevicePtr
	vectorsSlab, noised, refined, cotangent driver.DevicePtr
	arena                                   deviceArena
}

// deviceArena hands out a preallocated slab by liveness mark.
type deviceArena struct {
	base             driver.DevicePtr
	cursor, capacity int
}

func (a *deviceArena) f32(count int) (driver.DevicePtr, error) {
	if count <= 0 || a.cursor+count > a.capacity {
		return 0, fmt.Errorf("mosstrain: arena of %d values exceeded by %d more over %d", a.capacity, count, a.cursor)
	}
	pointer := devicemath.ResidentPtr(a.base, a.cursor)
	a.cursor += count
	return pointer, nil
}

// New places a transformer's weights on the device and an adapter's factors
// over them: every projection of every block needs its module.
func New(ctx context.Context, spec latentimage.TransformerSpec, options Options, weights Tensors, adapter map[string]peft.Module, config optimizer.Config) (trainer *Trainer, err error) {
	if !checked.PositiveInts(spec.Layers, spec.Hidden, spec.Heads, spec.HeadDim, spec.Intermediate, options.Rank) || !checked.PositiveFinite64(options.Alpha) {
		return nil, errors.New("mosstrain: invalid transformer or adapter shape")
	}
	worker, err := device.New(0)
	if err != nil {
		return nil, err
	}
	session, err := devicemath.NewResidentOpsSession(worker)
	if err != nil {
		return nil, errors.Join(err, worker.Close())
	}
	trainer = &Trainer{spec: spec, options: options, scale: float32(options.Alpha / float64(options.Rank)), worker: worker, session: session, config: config}
	defer func() {
		if err != nil {
			err = errors.Join(err, trainer.Close())
		}
	}()
	upload := func(name string) (driver.DevicePtr, error) {
		values, err := weights(name)
		if err != nil {
			return 0, fmt.Errorf("mosstrain: %s: %w", name, err)
		}
		return session.Upload(ctx, driver.Bytes(values))
	}
	trainer.blocks = make([]jointWeights, spec.Layers)
	for index := range trainer.blocks {
		block := &trainer.blocks[index]
		prefix := fmt.Sprintf("layers.%d.", index)
		for _, item := range []struct {
			pointer *driver.DevicePtr
			name    string
		}{{&block.norm1, "norm1"}, {&block.norm2, "norm2"}, {&block.qNorm, "q_norm"}, {&block.kNorm, "k_norm"}} {
			if *item.pointer, err = upload(prefix + item.name + ".weight"); err != nil {
				return nil, err
			}
		}
		for slot, name := range projectionNames {
			if block.projected[slot], err = upload(prefix + "proj." + name + ".weight"); err != nil {
				return nil, err
			}
		}
	}
	for _, item := range []struct {
		pointer *driver.DevicePtr
		name    string
	}{{&trainer.videoWeight, "video_embedder.weight"}, {&trainer.videoBias, "video_embedder.bias"}, {&trainer.finalNorm, "final_norm_video.weight"}, {&trainer.finalLinear, "final_linear_video.weight"}} {
		if *item.pointer, err = upload(item.name); err != nil {
			return nil, err
		}
	}
	for _, item := range []struct {
		target *[]float32
		name   string
	}{
		{&trainer.time0Weight, "time_embedding.0.weight"}, {&trainer.time0Bias, "time_embedding.0.bias"},
		{&trainer.time2Weight, "time_embedding.2.weight"}, {&trainer.time2Bias, "time_embedding.2.bias"},
		{&trainer.blockModulationWeight, "shared_block_modulation.weight"}, {&trainer.blockModulationBias, "shared_block_modulation.bias"},
		{&trainer.headModulationWeight, "video_head_modulation.weight"}, {&trainer.headModulationBias, "video_head_modulation.bias"},
	} {
		if *item.target, err = weights(item.name); err != nil {
			return nil, fmt.Errorf("mosstrain: %s: %w", item.name, err)
		}
	}
	if err := trainer.placeFactors(ctx, adapter); err != nil {
		return nil, err
	}
	return trainer, nil
}

func (t *Trainer) placeFactors(ctx context.Context, adapter map[string]peft.Module) error {
	sections := make([]optimizer.TensorSection, 0, factorsPerModule*t.spec.Layers*len(projectionNames))
	rank := t.options.Rank
	for layer := range t.spec.Layers {
		for slot, projection := range projectionNames {
			out, in := projectionShape(t.spec, slot)
			name := moduleName(layer, projection)
			module, known := adapter[name]
			if !known {
				return fmt.Errorf("mosstrain: adapter lacks module %s", name)
			}
			if module.Rank != rank || module.In != in || module.Out != out {
				return fmt.Errorf("mosstrain: module %s has rank %d and %d by %d, want rank %d and %d by %d", name, module.Rank, module.Out, module.In, rank, out, in)
			}
			sections = append(sections,
				optimizer.TensorSection{Name: name + ".down", Rows: rank, Cols: in, Values: module.Down},
				optimizer.TensorSection{Name: name + ".up", Rows: out, Cols: rank, Values: module.Up})
		}
	}
	pack, err := optimizer.NewSectionPack(sections)
	if err != nil {
		return err
	}
	t.pack = pack
	if t.stepper, err = pack.NewStepper(t.config); err != nil {
		return err
	}
	bytes, ok := checked.Bytes(uint64(pack.ParameterCount()), float32Bytes)
	if !ok {
		return errors.New("mosstrain: adapter size overflows")
	}
	if t.weightSlab, err = t.session.Allocate(ctx, bytes); err != nil {
		return err
	}
	if t.gradSlab, err = t.session.Allocate(ctx, bytes); err != nil {
		return err
	}
	t.factors = make([][projections]factorPair, t.spec.Layers)
	groups := pack.Plan().Groups()
	at := 0
	for layer := range t.spec.Layers {
		for slot := range projectionNames {
			down, up := groups[at], groups[at+1]
			t.factors[layer][slot] = factorPair{
				down: devicemath.ResidentPtr(t.weightSlab, down.Start), up: devicemath.ResidentPtr(t.weightSlab, up.Start),
				gradDown: devicemath.ResidentPtr(t.gradSlab, down.Start), gradUp: devicemath.ResidentPtr(t.gradSlab, up.Start),
			}
			at += factorsPerModule
		}
	}
	return t.uploadFactors()
}

func (t *Trainer) uploadFactors() error {
	weights, _ := t.pack.Slabs()
	return t.worker.Do(context.Background(), func(state *device.State) error {
		return state.Driver.MemcpyHtoD(t.weightSlab, driver.Bytes(weights))
	})
}

// Reserve gives the trainer the device buffers of a run's largest example:
// every smaller one shares them, so a caption's length costs no allocation.
func (t *Trainer) Reserve(ctx context.Context, largest Geometry) (err error) {
	if !checked.PositiveInts(largest.Height, largest.Width, largest.Text) {
		return errors.New("mosstrain: invalid picture geometry")
	}
	if t.checkpoints != 0 {
		return errors.New("mosstrain: buffers are already reserved")
	}
	tokens, hidden := largest.tokens(), t.spec.Hidden
	allocate := func(count int) (driver.DevicePtr, error) {
		bytes, ok := checked.Bytes(uint64(count), float32Bytes)
		if !ok {
			return 0, errors.New("mosstrain: buffer size overflows")
		}
		return t.session.Allocate(ctx, bytes)
	}
	unit := make([]float32, hidden)
	for index := range unit {
		unit[index] = 1
	}
	for _, item := range []struct {
		pointer *driver.DevicePtr
		count   int
	}{
		{&t.checkpoints, (t.spec.Layers + 1) * tokens * hidden}, {&t.rope, tokens * t.spec.HeadDim},
		{&t.vectorsSlab, stepVectorCount * hidden}, {&t.noised, largest.image() * t.spec.InChannels},
		{&t.refined, largest.Text * hidden}, {&t.cotangent, largest.image() * t.spec.InChannels},
	} {
		if *item.pointer, err = allocate(item.count); err != nil {
			return err
		}
	}
	if t.zero, err = t.session.Upload(ctx, driver.Bytes(make([]float32, hidden))); err != nil {
		return err
	}
	if t.one, err = t.session.Upload(ctx, driver.Bytes(unit)); err != nil {
		return err
	}
	capacity := t.blockFloats(tokens)
	base, err := allocate(capacity)
	if err != nil {
		return err
	}
	t.arena = deviceArena{base: base, capacity: capacity}
	t.reserved = largest
	return nil
}

// write copies host values to a device buffer.
func (t *Trainer) write(ctx context.Context, pointer driver.DevicePtr, values []float32) error {
	return t.worker.Do(ctx, func(state *device.State) error {
		return state.Driver.MemcpyHtoD(pointer, driver.Bytes(values))
	})
}

// useGeometry places an example's coordinates; its picture must fit what is reserved.
func (t *Trainer) useGeometry(ctx context.Context, geometry Geometry) error {
	if t.checkpoints == 0 || geometry.tokens() > t.reserved.tokens() || geometry.image() > t.reserved.image() || geometry.Text > t.reserved.Text {
		return errors.New("mosstrain: the example exceeds the reserved geometry")
	}
	if geometry == t.geometry {
		return nil
	}
	rope, err := latentimage.CoordinateRotary(t.spec.HeadDim, geometry.Height, geometry.Width, geometry.Text, t.spec.RopeTheta)
	if err != nil {
		return err
	}
	if err := t.write(ctx, t.rope, rope); err != nil {
		return err
	}
	t.geometry = geometry
	return nil
}

// blockFloats is the most a step holds at once beyond its checkpoints: one
// block's activations and adapter products, its attention probabilities, and
// the two scratch matrices of one head.
func (t *Trainer) blockFloats(tokens int) int {
	hidden, feed, heads, rank := t.spec.Hidden, t.spec.Intermediate, t.spec.Heads, t.options.Rank
	activations := tokens * (blockHiddenVectors*hidden + blockFeedVectors*feed)
	products := blockTokenProducts*tokens*rank + blockModuleProducts*(hidden+gatedFeed*feed)*rank
	attention := (heads + attentionScratchHeads) * tokens * tokens
	return activations + products + attention
}

// modulationVectors computes a timestep's block and head modulation on the
// host: the frozen embedding, in F32 as the source runs it.
func (t *Trainer) modulationVectors(timestep float64) (blocks, head []float32, err error) {
	embedding, err := (media.SinusoidalProgram{Dimensions: t.spec.TimestepEmbed, FrequencyBase: timestepBase, InputScale: 1}).Encode32(timestep)
	if err != nil {
		return nil, nil, err
	}
	hidden := t.spec.Hidden
	first := affine(t.time0Weight, t.time0Bias, embedding, hidden)
	for index, value := range first {
		first[index] = value / (1 + float32(math.Exp(float64(-value))))
	}
	time := affine(t.time2Weight, t.time2Bias, first, hidden)
	return affine(t.blockModulationWeight, t.blockModulationBias, time, modulationFields*hidden),
		affine(t.headModulationWeight, t.headModulationBias, time, headModulationParts*hidden), nil
}

// affine is out = weight[out, in] x + bias.
func affine(weight, bias, x []float32, out int) []float32 {
	result := make([]float32, out)
	width := len(x)
	for row := range out {
		var sum float32
		line := weight[row*width : (row+1)*width]
		for column, value := range x {
			sum += line[column] * value
		}
		result[row] = sum + bias[row]
	}
	return result
}

// stepVectors are one timestep's modulation vectors on the device: each scale
// is 1+scale, and the gates are as they stand.
type stepVectors struct {
	attentionShift, attentionScale, attentionGate driver.DevicePtr
	feedShift, feedScale, feedGate                driver.DevicePtr
	headShift, headScale                          driver.DevicePtr
}

func (t *Trainer) uploadVectors(ctx context.Context, blocks, head []float32) (stepVectors, error) {
	hidden := t.spec.Hidden
	data := make([]float32, 0, stepVectorCount*hidden)
	part := func(values []float32, index int, offset float32) {
		for _, value := range values[index*hidden : (index+1)*hidden] {
			data = append(data, value+offset)
		}
	}
	part(blocks, attentionShift, 0)
	part(blocks, attentionScale, 1)
	part(blocks, attentionGate, 0)
	part(blocks, feedShift, 0)
	part(blocks, feedScale, 1)
	part(blocks, feedGate, 0)
	part(head, 0, 0)
	part(head, 1, 1)
	if err := t.write(ctx, t.vectorsSlab, data); err != nil {
		return stepVectors{}, err
	}
	offset := 0
	next := func() driver.DevicePtr {
		vector := devicemath.ResidentPtr(t.vectorsSlab, offset)
		offset += hidden
		return vector
	}
	return stepVectors{next(), next(), next(), next(), next(), next(), next(), next()}, nil
}

// program is one scope of device work over the trainer's buffers.
type program struct {
	t       *Trainer
	ops     *devicemath.ResidentOps
	vectors stepVectors
	tokens  int
}

func (p *program) launch(name string, count int, arguments ...unsafe.Pointer) error {
	return p.ops.Launch(name, uint32(count), arguments...)
}

// linear writes y = x W^T + scale x D^T U^T for rows of x, keeping x D^T for the VJP.
func (p *program) linear(x, weight driver.DevicePtr, factors *factorPair, rows, in, out int) (y, product driver.DevicePtr, err error) {
	arena, rank := &p.t.arena, p.t.options.Rank
	if y, err = arena.f32(rows * out); err != nil {
		return 0, 0, err
	}
	if err = p.ops.GEMM(false, true, rows, in, out, x, weight, y); err != nil {
		return 0, 0, err
	}
	if factors == nil {
		return y, 0, nil
	}
	if product, err = arena.f32(rows * rank); err != nil {
		return 0, 0, err
	}
	update, err := arena.f32(rows * out)
	if err != nil {
		return 0, 0, err
	}
	if err = p.ops.GEMM(false, true, rows, in, rank, x, factors.down, product); err != nil {
		return 0, 0, err
	}
	if err = p.ops.GEMM(false, true, rows, rank, out, product, factors.up, update); err != nil {
		return 0, 0, err
	}
	if err = p.ops.Scale(update, update, p.t.scale, rows*out); err != nil {
		return 0, 0, err
	}
	return y, product, p.ops.Add(y, update, y, rows*out)
}

// linearBackward writes dx of linear for cotangent dy and accumulates the
// factors' gradients.
func (p *program) linearBackward(dy, x, product, weight driver.DevicePtr, factors *factorPair, rows, in, out int) (dx driver.DevicePtr, err error) {
	arena, rank, ops := &p.t.arena, p.t.options.Rank, p.ops
	if dx, err = arena.f32(rows * in); err != nil {
		return 0, err
	}
	if err = ops.GEMM(false, false, rows, out, in, dy, weight, dx); err != nil {
		return 0, err
	}
	if factors == nil {
		return dx, nil
	}
	dProduct, err := arena.f32(rows * rank)
	if err != nil {
		return 0, err
	}
	if err = ops.GEMM(false, false, rows, out, rank, dy, factors.up, dProduct); err != nil {
		return 0, err
	}
	accumulate := func(gradient driver.DevicePtr, count int, product func(driver.DevicePtr) error) error {
		mark := arena.cursor
		defer func() { arena.cursor = mark }()
		part, err := arena.f32(count)
		if err != nil {
			return err
		}
		if err = product(part); err != nil {
			return err
		}
		if err = ops.Scale(part, part, p.t.scale, count); err != nil {
			return err
		}
		return ops.Add(gradient, part, gradient, count)
	}
	if err = accumulate(factors.gradUp, out*rank, func(part driver.DevicePtr) error {
		return ops.GEMM(true, false, out, rows, rank, dy, product, part)
	}); err != nil {
		return 0, err
	}
	if err = accumulate(factors.gradDown, rank*in, func(part driver.DevicePtr) error {
		return ops.GEMM(true, false, rank, rows, in, dProduct, x, part)
	}); err != nil {
		return 0, err
	}
	return dx, accumulate(dx, rows*in, func(part driver.DevicePtr) error {
		return ops.GEMM(false, false, rows, rank, in, dProduct, factors.down, part)
	})
}

func (p *program) norm(x, weight driver.DevicePtr, rows, width int) (driver.DevicePtr, error) {
	y, err := p.t.arena.f32(rows * width)
	if err != nil {
		return 0, err
	}
	widthU, rowsU, single, epsilon := uint32(width), uint32(rows), uint32(1), float32(normEpsilon)
	return y, p.launch("weighted_rms_norm_f32", rows*rowBlockThreads, unsafe.Pointer(&x), unsafe.Pointer(&weight), unsafe.Pointer(&y),
		unsafe.Pointer(&widthU), unsafe.Pointer(&rowsU), unsafe.Pointer(&single), unsafe.Pointer(&epsilon))
}

func (p *program) normBackward(dy, x, weight driver.DevicePtr, rows, width int) (driver.DevicePtr, error) {
	arena := &p.t.arena
	dx, err := arena.f32(rows * width)
	if err != nil {
		return 0, err
	}
	mark := arena.cursor
	defer func() { arena.cursor = mark }()
	unused, err := arena.f32(width)
	if err != nil {
		return 0, err
	}
	if err = p.ops.Zero(unused, width); err != nil {
		return 0, err
	}
	rowsU, widthU, epsilon := uint32(rows), uint32(width), float32(normEpsilon)
	return dx, p.launch("rms_norm_backward_f32", rows, unsafe.Pointer(&dy), unsafe.Pointer(&x), unsafe.Pointer(&weight),
		unsafe.Pointer(&dx), unsafe.Pointer(&unused), unsafe.Pointer(&rowsU), unsafe.Pointer(&widthU), unsafe.Pointer(&epsilon))
}

// affineRows writes out = x*scale + shift with one vector of each for every row.
func (p *program) affineRows(x, scale, shift driver.DevicePtr, rows int) (driver.DevicePtr, error) {
	hidden := p.t.spec.Hidden
	out, err := p.t.arena.f32(rows * hidden)
	if err != nil {
		return 0, err
	}
	widthU, rowsU := uint32(hidden), uint32(rows)
	return out, p.launch("moss_row_affine_f32", rows*hidden, unsafe.Pointer(&x), unsafe.Pointer(&scale), unsafe.Pointer(&shift),
		unsafe.Pointer(&out), unsafe.Pointer(&widthU), unsafe.Pointer(&rowsU))
}

func (p *program) rotate(x driver.DevicePtr, sign float32) (driver.DevicePtr, error) {
	spec := p.t.spec
	out, err := p.t.arena.f32(p.tokens * spec.Hidden)
	if err != nil {
		return 0, err
	}
	rope, heads, headWidth, tokens := p.t.rope, uint32(spec.Heads), uint32(spec.HeadDim), uint32(p.tokens)
	return out, p.launch("moss_rotate_f32", p.tokens*spec.Heads*spec.HeadDim/rotaryPair, unsafe.Pointer(&x), unsafe.Pointer(&rope), unsafe.Pointer(&out),
		unsafe.Pointer(&heads), unsafe.Pointer(&headWidth), unsafe.Pointer(&tokens), unsafe.Pointer(&sign))
}

// headMajor copies [tokens, heads, head width] rows to one contiguous matrix per head.
func (p *program) headMajor(x driver.DevicePtr) (driver.DevicePtr, error) {
	spec := p.t.spec
	out, err := p.t.arena.f32(p.tokens * spec.Hidden)
	if err != nil {
		return 0, err
	}
	for head := range spec.Heads {
		if err := p.ops.StridedRowCopy(x, devicemath.ResidentPtr(out, head*p.tokens*spec.HeadDim), p.tokens, spec.HeadDim, spec.Hidden, head*spec.HeadDim, spec.HeadDim, 0); err != nil {
			return 0, err
		}
	}
	return out, nil
}

// tokenMajor is headMajor's inverse.
func (p *program) tokenMajor(x driver.DevicePtr) (driver.DevicePtr, error) {
	spec := p.t.spec
	out, err := p.t.arena.f32(p.tokens * spec.Hidden)
	if err != nil {
		return 0, err
	}
	for head := range spec.Heads {
		if err := p.ops.StridedRowCopy(devicemath.ResidentPtr(x, head*p.tokens*spec.HeadDim), out, p.tokens, spec.HeadDim, spec.HeadDim, 0, spec.Hidden, head*spec.HeadDim); err != nil {
			return 0, err
		}
	}
	return out, nil
}

// blockState is what a block's backward reads of its forward.
type blockState struct {
	y, q, k, v, qProduct, kProduct, vProduct driver.DevicePtr
	qRaw, kRaw                               driver.DevicePtr
	qHead, kHead, vHead, probabilities       driver.DevicePtr
	attended, oProduct                       driver.DevicePtr
	afterAttention, feedInput                driver.DevicePtr
	fused, fusedProduct, activated           driver.DevicePtr
	downProduct                              driver.DevicePtr
}

// block runs one joint block from x, writing next (the checkpoint after it) and
// returning its state in the arena.
func (p *program) block(layer int, x, next driver.DevicePtr) (s blockState, err error) {
	t, ops, v := p.t, p.ops, p.vectors
	weights, tokens, hidden, feed := &t.blocks[layer], p.tokens, t.spec.Hidden, t.spec.Intermediate
	spec := t.spec
	factors := &t.factors[layer]
	normed, err := p.norm(x, weights.norm1, tokens, hidden)
	if err != nil {
		return s, err
	}
	if s.y, err = p.affineRows(normed, v.attentionScale, v.attentionShift, tokens); err != nil {
		return s, err
	}
	if s.qRaw, s.qProduct, err = p.linear(s.y, weights.projected[projectionQ], &factors[projectionQ], tokens, hidden, hidden); err != nil {
		return s, err
	}
	if s.kRaw, s.kProduct, err = p.linear(s.y, weights.projected[projectionK], &factors[projectionK], tokens, hidden, hidden); err != nil {
		return s, err
	}
	if s.v, s.vProduct, err = p.linear(s.y, weights.projected[projectionV], &factors[projectionV], tokens, hidden, hidden); err != nil {
		return s, err
	}
	if s.q, err = p.norm(s.qRaw, weights.qNorm, tokens*spec.Heads, spec.HeadDim); err != nil {
		return s, err
	}
	if s.k, err = p.norm(s.kRaw, weights.kNorm, tokens*spec.Heads, spec.HeadDim); err != nil {
		return s, err
	}
	rotatedQ, err := p.rotate(s.q, 1)
	if err != nil {
		return s, err
	}
	rotatedK, err := p.rotate(s.k, 1)
	if err != nil {
		return s, err
	}
	if s.qHead, err = p.headMajor(rotatedQ); err != nil {
		return s, err
	}
	if s.kHead, err = p.headMajor(rotatedK); err != nil {
		return s, err
	}
	if s.vHead, err = p.headMajor(s.v); err != nil {
		return s, err
	}
	square := tokens * tokens
	if s.probabilities, err = t.arena.f32(spec.Heads * square); err != nil {
		return s, err
	}
	attendedHeads, err := t.arena.f32(tokens * hidden)
	if err != nil {
		return s, err
	}
	scale := float32(1 / math.Sqrt(float64(spec.HeadDim)))
	mark := t.arena.cursor
	scores, err := t.arena.f32(square)
	if err != nil {
		return s, err
	}
	matrix := tokens * spec.HeadDim
	for head := range spec.Heads {
		if err = ops.GEMM(false, true, tokens, spec.HeadDim, tokens, devicemath.ResidentPtr(s.qHead, head*matrix), devicemath.ResidentPtr(s.kHead, head*matrix), scores); err != nil {
			return s, err
		}
		if err = ops.Scale(scores, scores, scale, square); err != nil {
			return s, err
		}
		probability := devicemath.ResidentPtr(s.probabilities, head*square)
		widthU, rowsU := uint32(tokens), uint32(tokens)
		if err = p.launch("softmax_f32", tokens, unsafe.Pointer(&scores), unsafe.Pointer(&probability), unsafe.Pointer(&widthU), unsafe.Pointer(&rowsU)); err != nil {
			return s, err
		}
		if err = ops.GEMM(false, false, tokens, tokens, spec.HeadDim, probability, devicemath.ResidentPtr(s.vHead, head*matrix), devicemath.ResidentPtr(attendedHeads, head*matrix)); err != nil {
			return s, err
		}
	}
	t.arena.cursor = mark
	if s.attended, err = p.tokenMajor(attendedHeads); err != nil {
		return s, err
	}
	output, product, err := p.linear(s.attended, weights.projected[projectionO], &factors[projectionO], tokens, hidden, hidden)
	if err != nil {
		return s, err
	}
	s.oProduct = product
	gated, err := p.affineRows(output, v.attentionGate, t.zero, tokens)
	if err != nil {
		return s, err
	}
	if s.afterAttention, err = t.arena.f32(tokens * hidden); err != nil {
		return s, err
	}
	if err = ops.Add(x, gated, s.afterAttention, tokens*hidden); err != nil {
		return s, err
	}
	normed, err = p.norm(s.afterAttention, weights.norm2, tokens, hidden)
	if err != nil {
		return s, err
	}
	if s.feedInput, err = p.affineRows(normed, v.feedScale, v.feedShift, tokens); err != nil {
		return s, err
	}
	if s.fused, s.fusedProduct, err = p.linear(s.feedInput, weights.projected[projectionUpGate], &factors[projectionUpGate], tokens, hidden, gatedFeed*feed); err != nil {
		return s, err
	}
	if s.activated, err = t.arena.f32(tokens * feed); err != nil {
		return s, err
	}
	widthU, rowsU := uint32(feed), uint32(tokens)
	if err = p.launch("moss_situ_f32", tokens*feed, unsafe.Pointer(&s.fused), unsafe.Pointer(&s.activated), unsafe.Pointer(&widthU), unsafe.Pointer(&rowsU)); err != nil {
		return s, err
	}
	down, product, err := p.linear(s.activated, weights.projected[projectionDown], &factors[projectionDown], tokens, feed, hidden)
	if err != nil {
		return s, err
	}
	s.downProduct = product
	gated, err = p.affineRows(down, v.feedGate, t.zero, tokens)
	if err != nil {
		return s, err
	}
	return s, ops.Add(s.afterAttention, gated, next, tokens*hidden)
}

// blockBackward writes the cotangent of a block's input from the cotangent of
// its output, accumulating its adapters' gradients.
func (p *program) blockBackward(layer int, x driver.DevicePtr, s blockState, dNext, dInput driver.DevicePtr) (err error) {
	t, ops, v := p.t, p.ops, p.vectors
	weights, tokens, hidden, feed := &t.blocks[layer], p.tokens, t.spec.Hidden, t.spec.Intermediate
	spec := t.spec
	factors := &t.factors[layer]
	dGated, err := p.affineRows(dNext, v.feedGate, t.zero, tokens)
	if err != nil {
		return err
	}
	dActivated, err := p.linearBackward(dGated, s.activated, s.downProduct, weights.projected[projectionDown], &factors[projectionDown], tokens, feed, hidden)
	if err != nil {
		return err
	}
	dFused, err := t.arena.f32(tokens * gatedFeed * feed)
	if err != nil {
		return err
	}
	widthU, rowsU := uint32(feed), uint32(tokens)
	if err = p.launch("moss_situ_backward_f32", tokens*feed, unsafe.Pointer(&s.fused), unsafe.Pointer(&dActivated), unsafe.Pointer(&dFused), unsafe.Pointer(&widthU), unsafe.Pointer(&rowsU)); err != nil {
		return err
	}
	dFeedInput, err := p.linearBackward(dFused, s.feedInput, s.fusedProduct, weights.projected[projectionUpGate], &factors[projectionUpGate], tokens, hidden, gatedFeed*feed)
	if err != nil {
		return err
	}
	dNormed, err := p.affineRows(dFeedInput, v.feedScale, t.zero, tokens)
	if err != nil {
		return err
	}
	dFromNorm, err := p.normBackward(dNormed, s.afterAttention, weights.norm2, tokens, hidden)
	if err != nil {
		return err
	}
	dAfter, err := t.arena.f32(tokens * hidden)
	if err != nil {
		return err
	}
	if err = ops.Add(dNext, dFromNorm, dAfter, tokens*hidden); err != nil {
		return err
	}
	dOutput, err := p.affineRows(dAfter, v.attentionGate, t.zero, tokens)
	if err != nil {
		return err
	}
	dAttended, err := p.linearBackward(dOutput, s.attended, s.oProduct, weights.projected[projectionO], &factors[projectionO], tokens, hidden, hidden)
	if err != nil {
		return err
	}
	dHeads, err := p.headMajor(dAttended)
	if err != nil {
		return err
	}
	matrix, square := tokens*spec.HeadDim, tokens*tokens
	dQHead, err := t.arena.f32(tokens * hidden)
	if err != nil {
		return err
	}
	dKHead, err := t.arena.f32(tokens * hidden)
	if err != nil {
		return err
	}
	dVHead, err := t.arena.f32(tokens * hidden)
	if err != nil {
		return err
	}
	mark := t.arena.cursor
	dScores, err := t.arena.f32(square)
	if err != nil {
		return err
	}
	dProbability, err := t.arena.f32(square)
	if err != nil {
		return err
	}
	scale := 1 / math.Sqrt(float64(spec.HeadDim))
	for head := range spec.Heads {
		at := func(pointer driver.DevicePtr) driver.DevicePtr { return devicemath.ResidentPtr(pointer, head*matrix) }
		if err = ops.AttentionCoreBackwardWithScratch(at(s.qHead), at(s.kHead), at(s.vHead), devicemath.ResidentPtr(s.probabilities, head*square),
			at(dHeads), at(dQHead), at(dKHead), at(dVHead), dScores, dProbability, tokens, spec.HeadDim, scale); err != nil {
			return err
		}
	}
	t.arena.cursor = mark
	dQRotated, err := p.tokenMajor(dQHead)
	if err != nil {
		return err
	}
	dKRotated, err := p.tokenMajor(dKHead)
	if err != nil {
		return err
	}
	dV, err := p.tokenMajor(dVHead)
	if err != nil {
		return err
	}
	dQNormed, err := p.rotate(dQRotated, -1)
	if err != nil {
		return err
	}
	dKNormed, err := p.rotate(dKRotated, -1)
	if err != nil {
		return err
	}
	dQ, err := p.normBackward(dQNormed, s.qRaw, weights.qNorm, tokens*spec.Heads, spec.HeadDim)
	if err != nil {
		return err
	}
	dK, err := p.normBackward(dKNormed, s.kRaw, weights.kNorm, tokens*spec.Heads, spec.HeadDim)
	if err != nil {
		return err
	}
	dyQ, err := p.linearBackward(dQ, s.y, s.qProduct, weights.projected[projectionQ], &factors[projectionQ], tokens, hidden, hidden)
	if err != nil {
		return err
	}
	dyK, err := p.linearBackward(dK, s.y, s.kProduct, weights.projected[projectionK], &factors[projectionK], tokens, hidden, hidden)
	if err != nil {
		return err
	}
	dyV, err := p.linearBackward(dV, s.y, s.vProduct, weights.projected[projectionV], &factors[projectionV], tokens, hidden, hidden)
	if err != nil {
		return err
	}
	if err = ops.Add(dyQ, dyK, dyQ, tokens*hidden); err != nil {
		return err
	}
	if err = ops.Add(dyQ, dyV, dyQ, tokens*hidden); err != nil {
		return err
	}
	dNormed, err = p.affineRows(dyQ, v.attentionScale, t.zero, tokens)
	if err != nil {
		return err
	}
	dFromNorm, err = p.normBackward(dNormed, x, weights.norm1, tokens, hidden)
	if err != nil {
		return err
	}
	return ops.Add(dAfter, dFromNorm, dInput, tokens*hidden)
}

func (t *Trainer) checkpoint(layer int) driver.DevicePtr {
	return devicemath.ResidentPtr(t.checkpoints, layer*t.geometry.tokens()*t.spec.Hidden)
}

// forward runs the transformer from noised latent rows, keeping every block's
// input, and leaves the prediction rows on the host.
func (t *Trainer) forward(ctx context.Context, example Example, noised []float32, timestep float64) (prediction []float32, vectors stepVectors, err error) {
	if err = ctx.Err(); err != nil {
		return nil, vectors, err
	}
	geometry := example.Geometry
	if err = t.useGeometry(ctx, geometry); err != nil {
		return nil, vectors, err
	}
	image, tokens, hidden, channels := geometry.image(), geometry.tokens(), t.spec.Hidden, t.spec.InChannels
	if len(noised) != image*channels || len(example.Refined) != geometry.Text*hidden {
		return nil, vectors, errors.New("mosstrain: example extents differ from its geometry")
	}
	blocks, head, err := t.modulationVectors(timestep)
	if err != nil {
		return nil, vectors, err
	}
	if vectors, err = t.uploadVectors(ctx, blocks, head); err != nil {
		return nil, vectors, err
	}
	if err = t.write(ctx, t.noised, noised); err != nil {
		return nil, vectors, err
	}
	if err = t.write(ctx, t.refined, example.Refined); err != nil {
		return nil, vectors, err
	}
	prediction = make([]float32, image*channels)
	err = t.session.Run(func(ops *devicemath.ResidentOps) error {
		p := &program{t: t, ops: ops, vectors: vectors, tokens: tokens}
		t.arena.cursor = 0
		embedded, _, err := p.linear(t.noised, t.videoWeight, nil, image, channels, hidden)
		if err != nil {
			return err
		}
		video, err := p.affineRows(embedded, t.one, t.videoBias, image)
		if err != nil {
			return err
		}
		first := t.checkpoint(0)
		if err = ops.StridedRowCopy(video, first, image, hidden, hidden, 0, hidden, 0); err != nil {
			return err
		}
		if err = ops.StridedRowCopy(t.refined, devicemath.ResidentPtr(first, image*hidden), geometry.Text, hidden, hidden, 0, hidden, 0); err != nil {
			return err
		}
		t.arena.cursor = 0
		for layer := range t.spec.Layers {
			if _, err = p.block(layer, t.checkpoint(layer), t.checkpoint(layer+1)); err != nil {
				return err
			}
			t.arena.cursor = 0
			if err = ops.Flush(); err != nil {
				return err
			}
		}
		normed, err := p.norm(t.checkpoint(t.spec.Layers), t.finalNorm, image, hidden)
		if err != nil {
			return err
		}
		modulated, err := p.affineRows(normed, vectors.headScale, vectors.headShift, image)
		if err != nil {
			return err
		}
		output, _, err := p.linear(modulated, t.finalLinear, nil, image, hidden, channels)
		if err != nil {
			return err
		}
		return ops.DownloadF32(output, prediction)
	})
	if err != nil {
		return nil, vectors, err
	}
	return prediction, vectors, nil
}

// backward accumulates the gradient of the factors for a cotangent of the
// prediction, running each block again from its kept input.
func (t *Trainer) backward(ctx context.Context, cotangent []float32, vectors stepVectors) error {
	if err := t.write(ctx, t.cotangent, cotangent); err != nil {
		return err
	}
	geometry := t.geometry
	image, tokens, hidden, channels := geometry.image(), geometry.tokens(), t.spec.Hidden, t.spec.InChannels
	return t.session.Run(func(ops *devicemath.ResidentOps) error {
		p := &program{t: t, ops: ops, vectors: vectors, tokens: tokens}
		t.arena.cursor = 0
		current, err := t.arena.f32(tokens * hidden)
		if err != nil {
			return err
		}
		previous, err := t.arena.f32(tokens * hidden)
		if err != nil {
			return err
		}
		next, err := t.arena.f32(tokens * hidden)
		if err != nil {
			return err
		}
		mark := t.arena.cursor
		dModulated, err := p.linearBackward(t.cotangent, 0, 0, t.finalLinear, nil, image, hidden, channels)
		if err != nil {
			return err
		}
		dNormed, err := p.affineRows(dModulated, vectors.headScale, t.zero, image)
		if err != nil {
			return err
		}
		top := t.checkpoint(t.spec.Layers)
		dTop, err := p.normBackward(dNormed, top, t.finalNorm, image, hidden)
		if err != nil {
			return err
		}
		if err = ops.Zero(current, tokens*hidden); err != nil {
			return err
		}
		if err = ops.StridedRowCopy(dTop, current, image, hidden, hidden, 0, hidden, 0); err != nil {
			return err
		}
		t.arena.cursor = mark
		for layer := t.spec.Layers - 1; layer >= 0; layer-- {
			if err = ctx.Err(); err != nil {
				return err
			}
			x := t.checkpoint(layer)
			state, err := p.block(layer, x, next)
			if err != nil {
				return err
			}
			if err = p.blockBackward(layer, x, state, current, previous); err != nil {
				return err
			}
			current, previous = previous, current
			t.arena.cursor = mark
			if err = ops.Flush(); err != nil {
				return err
			}
		}
		return nil
	})
}

// Predict runs the transformer from noised latent rows.
func (t *Trainer) Predict(ctx context.Context, example Example, noised []float32, timestep float64) ([]float32, error) {
	prediction, _, err := t.forward(ctx, example, noised, timestep)
	return prediction, err
}

// meanSquared is the mean squared difference and its gradient at prediction.
func meanSquared(prediction, target []float32) (float64, []float32) {
	cotangent := make([]float32, len(prediction))
	var sum float64
	unit := squareSlope / float64(len(prediction))
	for index, value := range prediction {
		difference := float64(value) - float64(target[index])
		sum += difference * difference
		cotangent[index] = float32(unit * difference)
	}
	return sum / float64(len(prediction)), cotangent
}

// Loss is the example's flow-matching loss at a timestep with an adapter as it stands.
func (t *Trainer) Loss(ctx context.Context, example Example, noised, target []float32, timestep float64) (float64, error) {
	prediction, err := t.Predict(ctx, example, noised, timestep)
	if err != nil {
		return 0, err
	}
	loss, _ := meanSquared(prediction, target)
	return loss, nil
}

// Clear zeroes the accumulated gradient.
func (t *Trainer) Clear() error {
	return t.session.Run(func(ops *devicemath.ResidentOps) error {
		return ops.Zero(t.gradSlab, t.pack.ParameterCount())
	})
}

// Accumulate adds an example's gradient at a timestep to the accumulated one
// and returns its loss.
func (t *Trainer) Accumulate(ctx context.Context, example Example, noised, target []float32, timestep float64) (float64, error) {
	if len(target) != len(noised) {
		return 0, errors.New("mosstrain: target extent differs from the noised latent")
	}
	prediction, vectors, err := t.forward(ctx, example, noised, timestep)
	if err != nil {
		return 0, err
	}
	loss, cotangent := meanSquared(prediction, target)
	return loss, t.backward(ctx, cotangent, vectors)
}

// Gather brings the accumulated gradient to the host pack, divided by the
// examples that made it, and returns its norm.
func (t *Trainer) Gather(examples int) (float64, error) {
	_, gradients := t.pack.Slabs()
	if err := t.session.Run(func(ops *devicemath.ResidentOps) error {
		return ops.DownloadF32(t.gradSlab, gradients)
	}); err != nil {
		return 0, err
	}
	if examples > 1 {
		share := 1 / float32(examples)
		for index := range gradients {
			gradients[index] *= share
		}
	}
	return t.pack.GradientL2(), nil
}

// Apply steps the optimizer over the gathered gradient and places the new factors.
func (t *Trainer) Apply(loss, gradientL2 float64) (optimizer.ObservedStepResult, error) {
	result, err := optimizer.Advance(t.stepper, t.config, &t.step, loss, gradientL2)
	if err != nil {
		return result, err
	}
	return result, t.uploadFactors()
}

// Adapter copies the trained factors as a PEFT adapter of the checkpoint's module paths.
func (t *Trainer) Adapter(config peft.Config) peft.Adapter {
	adapter := peft.Adapter{Config: config, Modules: make(map[string]peft.Module, t.spec.Layers*len(projectionNames))}
	at := 0
	for layer := range t.spec.Layers {
		for slot, projection := range projectionNames {
			out, in := projectionShape(t.spec, slot)
			adapter.Modules[moduleName(layer, projection)] = peft.Module{
				Rank: t.options.Rank, In: in, Out: out, Alpha: t.options.Alpha,
				Down: slices.Clone(t.pack.Weights(at)), Up: slices.Clone(t.pack.Weights(at + 1)),
			}
			at += factorsPerModule
		}
	}
	return adapter
}

// OptimizerPlan is the plan the trainer's Muon steps: every adapter factor.
func (t *Trainer) OptimizerPlan() optimizer.Plan { return t.pack.Plan() }

// Snapshot copies the optimizer state the next step continues from.
func (t *Trainer) Snapshot() optimizer.State { return t.stepper.Snapshot() }

// SaveWeights writes the factors at path, the weights Snapshot pairs with.
func (t *Trainer) SaveWeights(path string) error { return t.pack.Save(path) }

// Close releases the trainer's device state.
func (t *Trainer) Close() error {
	var errs []error
	if t.stepper != nil {
		errs = append(errs, t.stepper.Close())
	}
	if t.session != nil {
		errs = append(errs, t.session.Close())
	}
	if t.worker != nil {
		errs = append(errs, t.worker.Close())
	}
	return errors.Join(errs...)
}
