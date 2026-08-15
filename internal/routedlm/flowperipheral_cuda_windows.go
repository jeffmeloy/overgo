//go:build windows

package routedlm

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// DeviceFlowPeripheralSession retains vision and flow-head projection weights.
type DeviceFlowPeripheralSession struct {
	worker    *device.Worker
	cuda      *executor.Executor
	resources device.AllocationSet
	plan      FlowPlan
	image     FlowImagePlan
	patch     flowDeviceProgram
	dense     flowDeviceProgram
	condition flowDeviceProgram
	head      flowDeviceProgram
	patchIn   []float32
	denseIn   []float32
	headIn    []float32
	timeIn    []float32
	noiseIn   []float32
	denomIn   [1]float32
	closed    bool
}

type flowDeviceProgram struct {
	compiled *executor.CompiledGraph
	inputs   *executor.DeviceInputs
	output   *tensor.Tensor
	dynamic  []*tensor.Tensor
	pointers []driver.DevicePtr
}

// NewDeviceFlowPeripheralSession compiles request-geometry projections once.
func NewDeviceFlowPeripheralSession(
	ctx context.Context,
	worker *device.Worker,
	cuda *executor.Executor,
	plan FlowPlan,
	image FlowImagePlan,
	vision VisionEmbedderWeights,
	terminal FlowTerminalWeights,
	finalNorm []float32,
	normEpsilon float64,
) (_ *DeviceFlowPeripheralSession, result error) {
	if worker == nil || cuda == nil || image.Tokens <= 0 || len(finalNorm) != plan.Hidden || normEpsilon <= 0 {
		return nil, errors.New("routed lm flow peripheral: invalid runtime or geometry")
	}
	session := &DeviceFlowPeripheralSession{
		worker: worker, cuda: cuda, resources: device.NewAllocationSet(worker), plan: plan, image: image,
		patchIn: make([]float32, image.GridHeight*image.GridWidth*plan.VisionChannels*plan.VisionPatch*plan.VisionPatch),
		denseIn: make([]float32, image.Tokens*plan.VisionHidden*plan.ImageMerge*plan.ImageMerge),
		headIn:  make([]float32, image.Tokens*plan.Hidden),
		timeIn:  make([]float32, plan.FrequencyDim), noiseIn: make([]float32, plan.FrequencyDim),
	}
	defer func() {
		if result != nil {
			_ = session.Close(context.WithoutCancel(ctx))
		}
	}()
	var statics []flowStaticInput
	if session.patch, statics, result = buildFlowVisionPatchProgram(plan, image); result != nil {
		return nil, result
	}
	if result = session.prepare(ctx, &session.patch, statics, []flowStaticValue{
		{statics[0].node, bf16MatrixBytes(vision.Patch)}, {statics[1].node, driver.Bytes(vision.PatchBias)},
	}); result != nil {
		return nil, result
	}
	if session.dense, statics, result = buildFlowVisionDenseProgram(plan, image); result != nil {
		return nil, result
	}
	if result = session.prepare(ctx, &session.dense, statics, []flowStaticValue{
		{statics[0].node, bf16MatrixBytes(vision.Dense)}, {statics[1].node, driver.Bytes(vision.DenseBias)},
	}); result != nil {
		return nil, result
	}
	if session.condition, statics, result = buildFlowConditionProgram(plan); result != nil {
		return nil, result
	}
	if result = session.prepare(ctx, &session.condition, statics, []flowStaticValue{
		{statics[0].node, bf16MatrixBytes(terminal.Timestep.W0)}, {statics[1].node, driver.Bytes(terminal.Timestep.B0)},
		{statics[2].node, bf16MatrixBytes(terminal.Timestep.W2)}, {statics[3].node, driver.Bytes(terminal.Timestep.B2)},
		{statics[4].node, bf16MatrixBytes(terminal.NoiseScale.W0)}, {statics[5].node, driver.Bytes(terminal.NoiseScale.B0)},
		{statics[6].node, bf16MatrixBytes(terminal.NoiseScale.W2)}, {statics[7].node, driver.Bytes(terminal.NoiseScale.B2)},
	}); result != nil {
		return nil, result
	}
	if session.head, statics, result = buildFlowHeadProgram(plan, image, normEpsilon); result != nil {
		return nil, result
	}
	if result = session.prepare(ctx, &session.head, statics, []flowStaticValue{
		{statics[0].node, driver.Bytes(finalNorm)},
		{statics[1].node, bf16MatrixBytes(terminal.Head.W0)}, {statics[2].node, driver.Bytes(terminal.Head.B0)},
		{statics[3].node, bf16MatrixBytes(terminal.Head.W2)}, {statics[4].node, driver.Bytes(terminal.Head.B2)},
	}); result != nil {
		return nil, result
	}
	return session, nil
}

func buildFlowConditionProgram(plan FlowPlan) (flowDeviceProgram, []flowStaticInput, error) {
	b := tensor.NewBuilder()
	b.SetMulMatCompute(tensor.MulMatComputeBF16TensorCore)
	makeMLP := func(name string) (*tensor.Tensor, []*tensor.Tensor) {
		w0 := b.Input(name+"_0_weight", dtype.BF16, tensor.MustShape(uint64(plan.FrequencyDim), uint64(plan.Hidden)))
		b0 := b.Input(name+"_0_bias", dtype.F32, tensor.MustShape(uint64(plan.Hidden), 1))
		w2 := b.Input(name+"_2_weight", dtype.BF16, tensor.MustShape(uint64(plan.Hidden), uint64(plan.Hidden)))
		b2 := b.Input(name+"_2_bias", dtype.F32, tensor.MustShape(uint64(plan.Hidden), 1))
		input := b.Input(name+"_frequency", dtype.F32, tensor.MustShape(uint64(plan.FrequencyDim), 1))
		mid := b.BF16Round(b.Add(b.MulMat(w0, input), b0))
		mid = b.BF16Round(b.SiLU(mid))
		return b.BF16Round(b.Add(b.MulMat(w2, mid), b2)), []*tensor.Tensor{w0, b0, w2, b2, input}
	}
	timestep, timestepInputs := makeMLP("timestep")
	noise, noiseInputs := makeMLP("noise")
	output := b.BF16Round(b.Add(timestep, noise))
	if err := b.Err(); err != nil {
		return flowDeviceProgram{}, nil, err
	}
	compiled, err := executor.Compile(output)
	statics := make([]flowStaticInput, 0, 8)
	for _, nodes := range [][]*tensor.Tensor{timestepInputs[:4], noiseInputs[:4]} {
		for _, node := range nodes {
			statics = append(statics, flowStaticInput{node})
		}
	}
	return flowDeviceProgram{
		compiled: compiled, output: output, dynamic: []*tensor.Tensor{timestepInputs[4], noiseInputs[4]},
	}, statics, err
}

type flowStaticInput struct{ node *tensor.Tensor }
type flowStaticValue struct {
	node *tensor.Tensor
	raw  []byte
}

func buildFlowVisionPatchProgram(plan FlowPlan, image FlowImagePlan) (flowDeviceProgram, []flowStaticInput, error) {
	patchIn := plan.VisionChannels * plan.VisionPatch * plan.VisionPatch
	rows := image.GridHeight * image.GridWidth
	b := tensor.NewBuilder()
	b.SetMulMatCompute(tensor.MulMatComputeBF16TensorCore)
	weight := b.Input("vision_patch_weight", dtype.BF16, tensor.MustShape(uint64(patchIn), uint64(plan.VisionHidden)))
	bias := b.Input("vision_patch_bias", dtype.F32, tensor.MustShape(uint64(plan.VisionHidden), 1))
	input := b.Input("vision_patch_input", dtype.F32, tensor.MustShape(uint64(patchIn), uint64(rows)))
	output := b.BF16Round(b.GELUErf(b.BF16Round(b.Add(b.MulMat(weight, input), bias))))
	if err := b.Err(); err != nil {
		return flowDeviceProgram{}, nil, err
	}
	compiled, err := executor.Compile(output)
	return flowDeviceProgram{compiled: compiled, output: output, dynamic: []*tensor.Tensor{input}},
		[]flowStaticInput{{weight}, {bias}}, err
}

func buildFlowVisionDenseProgram(plan FlowPlan, image FlowImagePlan) (flowDeviceProgram, []flowStaticInput, error) {
	denseIn := plan.VisionHidden * plan.ImageMerge * plan.ImageMerge
	b := tensor.NewBuilder()
	b.SetMulMatCompute(tensor.MulMatComputeBF16TensorCore)
	weight := b.Input("vision_dense_weight", dtype.BF16, tensor.MustShape(uint64(denseIn), uint64(plan.Hidden)))
	bias := b.Input("vision_dense_bias", dtype.F32, tensor.MustShape(uint64(plan.Hidden), 1))
	input := b.Input("vision_dense_input", dtype.F32, tensor.MustShape(uint64(denseIn), uint64(image.Tokens)))
	output := b.BF16Round(b.Add(b.MulMat(weight, input), bias))
	if err := b.Err(); err != nil {
		return flowDeviceProgram{}, nil, err
	}
	compiled, err := executor.Compile(output)
	return flowDeviceProgram{compiled: compiled, output: output, dynamic: []*tensor.Tensor{input}},
		[]flowStaticInput{{weight}, {bias}}, err
}

func buildFlowHeadProgram(plan FlowPlan, image FlowImagePlan, normEpsilon float64) (flowDeviceProgram, []flowStaticInput, error) {
	rows := uint64(image.Tokens)
	b := tensor.NewBuilder()
	b.SetMulMatCompute(tensor.MulMatComputeBF16TensorCore)
	norm := b.Input("generation_final_norm", dtype.F32, tensor.MustShape(uint64(plan.Hidden), 1))
	w0 := b.Input("flow_head_0_weight", dtype.BF16, tensor.MustShape(uint64(plan.Hidden), uint64(plan.Hidden)))
	b0 := b.Input("flow_head_0_bias", dtype.F32, tensor.MustShape(uint64(plan.Hidden), 1))
	w2 := b.Input("flow_head_2_weight", dtype.BF16, tensor.MustShape(uint64(plan.Hidden), uint64(plan.FlowDim)))
	b2 := b.Input("flow_head_2_bias", dtype.F32, tensor.MustShape(uint64(plan.FlowDim), 1))
	hidden := b.Input("flow_hidden", dtype.F32, tensor.MustShape(uint64(plan.Hidden), rows))
	z := b.Input("flow_state", dtype.F32, tensor.MustShape(uint64(plan.FlowDim), rows))
	denominator := b.Input("flow_denominator", dtype.F32, tensor.MustShape(1, 1))
	final := b.BF16Round(b.Multiply(b.BF16Round(b.RMSNorm(hidden, float32(normEpsilon))), norm))
	mid := b.BF16Round(b.Add(b.MulMat(w0, final), b0))
	mid = b.BF16Round(b.GELUErf(mid))
	out := b.BF16Round(b.Add(b.MulMat(w2, mid), b2))
	output := b.BF16Round(b.Divide(b.Add(out, b.Scale(z, -1)), denominator))
	if err := b.Err(); err != nil {
		return flowDeviceProgram{}, nil, err
	}
	compiled, err := executor.Compile(output)
	return flowDeviceProgram{compiled: compiled, output: output, dynamic: []*tensor.Tensor{hidden, z, denominator}},
		[]flowStaticInput{{norm}, {w0}, {b0}, {w2}, {b2}}, err
}

func (s *DeviceFlowPeripheralSession) prepare(
	ctx context.Context,
	program *flowDeviceProgram,
	statics []flowStaticInput,
	values []flowStaticValue,
) error {
	if program.compiled == nil || len(statics) != len(values) {
		return errors.New("routed lm flow peripheral: invalid program")
	}
	if err := s.cuda.PrepareCompiled(ctx, program.compiled); err != nil {
		return err
	}
	program.inputs = program.compiled.NewDeviceInputs()
	for index := range statics {
		if statics[index].node != values[index].node {
			return errors.New("routed lm flow peripheral: static binding order differs")
		}
		slot, err := compiledInputSlot(program.compiled, statics[index].node)
		if err != nil {
			return err
		}
		pointer, err := s.resources.Upload(ctx, values[index].raw)
		if err != nil {
			return err
		}
		program.inputs.Pointers[slot] = pointer
	}
	program.pointers = make([]driver.DevicePtr, len(program.dynamic))
	for index, node := range program.dynamic {
		slot, err := compiledInputSlot(program.compiled, node)
		if err != nil {
			return err
		}
		bytes, err := node.Shape.Bytes(dtype.F32)
		if err != nil {
			return err
		}
		program.pointers[index], err = s.resources.Allocate(ctx, bytes)
		if err != nil {
			return err
		}
		program.inputs.Pointers[slot] = program.pointers[index]
	}
	return nil
}

func (s *DeviceFlowPeripheralSession) run(ctx context.Context, program flowDeviceProgram, values ...[]float32) ([]float32, error) {
	if s == nil || s.closed || len(program.dynamic) != len(values) {
		return nil, errors.New("routed lm flow peripheral: invalid execution")
	}
	for index, node := range program.dynamic {
		elements, err := node.Shape.Elements()
		if err != nil || uint64(len(values[index])) != elements {
			return nil, fmt.Errorf("routed lm flow peripheral: input %q elements=%d want=%d", node.Name, len(values[index]), elements)
		}
	}
	if err := s.worker.Do(ctx, func(state *device.State) error {
		for index := range values {
			if err := state.Driver.MemcpyHtoD(program.pointers[index], driver.Bytes(values[index])); err != nil {
				return fmt.Errorf("input %q: %w", program.dynamic[index].Name, err)
			}
		}
		return nil
	}); err != nil {
		return nil, err
	}
	result, err := s.cuda.ExecuteCompiled(ctx, program.compiled, nil, program.inputs)
	if err != nil {
		return nil, err
	}
	return result[program.output].Data, nil
}

// VisionEmbedTokens executes both BF16 projections on CUDA.
func (s *DeviceFlowPeripheralSession) VisionEmbedTokens(ctx context.Context, planar []float32) ([]float32, error) {
	if s == nil || s.closed || len(planar) != s.plan.VisionChannels*s.image.Width*s.image.Height {
		return nil, errors.New("routed lm flow peripheral: invalid planar image")
	}
	patchIn := s.plan.VisionChannels * s.plan.VisionPatch * s.plan.VisionPatch
	patchRows := s.image.GridHeight * s.image.GridWidth
	for patch := range patchRows {
		patchVector(s.patchIn[patch*patchIn:(patch+1)*patchIn], planar, patch, s.image, s.plan.VisionChannels)
	}
	patches, err := s.run(ctx, s.patch, s.patchIn)
	if err != nil {
		return nil, err
	}
	for patch := range patchRows {
		ropePatch2D(
			patches[patch*s.plan.VisionHidden:(patch+1)*s.plan.VisionHidden],
			patch/s.image.GridWidth, patch%s.image.GridWidth, s.plan.VisionRopeTheta,
		)
	}
	denseIn := s.plan.VisionHidden * s.plan.ImageMerge * s.plan.ImageMerge
	for token := range s.image.Tokens {
		denseVector(s.denseIn[token*denseIn:(token+1)*denseIn], patches, token, s.image, s.plan.VisionHidden)
	}
	return s.run(ctx, s.dense, s.denseIn)
}

// FlowConditionRow applies both scalar embedders on CUDA.
func (s *DeviceFlowPeripheralSession) FlowConditionRow(
	ctx context.Context,
	timestep, normalizedNoise float64,
) ([]float32, error) {
	if timestep < 0 || timestep >= 1 {
		return nil, fmt.Errorf("routed lm flow peripheral: timestep %g outside [0,1)", timestep)
	}
	timestepFrequency, err := SinusoidalEmbedding([]float64{timestep}, s.plan.FrequencyDim, s.plan.SinusoidalPeriod, 1)
	if err != nil {
		return nil, err
	}
	noiseFrequency, err := SinusoidalEmbedding([]float64{normalizedNoise}, s.plan.FrequencyDim, s.plan.SinusoidalPeriod, 1)
	if err != nil {
		return nil, err
	}
	bf16RoundSlice(timestepFrequency)
	bf16RoundSlice(noiseFrequency)
	copy(s.timeIn, timestepFrequency)
	copy(s.noiseIn, noiseFrequency)
	return s.run(ctx, s.condition, s.timeIn, s.noiseIn)
}

// FlowHeadVelocity applies final norm and the BF16 flow head on CUDA.
func (s *DeviceFlowPeripheralSession) FlowHeadVelocity(
	ctx context.Context,
	hidden, z []float32,
	timestep float64,
) ([]float32, error) {
	if timestep < 0 || timestep >= 1 {
		return nil, fmt.Errorf("routed lm flow peripheral: timestep %g outside [0,1)", timestep)
	}
	if len(hidden) != len(s.headIn) {
		return nil, fmt.Errorf("routed lm flow peripheral: hidden elements=%d want=%d", len(hidden), len(s.headIn))
	}
	copy(s.headIn, hidden)
	s.denomIn[0] = float32(max(1-timestep, s.plan.TEps))
	return s.run(ctx, s.head, s.headIn, z, s.denomIn[:])
}

// Close releases retained projection weights.
func (s *DeviceFlowPeripheralSession) Close(ctx context.Context) error {
	if s == nil || s.closed {
		return nil
	}
	s.closed = true
	return s.resources.Close(context.WithoutCancel(ctx))
}
