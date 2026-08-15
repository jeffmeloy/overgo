//go:build windows

package devicemath

import (
	"context"
	"errors"
	"math"
	"unsafe"

	"overgo/internal/cuda/cublas"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
)

// ResidentOps shares one CUDA/cuBLAS scope across a composed device program.
type ResidentOps struct {
	session   *cudaBLAS
	functions map[string]driver.Function
}

// ResidentOpsSession retains one module and cuBLAS handle across programs.
type ResidentOpsSession struct {
	worker      *device.Worker
	allocations device.AllocationSet
	library     *cublas.Library
	handle      cublas.Handle
	module      driver.Module
	closed      bool
}

func NewResidentOpsSession(worker *device.Worker) (*ResidentOpsSession, error) {
	if worker == nil {
		return nil, errors.New("resident ops session: worker absent")
	}
	session := &ResidentOpsSession{worker: worker, allocations: device.NewAllocationSet(worker)}
	err := worker.Do(context.Background(), func(state *device.State) (result error) {
		if err := kernel.ValidateAssets(); err != nil {
			return err
		}
		library, err := cublas.Open()
		if err != nil {
			return err
		}
		session.library = library
		handle, err := library.Create()
		if err != nil {
			_ = library.Close()
			session.library = nil
			return err
		}
		session.handle = handle
		if err := library.SetStream(handle, state.Stream); err != nil {
			_ = library.Destroy(handle)
			_ = library.Close()
			session.library, session.handle = nil, 0
			return err
		}
		module, err := state.Driver.ModuleLoadData(kernel.OpsF32PTX)
		if err != nil {
			_ = library.Destroy(handle)
			_ = library.Close()
			session.library, session.handle = nil, 0
			return err
		}
		session.module = module
		return nil
	})
	if err != nil {
		return nil, err
	}
	return session, nil
}

func (s *ResidentOpsSession) Allocate(ctx context.Context, bytes uint64) (driver.DevicePtr, error) {
	if s == nil || s.closed {
		return 0, errors.New("resident ops session: unavailable")
	}
	return s.allocations.Allocate(ctx, bytes)
}

func (s *ResidentOpsSession) Upload(ctx context.Context, bytes []byte) (driver.DevicePtr, error) {
	if s == nil || s.closed {
		return 0, errors.New("resident ops session: unavailable")
	}
	return s.allocations.Upload(ctx, bytes)
}

func (s *ResidentOpsSession) Do(ctx context.Context, run func(*device.State) error) error {
	if s == nil || s.closed || run == nil {
		return errors.New("resident ops session: unavailable")
	}
	return s.worker.Do(ctx, run)
}

func (s *ResidentOpsSession) Run(run func(*ResidentOps) error) error {
	if s == nil || s.closed || s.worker == nil || s.library == nil || s.handle == 0 || s.module == 0 || run == nil {
		return errors.New("resident ops session: unavailable")
	}
	return s.worker.Do(context.Background(), func(state *device.State) (result error) {
		scope := &cudaScope{state: state, module: s.module}
		defer func() { result = errors.Join(result, scope.close()) }()
		ops := &ResidentOps{
			session:   &cudaBLAS{cudaScope: scope, library: s.library, handle: s.handle},
			functions: map[string]driver.Function{},
		}
		if err := run(ops); err != nil {
			return err
		}
		return ops.session.finish()
	})
}

func (s *ResidentOpsSession) Close() error {
	if s == nil || s.closed {
		return nil
	}
	s.closed = true
	result := s.worker.Do(context.Background(), func(state *device.State) error {
		result := errors.Join(state.Driver.ModuleUnload(s.module), s.library.Destroy(s.handle), s.library.Close())
		s.module, s.handle, s.library = 0, 0, nil
		return result
	})
	return errors.Join(result, s.allocations.Close(context.Background()))
}

// ResidentArena reuses one scope-owned device slab by liveness mark.
type ResidentArena struct {
	cursor, capacity int
	pointer          driver.DevicePtr
}

// WithResidentOps runs one composed resident program and synchronizes once.
func WithResidentOps(worker *device.Worker, run func(*ResidentOps) error) error {
	if worker == nil || run == nil {
		return errors.New("resident ops: worker or program absent")
	}
	return withCUDABLAS(worker, func(session *cudaBLAS) error {
		ops := &ResidentOps{session: session, functions: map[string]driver.Function{}}
		if err := run(ops); err != nil {
			return err
		}
		return session.finish()
	})
}

func (o *ResidentOps) function(name string) (driver.Function, error) {
	if function := o.functions[name]; function != 0 {
		return function, nil
	}
	function, err := o.session.function(name)
	if err == nil {
		o.functions[name] = function
	}
	return function, err
}

func (o *ResidentOps) AllocF32(count int) (driver.DevicePtr, error) {
	if o == nil || o.session == nil || count <= 0 {
		return 0, errors.New("resident ops: invalid scratch allocation")
	}
	return o.session.alloc(count)
}

func (o *ResidentOps) NewArena(count int) (*ResidentArena, error) {
	pointer, err := o.AllocF32(count)
	if err != nil {
		return nil, err
	}
	return &ResidentArena{pointer: pointer, capacity: count}, nil
}

func (a *ResidentArena) AllocF32(count int) (driver.DevicePtr, error) {
	if a == nil || a.pointer == 0 || count <= 0 || a.cursor+count > a.capacity {
		return 0, errors.New("resident arena: capacity exceeded")
	}
	pointer := ResidentPtr(a.pointer, a.cursor)
	a.cursor += count
	return pointer, nil
}

func (a *ResidentArena) Mark() int {
	if a == nil {
		return 0
	}
	return a.cursor
}

func (a *ResidentArena) Reset(mark int) error {
	if a == nil || mark < 0 || mark > a.cursor {
		return errors.New("resident arena: invalid liveness mark")
	}
	a.cursor = mark
	return nil
}

func (o *ResidentOps) UploadU32(values []uint32) (driver.DevicePtr, error) {
	if o == nil || o.session == nil || len(values) == 0 {
		return 0, errors.New("resident ops: indices absent")
	}
	pointer, err := o.session.alloc(len(values))
	if err != nil {
		return 0, err
	}
	return pointer, o.session.state.Driver.MemcpyHtoD(pointer, driver.Bytes(values))
}

func (o *ResidentOps) DownloadF32(pointer driver.DevicePtr, values []float32) error {
	if o == nil || o.session == nil || pointer == 0 || len(values) == 0 {
		return errors.New("resident ops: invalid download")
	}
	return o.session.finish(cudaDownload{data: values, pointer: pointer})
}

func (o *ResidentOps) Zero(buffer driver.DevicePtr, count int) error {
	if buffer == 0 || count <= 0 {
		return errors.New("resident ops: invalid clear")
	}
	return o.session.state.Driver.MemsetD32Async(buffer, 0, uint64(count), o.session.state.Stream)
}

func (o *ResidentOps) LinearBackwardT(x, weight, dY, dX, dWeight driver.DevicePtr, rows, in, out int) error {
	if x == 0 || weight == 0 || dY == 0 || dX == 0 || dWeight == 0 || rows <= 0 || in <= 0 || out <= 0 {
		return errors.New("resident ops: invalid linear VJP")
	}
	if err := o.session.gemm(false, false, rows, out, in, dY, weight, dX); err != nil {
		return err
	}
	return o.session.gemm(true, false, out, rows, in, dY, x, dWeight)
}

func (o *ResidentOps) MADNormBackward(incoming, input, output, gradient driver.DevicePtr, rows, width int, epsilon float64) error {
	if incoming == 0 || input == 0 || output == 0 || gradient == 0 || rows <= 0 || width <= 0 || epsilon <= 0 || math.IsNaN(epsilon) || math.IsInf(epsilon, 0) {
		return errors.New("resident ops: invalid MADNorm VJP")
	}
	function, err := o.function("mad_norm_backward_f32")
	if err != nil {
		return err
	}
	widthU, rowsU, epsilonF := uint32(width), uint32(rows), float32(epsilon)
	return o.session.launch1D(function, rowsU,
		unsafe.Pointer(&incoming), unsafe.Pointer(&input), unsafe.Pointer(&output), unsafe.Pointer(&gradient),
		unsafe.Pointer(&widthU), unsafe.Pointer(&rowsU), unsafe.Pointer(&epsilonF))
}

func (o *ResidentOps) AttentionCoreBackward(
	q, k, v, probability, dOut, dQ, dK, dV, dScores driver.DevicePtr,
	sequence, headDim int,
	scale float64,
) error {
	if q == 0 || k == 0 || v == 0 || probability == 0 || dOut == 0 || dQ == 0 || dK == 0 || dV == 0 || dScores == 0 || sequence <= 0 || headDim <= 0 || math.IsNaN(scale) || math.IsInf(scale, 0) {
		return errors.New("resident ops: invalid attention VJP")
	}
	dProbability, err := o.AllocF32(sequence * sequence)
	if err != nil {
		return err
	}
	return o.AttentionCoreBackwardWithScratch(q, k, v, probability, dOut, dQ, dK, dV, dScores, dProbability, sequence, headDim, scale)
}

func (o *ResidentOps) AttentionCoreBackwardWithScratch(
	q, k, v, probability, dOut, dQ, dK, dV, dScores, dProbability driver.DevicePtr,
	sequence, headDim int,
	scale float64,
) error {
	if q == 0 || k == 0 || v == 0 || probability == 0 || dOut == 0 || dQ == 0 || dK == 0 || dV == 0 || dScores == 0 || dProbability == 0 || sequence <= 0 || headDim <= 0 || math.IsNaN(scale) || math.IsInf(scale, 0) {
		return errors.New("resident ops: invalid attention VJP scratch")
	}
	if err := o.session.gemm(false, true, sequence, headDim, sequence, dOut, v, dProbability); err != nil {
		return err
	}
	softmax, err := o.function("softmax_backward_f32")
	if err != nil {
		return err
	}
	rowsU, widthU := uint32(sequence), uint32(sequence)
	if err := o.session.launch1D(softmax, rowsU,
		unsafe.Pointer(&probability), unsafe.Pointer(&dProbability), unsafe.Pointer(&dScores),
		unsafe.Pointer(&rowsU), unsafe.Pointer(&widthU)); err != nil {
		return err
	}
	if err := o.session.gemm(true, false, sequence, sequence, headDim, probability, dOut, dV); err != nil {
		return err
	}
	if err := o.session.gemm(false, false, sequence, sequence, headDim, dScores, k, dQ); err != nil {
		return err
	}
	if err := o.session.gemm(true, false, sequence, sequence, headDim, dScores, q, dK); err != nil {
		return err
	}
	if scale == 1 {
		return nil
	}
	if err := o.Scale(dQ, dQ, float32(scale), sequence*headDim); err != nil {
		return err
	}
	return o.Scale(dK, dK, float32(scale), sequence*headDim)
}

func (o *ResidentOps) SoftmaxCrossEntropy(logits, targets, losses driver.DevicePtr, rows, classes int) error {
	if logits == 0 || targets == 0 || losses == 0 || rows <= 0 || classes <= 0 || uint64(rows) > math.MaxUint32 || uint64(classes) > math.MaxUint32 {
		return errors.New("resident ops: invalid cross entropy")
	}
	function, err := o.function("softmax_ce_grad_rows_f32")
	if err != nil {
		return err
	}
	rowsU, classesU, scale := uint32(rows), uint32(classes), float32(1)/float32(rows)
	return o.session.launch1D(function, rowsU,
		unsafe.Pointer(&logits), unsafe.Pointer(&targets), unsafe.Pointer(&losses),
		unsafe.Pointer(&rowsU), unsafe.Pointer(&classesU), unsafe.Pointer(&scale))
}

func (o *ResidentOps) Add(left, right, output driver.DevicePtr, count int) error {
	if left == 0 || right == 0 || output == 0 || count <= 0 || uint64(count) > math.MaxUint32 {
		return errors.New("resident ops: invalid add")
	}
	function, err := o.function("add_f32")
	if err != nil {
		return err
	}
	return o.session.launchVector3(function, left, right, output, count)
}

func (o *ResidentOps) Scale(input, output driver.DevicePtr, scale float32, count int) error {
	if input == 0 || output == 0 || count <= 0 || uint64(count) > math.MaxUint32 {
		return errors.New("resident ops: invalid scale")
	}
	function, err := o.function("scale_f32")
	if err != nil {
		return err
	}
	countU := uint32(count)
	return o.session.launch1D(function, countU,
		unsafe.Pointer(&input), unsafe.Pointer(&output), unsafe.Pointer(&scale), unsafe.Pointer(&countU))
}

func (o *ResidentOps) ScaleByScalar(input, scale, output driver.DevicePtr, count int) error {
	if input == 0 || scale == 0 || output == 0 || count <= 0 {
		return errors.New("resident ops: invalid device scale")
	}
	return o.session.gemm(false, false, count, 1, 1, input, scale, output)
}

func (o *ResidentOps) ReLUBackward(incoming, input, gradient driver.DevicePtr, count int) error {
	if incoming == 0 || input == 0 || gradient == 0 || count <= 0 || uint64(count) > math.MaxUint32 {
		return errors.New("resident ops: invalid ReLU VJP")
	}
	function, err := o.function("relu_backward_f32")
	if err != nil {
		return err
	}
	countU := uint32(count)
	return o.session.launch1D(function, countU,
		unsafe.Pointer(&incoming), unsafe.Pointer(&input), unsafe.Pointer(&gradient), unsafe.Pointer(&countU))
}

func (o *ResidentOps) SiLUBackward(incoming, input, gradient driver.DevicePtr, count int) error {
	if incoming == 0 || input == 0 || gradient == 0 || count <= 0 || uint64(count) > math.MaxUint32 {
		return errors.New("resident ops: invalid SiLU VJP")
	}
	function, err := o.function("silu_backward_f32")
	if err != nil {
		return err
	}
	return o.session.launchVector3(function, incoming, input, gradient, count)
}

func (o *ResidentOps) Conv2DBackward(
	input, weight, incoming, inputGradient, weightGradient, biasGradient driver.DevicePtr,
	batches, inputChannels, inputWidth, inputHeight, kernelWidth, kernelHeight,
	outputChannels, strideX, strideY, padLeft, padTop int,
) error {
	if input == 0 || weight == 0 || incoming == 0 || inputGradient == 0 || weightGradient == 0 || biasGradient == 0 ||
		batches <= 0 || inputChannels <= 0 || inputWidth <= 0 || inputHeight <= 0 || kernelWidth <= 0 || kernelHeight <= 0 ||
		outputChannels <= 0 || strideX <= 0 || strideY <= 0 || padLeft < 0 || padTop < 0 ||
		inputWidth+2*padLeft < kernelWidth || inputHeight+2*padTop < kernelHeight {
		return errors.New("resident ops: invalid Conv2D VJP")
	}
	outputWidth := (inputWidth+2*padLeft-kernelWidth)/strideX + 1
	outputHeight := (inputHeight+2*padTop-kernelHeight)/strideY + 1
	inputCount := batches * inputChannels * inputWidth * inputHeight
	weightCount := kernelWidth * kernelHeight * inputChannels * outputChannels
	if outputWidth <= 0 || outputHeight <= 0 || uint64(inputCount) > math.MaxUint32 || uint64(weightCount) > math.MaxUint32 || uint64(outputChannels) > math.MaxUint32 {
		return errors.New("resident ops: invalid Conv2D VJP geometry")
	}
	inputFunction, err := o.function("conv_2d_input_backward_f32")
	if err != nil {
		return err
	}
	weightFunction, err := o.function("conv_2d_weight_backward_f32")
	if err != nil {
		return err
	}
	biasFunction, err := o.function("conv_2d_bias_backward_f32")
	if err != nil {
		return err
	}
	batchesU, inputChannelsU := uint32(batches), uint32(inputChannels)
	inputWidthU, inputHeightU := uint32(inputWidth), uint32(inputHeight)
	kernelWidthU, kernelHeightU := uint32(kernelWidth), uint32(kernelHeight)
	outputChannelsU := uint32(outputChannels)
	outputWidthU, outputHeightU := uint32(outputWidth), uint32(outputHeight)
	strideXU, strideYU := uint32(strideX), uint32(strideY)
	padLeftU, padTopU := uint32(padLeft), uint32(padTop)
	inputCountU, weightCountU := uint32(inputCount), uint32(weightCount)
	if err := o.session.launch1D(inputFunction, inputCountU,
		unsafe.Pointer(&weight), unsafe.Pointer(&incoming), unsafe.Pointer(&inputGradient),
		unsafe.Pointer(&batchesU), unsafe.Pointer(&inputChannelsU), unsafe.Pointer(&inputWidthU), unsafe.Pointer(&inputHeightU),
		unsafe.Pointer(&kernelWidthU), unsafe.Pointer(&kernelHeightU), unsafe.Pointer(&outputChannelsU),
		unsafe.Pointer(&outputWidthU), unsafe.Pointer(&outputHeightU), unsafe.Pointer(&strideXU), unsafe.Pointer(&strideYU),
		unsafe.Pointer(&padLeftU), unsafe.Pointer(&padTopU), unsafe.Pointer(&inputCountU)); err != nil {
		return err
	}
	if err := o.session.launch1D(weightFunction, weightCountU,
		unsafe.Pointer(&input), unsafe.Pointer(&incoming), unsafe.Pointer(&weightGradient),
		unsafe.Pointer(&batchesU), unsafe.Pointer(&inputChannelsU), unsafe.Pointer(&inputWidthU), unsafe.Pointer(&inputHeightU),
		unsafe.Pointer(&kernelWidthU), unsafe.Pointer(&kernelHeightU), unsafe.Pointer(&outputChannelsU),
		unsafe.Pointer(&outputWidthU), unsafe.Pointer(&outputHeightU), unsafe.Pointer(&strideXU), unsafe.Pointer(&strideYU),
		unsafe.Pointer(&padLeftU), unsafe.Pointer(&padTopU), unsafe.Pointer(&weightCountU)); err != nil {
		return err
	}
	outputTokensU, biasCountU := uint32(outputWidth*outputHeight), outputChannelsU
	return o.session.launch1D(biasFunction, biasCountU,
		unsafe.Pointer(&incoming), unsafe.Pointer(&biasGradient), unsafe.Pointer(&batchesU),
		unsafe.Pointer(&outputChannelsU), unsafe.Pointer(&outputTokensU), unsafe.Pointer(&biasCountU))
}

func (o *ResidentOps) GroupNormBackward(
	input, weight, incoming, inputGradient, weightGradient, biasGradient driver.DevicePtr,
	batches, channels, tokens, groups int,
	epsilon float64,
) error {
	if input == 0 || weight == 0 || incoming == 0 || inputGradient == 0 || weightGradient == 0 || biasGradient == 0 ||
		batches <= 0 || channels <= 0 || tokens <= 0 || groups <= 0 || channels%groups != 0 ||
		epsilon <= 0 || math.IsNaN(epsilon) || math.IsInf(epsilon, 0) {
		return errors.New("resident ops: invalid GroupNorm VJP")
	}
	count := batches * channels * tokens
	if uint64(count) > math.MaxUint32 || uint64(channels) > math.MaxUint32 {
		return errors.New("resident ops: GroupNorm VJP geometry exceeds ABI")
	}
	inputFunction, err := o.function("group_norm_input_backward_f32")
	if err != nil {
		return err
	}
	parameterFunction, err := o.function("group_norm_parameter_backward_f32")
	if err != nil {
		return err
	}
	batchesU, channelsU, tokensU, groupsU := uint32(batches), uint32(channels), uint32(tokens), uint32(groups)
	countU := uint32(count)
	if err := o.session.launch1D(inputFunction, countU,
		unsafe.Pointer(&input), unsafe.Pointer(&weight), unsafe.Pointer(&incoming), unsafe.Pointer(&inputGradient),
		unsafe.Pointer(&batchesU), unsafe.Pointer(&channelsU), unsafe.Pointer(&tokensU), unsafe.Pointer(&groupsU),
		unsafe.Pointer(&epsilon), unsafe.Pointer(&countU)); err != nil {
		return err
	}
	parameterCountU := channelsU
	return o.session.launch1D(parameterFunction, parameterCountU,
		unsafe.Pointer(&input), unsafe.Pointer(&incoming), unsafe.Pointer(&weightGradient), unsafe.Pointer(&biasGradient),
		unsafe.Pointer(&batchesU), unsafe.Pointer(&channelsU), unsafe.Pointer(&tokensU), unsafe.Pointer(&groupsU),
		unsafe.Pointer(&epsilon), unsafe.Pointer(&parameterCountU))
}

func (o *ResidentOps) StridedRowCopy(
	source, destination driver.DevicePtr,
	rows, width, sourceStride, sourceOffset, destinationStride, destinationOffset int,
) error {
	if source == 0 || destination == 0 || rows <= 0 || width <= 0 || sourceStride < sourceOffset+width || destinationStride < destinationOffset+width {
		return errors.New("resident ops: invalid strided row copy")
	}
	function, err := o.function("strided_row_copy_f32")
	if err != nil {
		return err
	}
	rowsU, widthU := uint32(rows), uint32(width)
	sourceStrideU, sourceOffsetU := uint32(sourceStride), uint32(sourceOffset)
	destinationStrideU, destinationOffsetU := uint32(destinationStride), uint32(destinationOffset)
	return o.session.launch1D(function, rowsU*widthU,
		unsafe.Pointer(&source), unsafe.Pointer(&destination), unsafe.Pointer(&rowsU), unsafe.Pointer(&widthU),
		unsafe.Pointer(&sourceStrideU), unsafe.Pointer(&sourceOffsetU),
		unsafe.Pointer(&destinationStrideU), unsafe.Pointer(&destinationOffsetU))
}

func (o *ResidentOps) IndexedRowScatterAdd(source, rows, destination driver.DevicePtr, count, width int) error {
	if source == 0 || rows == 0 || destination == 0 || count <= 0 || width <= 0 {
		return errors.New("resident ops: invalid indexed scatter")
	}
	function, err := o.function("indexed_row_scatter_add_f32")
	if err != nil {
		return err
	}
	countU, widthU := uint32(count), uint32(width)
	return o.session.launch1D(function, countU*widthU,
		unsafe.Pointer(&source), unsafe.Pointer(&rows), unsafe.Pointer(&destination),
		unsafe.Pointer(&countU), unsafe.Pointer(&widthU))
}

func (o *ResidentOps) AttentionScoreAffineBackward(
	dScores, query, key, dLagBias, dScale driver.DevicePtr,
	sequence, headDim, lagCount int,
) error {
	if dScores == 0 || query == 0 || key == 0 || dLagBias == 0 || dScale == 0 || sequence <= 0 || headDim <= 0 || lagCount <= 0 {
		return errors.New("resident ops: invalid attention affine VJP")
	}
	function, err := o.function("attention_score_affine_backward_f32")
	if err != nil {
		return err
	}
	sequenceU, headDimU, lagCountU := uint32(sequence), uint32(headDim), uint32(lagCount)
	return o.session.launch1D(function, sequenceU*sequenceU,
		unsafe.Pointer(&dScores), unsafe.Pointer(&query), unsafe.Pointer(&key),
		unsafe.Pointer(&dLagBias), unsafe.Pointer(&dScale),
		unsafe.Pointer(&sequenceU), unsafe.Pointer(&headDimU), unsafe.Pointer(&lagCountU))
}
