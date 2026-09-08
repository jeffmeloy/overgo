package executor

import (
	"strings"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func TestIndexedGraphOwnsCompilation(t *testing.T) {
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(4, 2)
	left := builder.Input("left", dtype.F32, shape)
	right := builder.Input("right", dtype.F32, shape)
	output := builder.Add(left, right)
	indexed, err := CompileIndexed(output)
	if err != nil {
		t.Fatal(err)
	}
	compiled := indexed.Graph
	if len(compiled.order) != 3 || len(compiled.outputs) != 1 || compiled.outputs[0] != output {
		t.Fatalf("compiled graph = %+v", compiled)
	}
	if compiled.memory.ArenaSize == 0 || compiled.needBlas {
		t.Fatalf("compiled memory/BLAS = %d/%t", compiled.memory.ArenaSize, compiled.needBlas)
	}
	if compiled.nodes[2].launcher == nil {
		t.Fatal("compiled add launcher is nil")
	}
	inputNodes := [...]*tensor.Tensor{left, right}
	if len(indexed.Inputs.Pointers) != len(inputNodes) {
		t.Fatalf("input pointers = %d, want %d", len(indexed.Inputs.Pointers), len(inputNodes))
	}
	for _, input := range inputNodes {
		if _, ok := compiled.InputSlot(input); !ok {
			t.Fatalf("input %q has no slot", input.Name)
		}
	}
	if _, ok := compiled.InputSlot(output); ok {
		t.Fatal("operator output received an input slot")
	}
}

func TestCUDAOperationProgramsCoverRegistry(t *testing.T) {
	for operation := tensor.OpInput + 1; ; operation++ {
		descriptor, ok := tensor.DescribeOperation(operation)
		if !ok {
			break
		}
		if descriptor.Backends&tensor.BackendCUDA == 0 {
			continue
		}
		if descriptor.CUDA == tensor.CUDAProgramNone {
			t.Fatalf("CUDA operation %s has no compiled program", operation)
		}
	}
}

func TestCompiledRetainedTargetsUseOutputSlots(t *testing.T) {
	const (
		fixtureWidth   = 4
		fixturePointer = driver.DevicePtr(deviceAllocationAlignment)
	)
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(fixtureWidth))
	first := builder.Scale(input, 2)
	second := builder.Scale(first, 3)
	compiled, err := Compile(first, second)
	if err != nil {
		t.Fatal(err)
	}
	fixtureBytes, err := second.Shape.Bytes(dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	targets := compiled.NewRetainedTargets()
	value := DeviceValue{
		Pointer:       fixturePointer,
		Shape:         second.Shape,
		CapacityBytes: fixtureBytes,
	}
	if err := targets.Set(second, value); err != nil {
		t.Fatal(err)
	}
	if targets.values[0].Pointer != 0 || targets.values[1].Pointer != value.Pointer ||
		!targets.values[1].Shape.Equal(value.Shape) ||
		targets.values[1].CapacityBytes != value.CapacityBytes {
		t.Fatalf("indexed targets = %+v", targets.values)
	}
	otherBuilder := tensor.NewBuilder()
	other := otherBuilder.Input("other", dtype.F32, tensor.MustShape(fixtureWidth))
	if err := targets.Set(other, value); err == nil {
		t.Fatal("non-output target accepted")
	}
	if _, err := Compile(first, first); err == nil {
		t.Fatal("duplicate output compiled")
	}
}

func TestNativeWeightStagingIsBoundedAndRowAligned(t *testing.T) {
	if got := nativeWeightStagingBytes(1024, 2); got != 8192 {
		t.Fatalf("small staging = %d, want 8192", got)
	}
	if got := nativeWeightStagingBytes(4096, 65536); got != nativeWeightStagingLimitBytes {
		t.Fatalf("bounded staging = %d, want %d", got, nativeWeightStagingLimitBytes)
	}
	widerThanLimit := nativeWeightStagingLimitBytes/4 + 1
	if got := nativeWeightStagingBytes(widerThanLimit, 2); got != widerThanLimit*4 {
		t.Fatalf("wide-row staging = %d, want %d", got, widerThanLimit*4)
	}
}

func TestCompileExternalOmitsCallerOwnedOutput(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(1024))
	output := builder.Scale(input, 2)
	compiled, err := CompileExternal(output)
	if err != nil {
		t.Fatal(err)
	}
	if !compiled.externalOutputs || compiled.memory.ArenaSize != 0 {
		t.Fatalf("external output plan external=%t arena=%d", compiled.externalOutputs, compiled.memory.ArenaSize)
	}
}

func TestRetainedPlanOmitsOutputStorage(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(1024))
	first := builder.Scale(input, 2)
	second := builder.Add(first, input)
	compiled, err := Compile(first, second)
	if err != nil {
		t.Fatal(err)
	}
	if compiled.memory.ArenaSize == 0 {
		t.Fatal("host-output execution lost its arena")
	}
	if compiled.retainedMemory.ArenaSize != 0 {
		t.Fatalf("retained outputs still reserve %d duplicate arena bytes", compiled.retainedMemory.ArenaSize)
	}
}

func TestRetainedPlanDoesNotIncreaseArena(t *testing.T) {
	builder := tensor.NewBuilder()
	width := uint64(deviceAllocationAlignment / 4)
	value := builder.Input("input", dtype.F32, tensor.MustShape(width, 1))
	var outputs []*tensor.Tensor
	// Excluding the third output changes best-fit placement of later tensors.
	for index, units := range []uint64{15, 11, 10, 10, 10, 15, 11, 4} {
		next := units * deviceAllocationAlignment / 4
		weights := builder.Input("weights", dtype.F32, tensor.MustShape(width, next))
		value = builder.MulMat(weights, value)
		width = next
		if index == 2 || index == 6 || index == 7 {
			outputs = append(outputs, value)
		}
	}
	compiled, err := Compile(outputs...)
	if err != nil {
		t.Fatal(err)
	}
	if compiled.retainedMemory.ArenaSize > compiled.memory.ArenaSize {
		t.Fatalf("retained arena = %d, managed arena = %d", compiled.retainedMemory.ArenaSize, compiled.memory.ArenaSize)
	}
	for _, output := range outputs {
		if _, exists := compiled.retainedMemory.Allocations[output]; exists {
			t.Fatal("retained output still owns an arena allocation")
		}
	}
}

func TestRetainedAliasPlanOmitsProducerStorage(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(1024))
	producer := builder.Scale(input, 2)
	view := builder.FlatSlice(producer, 1000, 24)
	compiled, err := Compile(view)
	if err != nil {
		t.Fatal(err)
	}
	if arena := compiled.executionMemory(true, nil).ArenaSize; arena != 0 {
		t.Fatalf("retained alias producer still reserves %d duplicate arena bytes", arena)
	}
	if compiled.executionMemory(false, nil).ArenaSize == 0 {
		t.Fatal("host output lost its producer storage")
	}
	targets := compiled.NewRetainedTargets()
	if err := targets.Set(view, DeviceValue{Pointer: driver.DevicePtr(deviceAllocationAlignment), Shape: view.Shape, CapacityBytes: 24 * 4}); err != nil {
		t.Fatal(err)
	}
	if compiled.executionMemory(true, targets).ArenaSize == 0 {
		t.Fatal("caller-bound alias lost its intermediate producer storage")
	}
}

func TestRetainedAliasPlanPreservesBoundWholeViews(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(1024))
	producer := builder.Scale(input, 2)
	view := builder.FlatSlice(producer, 1000, 24)
	other := builder.Scale(input, 3)
	whole := builder.Reshape(other, 32, 32)
	compiled, err := Compile(view, whole)
	if err != nil {
		t.Fatal(err)
	}
	targets := compiled.NewRetainedTargets()
	if err := targets.Set(whole, DeviceValue{Pointer: driver.DevicePtr(deviceAllocationAlignment), Shape: whole.Shape, CapacityBytes: 1024 * 4}); err != nil {
		t.Fatal(err)
	}
	plan := compiled.executionMemory(true, targets)
	if _, duplicate := plan.Allocations[producer]; duplicate {
		t.Fatal("bound whole view kept an unrelated retained slice producer in the arena")
	}
	if _, required := plan.Allocations[other]; !required {
		t.Fatal("bound whole view lost its producer storage")
	}
}

func TestCompiledRuntimeAttributesUseNodeIndexes(t *testing.T) {
	builder := tensor.NewBuilder()
	builder.SetCacheAppendPlan(tensor.CacheAppendPlan{ActiveTokens: 2, CapacityTokens: 4})
	input := builder.Input("input", dtype.F32, tensor.MustShape(2, 1, 4))
	appendValue := builder.Input("append", dtype.F32, tensor.MustShape(2, 1, 1))
	output := builder.AppendCache(input, appendValue, 2)
	compiled, err := Compile(output)
	if err != nil {
		t.Fatal(err)
	}
	attributes := compiled.NewRuntimeAttributes()
	if err = attributes.Set(output, tensor.CacheAppendAttributes{Axis: 2, Offset: 3}); err != nil {
		t.Fatal(err)
	}
	if attributes.values[compiled.orderIndexes[output]].(tensor.CacheAppendAttributes).Offset != 3 {
		t.Fatalf("runtime attributes = %+v", attributes.values)
	}
	if err = attributes.Set(output, tensor.RMSNormAttributes{Epsilon: 1e-5}); err == nil {
		t.Fatal("mismatched runtime attributes accepted")
	}
	if err = attributes.Set(output, tensor.CacheAppendAttributes{Axis: 2, Offset: 4}); err == nil {
		t.Fatal("out-of-capacity runtime append accepted")
	}
	other := tensor.NewBuilder().Input("other", dtype.F32, tensor.MustShape(4))
	if err = attributes.Set(other, tensor.GetRowsAttributes{Rows: []uint32{1}}); err == nil {
		t.Fatal("foreign runtime node accepted")
	}
}

func TestGeneratedKernelArgumentCountValidation(t *testing.T) {
	kernel := boundKernel{
		id:            kernelAddF32,
		argumentCount: kernelFunctionArgumentCounts[kernelAddF32],
	}
	if err := validateKernelArgumentCount(kernel, int(kernel.argumentCount)); err != nil {
		t.Fatal(err)
	}
	if err := validateKernelArgumentCount(kernel, int(kernel.argumentCount)-1); err == nil ||
		!strings.Contains(err.Error(), kernelFunctionNames[kernelAddF32]) {
		t.Fatalf("argument mismatch error = %v", err)
	}
}

func TestCompileDetectsBLASAndRejectsNilGraph(t *testing.T) {
	builder := tensor.NewBuilder()
	left := builder.Input("left", dtype.F32, tensor.MustShape(2, 2))
	right := builder.Input("right", dtype.F32, tensor.MustShape(2, 2))
	compiled, err := Compile(builder.MulMat(left, right))
	if err != nil {
		t.Fatal(err)
	}
	if !compiled.needBlas {
		t.Fatal("F32 matrix graph did not request BLAS")
	}
	if _, err := Compile(nil); err == nil {
		t.Fatal("nil graph compiled")
	}
}

func TestDeviceBufferBucket(t *testing.T) {
	for _, fixture := range []struct {
		size uint64
		want uint64
	}{
		{1, deviceAllocationAlignment},
		{deviceAllocationAlignment, deviceAllocationAlignment},
		{deviceAllocationAlignment + 1, deviceAllocationAlignment * 2},
		{deviceAllocationAlignment * 4, deviceAllocationAlignment * 4},
	} {
		got, err := deviceBufferBucket(fixture.size)
		if err != nil || got != fixture.want {
			t.Fatalf("bucket(%d) = (%d, %v), want (%d, nil)", fixture.size, got, err, fixture.want)
		}
	}
	if _, err := deviceBufferBucket(0); err == nil {
		t.Fatal("expected zero-size buffer rejection")
	}
}

func TestRetainedOutputLayoutUsesOneAlignedSpan(t *testing.T) {
	const fixtureWidth = 5
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(fixtureWidth, 1))
	first := builder.SiLU(input)
	second := builder.Sigmoid(first)
	compiled, err := Compile(second)
	if err != nil {
		t.Fatal(err)
	}
	storage := make([]bool, len(compiled.order))
	offsets := make([]uint64, len(compiled.order))
	storage[compiled.orderIndexes[first]] = true
	storage[compiled.orderIndexes[second]] = true
	bytes, err := retainedOutputLayout(compiled, storage, offsets)
	if err != nil {
		t.Fatal(err)
	}
	firstOffset := offsets[compiled.orderIndexes[first]]
	secondOffset := offsets[compiled.orderIndexes[second]]
	if firstOffset%deviceAllocationAlignment != 0 || secondOffset%deviceAllocationAlignment != 0 {
		t.Fatalf("retained offsets are not aligned: %v", offsets)
	}
	if firstOffset == secondOffset || bytes <= secondOffset {
		t.Fatalf("retained layout overlaps or truncates: offsets %v bytes %d", offsets, bytes)
	}
}

func TestCompileFusesSingleUseWeightedRMSNorm(t *testing.T) {
	const fixtureWidth = 8
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(fixtureWidth, 2))
	weight := builder.Input("weight", dtype.F32, tensor.MustShape(fixtureWidth))
	normalized := builder.RMSNorm(input, 1e-5)
	output := builder.Multiply(normalized, weight)
	compiled, err := Compile(output)
	if err != nil {
		t.Fatal(err)
	}
	descriptor := compiled.nodes[compiled.orderIndexes[output]].fusion
	if descriptor == nil || descriptor.kind != compiledFusionWeightedRMS ||
		descriptor.weightedRMS.normalization != normalized || descriptor.weightedRMS.weight != weight {
		t.Fatalf("weighted RMSNorm fusion = %+v", descriptor)
	}
	if compiled.fusions != nil || descriptor.operands != nil || descriptor.operandCount == 0 {
		t.Fatal("fusion compile state remains resident")
	}
	if !compiled.nodes[compiled.orderIndexes[normalized]].skipped {
		t.Fatal("fused RMSNorm launch was not skipped")
	}
}

func TestCompileFusesResidualAddIntoWeightedRMSNorm(t *testing.T) {
	const fixtureWidth = 8
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(fixtureWidth, 2)
	left := builder.Input("left", dtype.F32, shape)
	right := builder.Input("right", dtype.F32, shape)
	weight := builder.Input("weight", dtype.F32, tensor.MustShape(fixtureWidth))
	residual := builder.Add(left, right)
	normalized := builder.RMSNorm(residual, 1e-5)
	output := builder.Multiply(normalized, weight)
	compiled, err := Compile(output)
	if err != nil {
		t.Fatal(err)
	}
	descriptor := compiled.nodes[compiled.orderIndexes[output]].fusion
	if descriptor == nil || descriptor.kind != compiledFusionWeightedRMS ||
		descriptor.weightedRMS.addLeft != left || descriptor.weightedRMS.addRight != right {
		t.Fatalf("weighted residual RMSNorm fusion = %+v", descriptor)
	}
	if !compiled.nodes[compiled.orderIndexes[residual]].skipped {
		t.Fatal("fused residual launch was not skipped")
	}
}

// A broadcast add (rank-1 bias) ahead of the norm must NOT fold into the
// fused kernel: weighted_rms_norm_add_f32 reads both operands at full row
// extent, so folding a bias reads past its storage.
func TestCompileKeepsBroadcastAddOutOfWeightedRMSNorm(t *testing.T) {
	const fixtureWidth = 8
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(fixtureWidth, 2))
	bias := builder.Input("bias", dtype.F32, tensor.MustShape(fixtureWidth))
	weight := builder.Input("weight", dtype.F32, tensor.MustShape(fixtureWidth))
	biased := builder.Add(input, bias)
	normalized := builder.RMSNorm(biased, 1e-5)
	output := builder.Multiply(normalized, weight)
	compiled, err := Compile(output)
	if err != nil {
		t.Fatal(err)
	}
	descriptor := compiled.nodes[compiled.orderIndexes[output]].fusion
	if descriptor == nil || descriptor.kind != compiledFusionWeightedRMS ||
		descriptor.weightedRMS.normalization != normalized {
		t.Fatalf("weighted RMSNorm fusion = %+v", descriptor)
	}
	fusion := descriptor.weightedRMS
	if fusion.addLeft != nil || fusion.addRight != nil {
		t.Fatal("broadcast bias add was folded into the fused RMSNorm")
	}
	if compiled.nodes[compiled.orderIndexes[biased]].skipped {
		t.Fatal("broadcast bias add launch was skipped")
	}
}

func TestCompileFusesWeightedRMSNormAndActivatedGate(t *testing.T) {
	const fixtureWidth = 8
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(fixtureWidth, 2)
	input := builder.Input("input", dtype.F32, shape)
	gate := builder.Input("gate", dtype.F32, shape)
	weight := builder.Input("weight", dtype.F32, tensor.MustShape(fixtureWidth))
	normalized := builder.RMSNorm(input, 1e-5)
	weighted := builder.Multiply(normalized, weight)
	activation := builder.SiLU(gate)
	output := builder.Multiply(weighted, activation)
	compiled, err := Compile(output)
	if err != nil {
		t.Fatal(err)
	}
	descriptor := compiled.nodes[compiled.orderIndexes[output]].fusion
	if descriptor == nil || descriptor.kind != compiledFusionWeightedRMSGate {
		t.Fatalf("weighted RMS gate fusion = %+v", descriptor)
	}
	fusion := descriptor.weightedGate
	if fusion.normalization != normalized || fusion.weight != weight ||
		fusion.gate != gate || fusion.kind != activatedGateSiLU {
		t.Fatalf("weighted RMS gate fusion = %+v", fusion)
	}
	for _, skipped := range []*tensor.Tensor{normalized, weighted, activation} {
		if !compiled.nodes[compiled.orderIndexes[skipped]].skipped {
			t.Fatalf("fused tensor %d was not skipped", skipped.ID)
		}
	}
	if compiled.nodes[compiled.orderIndexes[weighted]].fusion != nil {
		t.Fatal("subsumed weighted RMS fusion remains active")
	}
}
