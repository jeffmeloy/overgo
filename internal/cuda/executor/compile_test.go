package executor

import (
	"math"
	"strings"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func TestCompiledRetainedMemory(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(1024))
	producer := builder.Add(input, input)
	view := builder.FlatSlice(producer, 24, 1000)
	compiled, err := Compile(view)
	if err != nil {
		t.Fatal(err)
	}
	inputs := compiled.NewDeviceInputs()
	check := func(slots []OutputSlot, capacity []uint64, arena, pooled, owned uint64) {
		t.Helper()
		a, p, c, err := compiled.retainedAllocations(inputs, slots, capacity)
		if err != nil || a != arena || p != pooled || c != owned {
			t.Fatalf("allocation bytes = %d/%d/%d, want %d/%d/%d: %v", a, p, c, arena, pooled, owned, err)
		}
	}
	// The 1000-value view retains its 1024-value producer exactly once.
	check(nil, nil, 0, 8192, 0)
	if err := inputs.Set(input, driver.DevicePtr(deviceAllocationAlignment)); err != nil {
		t.Fatal(err)
	}
	check(nil, nil, 0, 4096, 0)
	slot, _ := compiled.OutputSlot(view)
	// An external view destination restores the producer's arena allocation.
	check([]OutputSlot{slot}, []uint64{96}, 4096, 0, 256)
	if _, _, _, err := compiled.retainedAllocations(inputs, []OutputSlot{1}, nil); err == nil {
		t.Fatal("invalid output slot accepted")
	}
	if _, _, _, err := compiled.retainedAllocations(inputs, nil, []uint64{math.MaxUint64}); err == nil {
		t.Fatal("overflowing capacity accepted")
	}
	if _, _, _, err := compiled.retainedAllocations(inputs, nil, []uint64{1 << 63, 1 << 63}); err == nil {
		t.Fatal("overflowing allocation sum accepted")
	}
}

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

// TestCompileHoldsLaunchWords holds a compiled graph's launches to the 32-bit words their kernels take, with shapes alone and no device.
//   - refused = a dimension or a counted element total past a kernel word, and a dimension past a signed word on an operation cuBLAS may take, each naming its operation
//   - compiled = a dimension past a signed word on an operation cuBLAS never takes, a mul_mat output past a kernel word, which its launch never counts, and a view past it that is no graph output and so never launches
func TestCompileHoldsLaunchWords(t *testing.T) {
	t.Parallel()
	const (
		wordPast    = uint64(1) << 32
		signedPast  = uint64(1) << 31
		wideRows    = uint64(1) << 17
		wideColumns = uint64(1) << 16
		inner       = 2
	)
	wide := tensor.MustShape(wideRows, wideColumns)
	// Each case's refusal is the operation the error opens on and what it says of the shape, and none means the graph compiles.
	for _, test := range []struct {
		name    string
		output  func(*tensor.Builder) *tensor.Tensor
		refusal []string
	}{
		{"a dimension past a word", func(b *tensor.Builder) *tensor.Tensor {
			input := b.Input("input", dtype.F32, tensor.MustShape(wordPast))
			return b.Add(input, input)
		}, []string{"CUDA add (tensor ", "output dimension 0 is 4294967296, past the 4294967295"}},
		{"a count past a word", func(b *tensor.Builder) *tensor.Tensor {
			return b.Scale(b.Input("input", dtype.F32, wide), 1)
		}, []string{"CUDA scale (tensor ", "counts 8589934592 elements, past the 4294967295"}},
		{"an output view past a word", func(b *tensor.Builder) *tensor.Tensor {
			return b.Reshape(b.Input("input", dtype.F32, wide), wideColumns, wideRows)
		}, []string{"CUDA reshape (tensor ", "counts 8589934592 elements, past the 4294967295"}},
		{"a cuBLAS dimension past a signed word", func(b *tensor.Builder) *tensor.Tensor {
			shape := tensor.MustShape(signedPast, 1)
			return b.MulMat(b.Input("weight", dtype.F32, shape), b.Input("input", dtype.F32, shape))
		}, []string{"CUDA mul_mat (tensor ", "input 0 dimension 0 is 2147483648, past the 2147483647"}},
		{"an elementwise dimension past a signed word", func(b *tensor.Builder) *tensor.Tensor {
			input := b.Input("input", dtype.F32, tensor.MustShape(signedPast))
			return b.Add(input, input)
		}, nil},
		{"a mul_mat output past a word", func(b *tensor.Builder) *tensor.Tensor {
			weight := b.Input("weight", dtype.F32, tensor.MustShape(inner, wideRows))
			return b.MulMat(weight, b.Input("input", dtype.F32, tensor.MustShape(inner, wideColumns)))
		}, nil},
		{"a view past a word that is no output", func(b *tensor.Builder) *tensor.Tensor {
			view := b.Reshape(b.Input("input", dtype.F32, wide), wideColumns, wideRows)
			return b.MulMat(b.Input("weight", dtype.F32, tensor.MustShape(wideColumns, 1)), view)
		}, nil},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			builder := tensor.NewBuilder()
			output := test.output(builder)
			if err := builder.Err(); err != nil {
				t.Fatal(err)
			}
			_, err := Compile(output)
			if test.refusal == nil {
				if err != nil {
					t.Fatalf("compile refused %s: %v", output.Op, err)
				}
				return
			}
			if err == nil || !strings.HasPrefix(err.Error(), test.refusal[0]) || !strings.Contains(err.Error(), test.refusal[1]) {
				t.Fatalf("compile of %s = %v, want a refusal opening %q and holding %q", output.Op, err, test.refusal[0], test.refusal[1])
			}
		})
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

// TestRetainedSmallViewCopiesOutOfItsSource holds a retained view smaller
// than half its source to storage of its own: the 24 values at the tail of
// a 4 MiB producer pool only their own buffer, and the producer joins
// the arena with the run's other intermediates, as a long prompt's recurrent
// state at the tail of a layer's packed output does.
func TestRetainedSmallViewCopiesOutOfItsSource(t *testing.T) {
	const (
		producerValues, viewValues = 1 << 20, 24
		producerBytes              = producerValues * 4
		viewBucket                 = 256
	)
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(producerValues))
	producer := builder.Add(input, input)
	view := builder.FlatSlice(producer, producerValues-viewValues, viewValues)
	compiled, err := Compile(view)
	if err != nil {
		t.Fatal(err)
	}
	if compiled.outputAliases[0] {
		t.Fatal("small retained view still aliases its source")
	}
	inputs := compiled.NewDeviceInputs()
	check := func(arena, pooled uint64) {
		t.Helper()
		a, p, c, err := compiled.retainedAllocations(inputs, nil, nil)
		if err != nil || a != arena || p != pooled || c != 0 {
			t.Fatalf("allocation bytes = %d/%d/%d, want %d/%d/0: %v", a, p, c, arena, pooled, err)
		}
	}
	check(producerBytes, viewBucket+producerBytes)
	if err := inputs.Set(input, driver.DevicePtr(deviceAllocationAlignment)); err != nil {
		t.Fatal(err)
	}
	check(producerBytes, viewBucket)
	// Layers whose small states are copied out reuse one another's storage:
	// three chained producers with a retained tail each hold at most two at
	// once, where holding each source to the end held all three.
	const layers = 3
	value := producer
	var views []*tensor.Tensor
	for range layers {
		views = append(views, builder.FlatSlice(value, producerValues-viewValues, viewValues))
		value = builder.Scale(value, 2)
	}
	chained, err := Compile(views...)
	if err != nil {
		t.Fatal(err)
	}
	if arena := chained.retainedMemory.ArenaSize; arena != 2*producerBytes {
		t.Fatalf("chained retained arena = %d, want %d", arena, 2*producerBytes)
	}
	// A view holding at least half its source keeps aliasing it.
	half := builder.FlatSlice(producer, producerValues/2, producerValues/2)
	if compiled, err = Compile(half); err != nil || !compiled.outputAliases[0] {
		t.Fatalf("half view does not alias its source: %v", err)
	}
}

func TestRetainedAliasPlanOmitsProducerStorage(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(1024))
	producer := builder.Scale(input, 2)
	view := builder.FlatSlice(producer, 24, 1000)
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
	if err := targets.Set(view, DeviceValue{Pointer: driver.DevicePtr(deviceAllocationAlignment), Shape: view.Shape, CapacityBytes: 1000 * 4}); err != nil {
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
	view := builder.FlatSlice(producer, 24, 1000)
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

func TestCompiledMultiRoPERetainsPolicy(t *testing.T) {
	for _, layout := range []tensor.RoPELayout{tensor.RoPELayoutNormal, tensor.RoPELayoutNeoX} {
		builder := tensor.NewBuilder()
		input := builder.Input("input", dtype.F32, tensor.MustShape(8, 1, 1))
		positions := [tensor.MaxDimensions][]uint32{{3}, {5}, {7}, {0}}
		output := builder.RoPEWithOptions(input, tensor.RoPEOptions{
			Layout: layout, MultiPositions: &positions, Sections: [tensor.MaxDimensions]int32{2, 1, 1},
			RotaryDimensions: 8, FrequencyBase: 10000, FrequencyScale: tensor.UnitFrequencyScale,
			InterleavedSections: true,
		})
		if err := builder.Err(); err != nil {
			t.Fatal(err)
		}
		compiled, err := Compile(output)
		if err != nil {
			t.Fatal(err)
		}
		attributes := compiled.NewRuntimeAttributes()
		next := output.Attrs.(tensor.RoPEMultiAttributes)
		next.Positions[0] = []uint32{4}
		if err := attributes.Set(output, next); err != nil {
			t.Fatal("coordinate update refused:", err)
		}
		for _, field := range []string{"layout", "interleaving", "shape"} {
			changed := next
			switch field {
			case "layout":
				if layout == tensor.RoPELayoutNormal {
					changed.Layout = tensor.RoPELayoutNeoX
				} else {
					changed.Layout = tensor.RoPELayoutNormal
				}
			case "interleaving":
				changed.InterleavedSections = !changed.InterleavedSections
			case "shape":
				changed.Positions[0] = append(changed.Positions[0], 5)
			}
			if err := attributes.Set(output, changed); err == nil {
				t.Fatalf("layout %d: compiled %s change accepted", layout, field)
			}
		}
		retained := attributes.values[compiled.orderIndexes[output]].(tensor.RoPEMultiAttributes)
		if retained.Layout != next.Layout || retained.InterleavedSections != next.InterleavedSections ||
			len(retained.Positions[0]) != len(next.Positions[0]) || retained.Positions[0][0] != next.Positions[0][0] {
			t.Fatal("rejected override changed retained runtime attributes")
		}
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

// streamMix builds a gated hyper-connection mix's reduction over streams
// slices of width values of x: Scale(((p0 + p1) + p2) + p3, 1/streams), each
// p_s a reshaped GroupSlice. It returns the scaled sum and the first add.
func streamMix(builder *tensor.Builder, x *tensor.Tensor, width, streams uint64) (*tensor.Tensor, *tensor.Tensor) {
	tokens := x.Shape.Dims[1]
	var sum, first *tensor.Tensor
	for stream := range streams {
		part := builder.Reshape(builder.GroupSlice(x, stream*width, width, 1, width), width, tokens)
		if sum == nil {
			sum = part
			continue
		}
		sum = builder.Add(sum, part)
		if first == nil {
			first = sum
		}
	}
	return builder.Scale(sum, 1/float32(streams)), first
}

// TestCompileFusesStreamSum holds a mix's reduction to one stream-sum
// fusion: of a gated x, the gate folded in and x skipped; of a plain x, x
// read; and none where an add is itself an output.
func TestCompileFusesStreamSum(t *testing.T) {
	const width, streams, tokens = 8, 4, 2
	builder := tensor.NewBuilder()
	gate := builder.Input("gate", dtype.F32, tensor.MustShape(width*streams, tokens))
	up := builder.Input("up", dtype.F32, gate.Shape)
	gated := builder.Multiply(builder.Sigmoid(gate), up)
	output, first := streamMix(builder, gated, width, streams)
	compiled, err := Compile(output)
	if err != nil {
		t.Fatal(err)
	}
	descriptor := compiled.nodes[compiled.orderIndexes[output]].fusion
	if descriptor == nil || descriptor.kind != compiledFusionStreamSum {
		t.Fatalf("stream-sum fusion = %+v", descriptor)
	}
	fusion := descriptor.streamSum
	if fusion.gate != gate || fusion.values != up || fusion.kind != activatedGateSigmoid ||
		fusion.width != width || fusion.streams != streams || fusion.scale != 1/float32(streams) {
		t.Fatalf("stream-sum fusion = %+v", fusion)
	}
	for _, skipped := range []*tensor.Tensor{gated, first} {
		if !compiled.nodes[compiled.orderIndexes[skipped]].skipped {
			t.Fatalf("fused tensor %d was not skipped", skipped.ID)
		}
	}
	plain := builder.Input("plain", dtype.F32, gate.Shape)
	plainOutput, _ := streamMix(builder, plain, width, streams)
	if compiled, err = Compile(plainOutput); err != nil {
		t.Fatal(err)
	}
	if descriptor := compiled.nodes[compiled.orderIndexes[plainOutput]].fusion; descriptor == nil ||
		descriptor.kind != compiledFusionStreamSum || descriptor.streamSum.gate != nil || descriptor.streamSum.values != plain {
		t.Fatalf("plain stream-sum fusion = %+v", descriptor)
	}
	if compiled, err = Compile(output, first); err != nil {
		t.Fatal(err)
	}
	if descriptor := compiled.nodes[compiled.orderIndexes[output]].fusion; descriptor != nil {
		t.Fatalf("a reduction whose add is an output fused: %+v", descriptor)
	}
}

// streamCombine builds a hyper-connection combine: residual
// [width, streams, tokens] plus current [width, tokens] repeated over the
// streams times the gate Scale(Sigmoid(Scale(x, 1/streams)), 2) of x
// [streams, tokens]. It returns the combine, its product and the gate.
func streamCombine(builder *tensor.Builder, residual, current, x *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
	width, streams, tokens := residual.Shape.Dims[0], residual.Shape.Dims[1], residual.Shape.Dims[2]
	gate := builder.Scale(builder.Sigmoid(builder.Scale(x, 1/float32(streams))), 2)
	repeated := builder.RepeatHeads(builder.Reshape(current, width, 1, tokens), uint32(streams))
	product := builder.Multiply(repeated, builder.Reshape(gate, 1, streams, tokens))
	return builder.Add(residual, product), product, gate
}

// TestCompileFusesStreamCombine holds a combine to one stream-combine
// fusion with its scaled-sigmoid gate folded in; a gate that is itself an
// output stays a tensor the fusion reads, and a product that is an output
// leaves the combine unfused.
func TestCompileFusesStreamCombine(t *testing.T) {
	const width, streams, tokens = 8, 4, 2
	builder := tensor.NewBuilder()
	residual := builder.Input("residual", dtype.F32, tensor.MustShape(width, streams, tokens))
	current := builder.Input("current", dtype.F32, tensor.MustShape(width, tokens))
	x := builder.Input("x", dtype.F32, tensor.MustShape(streams, tokens))
	output, product, gate := streamCombine(builder, residual, current, x)
	compiled, err := Compile(output)
	if err != nil {
		t.Fatal(err)
	}
	descriptor := compiled.nodes[compiled.orderIndexes[output]].fusion
	if descriptor == nil || descriptor.kind != compiledFusionStreamCombine {
		t.Fatalf("stream-combine fusion = %+v", descriptor)
	}
	fusion := descriptor.streamCombine
	if fusion.residual != residual || fusion.current != current || fusion.gate != x || !fusion.scaled ||
		fusion.inner != 1/float32(streams) || fusion.outer != 2 {
		t.Fatalf("stream-combine fusion = %+v", fusion)
	}
	for _, skipped := range []*tensor.Tensor{product, gate} {
		if !compiled.nodes[compiled.orderIndexes[skipped]].skipped {
			t.Fatalf("fused tensor %d was not skipped", skipped.ID)
		}
	}
	if compiled, err = Compile(output, gate); err != nil {
		t.Fatal(err)
	}
	if descriptor := compiled.nodes[compiled.orderIndexes[output]].fusion; descriptor == nil || descriptor.streamCombine.scaled {
		t.Fatalf("a combine whose gate is an output = %+v", descriptor)
	}
	if compiled, err = Compile(output, product); err != nil {
		t.Fatal(err)
	}
	if descriptor := compiled.nodes[compiled.orderIndexes[output]].fusion; descriptor != nil {
		t.Fatalf("a combine whose product is an output fused: %+v", descriptor)
	}
}

// hyperNorms builds a gated hyper-connection mix's normalization of input
// [width, streams, tokens] by a [width, streams] weight, its low-rank down
// product and gate SiLU(Scale(down·normalized, 1/streams)), and the up
// product reading the gate. It returns the normalization, the gate and the
// two products.
func hyperNorms(builder *tensor.Builder, input, weight, down, up *tensor.Tensor) (normalized, gate, low, high *tensor.Tensor) {
	width, streams, tokens := input.Shape.Dims[0], input.Shape.Dims[1], input.Shape.Dims[2]
	normalized = builder.Multiply(builder.RMSNorm(input, 1e-6), weight)
	low = builder.MulMat(down, builder.Reshape(normalized, width*streams, tokens))
	gate = builder.SiLU(builder.Scale(low, 1/float32(streams)))
	return normalized, gate, low, builder.MulMat(up, gate)
}

// TestCompileFusesHyperNorms holds a hyper-connection mix's normalization to
// a weighted-RMS fusion of one weight row a stream, writing the Q8_0 blocks
// its product reads through a reshape, and its gate to a scaled-activation
// fusion writing the up product's blocks.
func TestCompileFusesHyperNorms(t *testing.T) {
	const width, streams, rank = 64, 4, 32
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(width, streams, 1))
	weight := builder.Input("weight", dtype.F32, tensor.MustShape(width, streams))
	down := builder.Input("down", dtype.Q8_0, tensor.MustShape(width*streams, rank))
	up := builder.Input("up", dtype.Q8_0, tensor.MustShape(rank, width))
	normalized, gate, low, output := hyperNorms(builder, input, weight, down, up)
	compiled, err := Compile(output)
	if err != nil {
		t.Fatal(err)
	}
	descriptor := compiled.nodes[compiled.orderIndexes[normalized]].fusion
	if descriptor == nil || descriptor.kind != compiledFusionWeightedRMS || descriptor.weightedRMS.weightRows != streams || !descriptor.emitQ8 {
		t.Fatalf("hyper-connection normalization fusion = %+v", descriptor)
	}
	descriptor = compiled.nodes[compiled.orderIndexes[gate]].fusion
	if descriptor == nil || descriptor.kind != compiledFusionScaledActivation || !descriptor.emitQ8 ||
		descriptor.scaledActivation.input != low || descriptor.scaledActivation.kind != activatedGateSiLU ||
		descriptor.scaledActivation.scale != 1/float32(streams) {
		t.Fatalf("scaled activation fusion = %+v", descriptor)
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
