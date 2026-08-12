//go:build windows

package executor

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"testing"

	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/model"
	"overgo/internal/quant"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestExecutorWavTokenizerDecoderMatchesReference(t *testing.T) {
	fixture := newCUDAReferenceFixture(t, 3)
	builder := fixture.builder
	input := func(shape tensor.Shape, scale, offset float32) *tensor.Tensor {
		return fixture.nextInput("wav", shape, scale, offset)
	}
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "wavtokenizer-dec", EmbeddingLength: 2, OutputEmbeddingLength: 3,

		FeedForwardLength: 4,
		LayerNormEpsilon:  1e-5}, MultimodalSpec: model.MultimodalSpec{PosNetEmbeddingLength: 2, PosNetBlockCount: 6,
		ConvNextEmbeddingLength: 2, ConvNextBlockCount: 1,
		GroupNormGroups: 1, GroupNormEpsilon: 1e-5},
	}
	weights := model.WavTokenizerGraphWeights{
		InputConv:      input(tensor.MustShape(7, 2, 2), 0.03, -0.1),
		InputConvBias:  input(tensor.MustShape(1, 2), 0.02, -0.03),
		PosNet:         make([]model.WavPosNetGraphWeights, 6),
		TokenNorm:      input(tensor.MustShape(2), 0.03, 0.9),
		TokenNormBias:  input(tensor.MustShape(2), 0.02, -0.03),
		ConvNext:       make([]model.WavConvNextGraphWeights, 1),
		OutputNorm:     input(tensor.MustShape(2), 0.03, 0.9),
		OutputNormBias: input(tensor.MustShape(2), 0.02, -0.03),
		Output:         input(tensor.MustShape(2, 3), 0.04, -0.1),
		OutputBias:     input(tensor.MustShape(3), 0.02, -0.03),
	}
	for _, block := range []int{0, 1, 3, 4} {
		weights.PosNet[block] = model.WavPosNetGraphWeights{
			Norm1: input(tensor.MustShape(1, 2), 0.03, 0.9), Norm1Bias: input(tensor.MustShape(1, 2), 0.02, -0.03),
			Conv1: input(tensor.MustShape(3, 2, 2), 0.03, -0.1), Conv1Bias: input(tensor.MustShape(1, 2), 0.02, -0.03),
			Norm2: input(tensor.MustShape(1, 2), 0.03, 0.9), Norm2Bias: input(tensor.MustShape(1, 2), 0.02, -0.03),
			Conv2: input(tensor.MustShape(3, 2, 2), 0.03, -0.1), Conv2Bias: input(tensor.MustShape(1, 2), 0.02, -0.03),
		}
	}
	weights.PosNet[2] = model.WavPosNetGraphWeights{
		AttentionNorm: input(tensor.MustShape(1, 2), 0.03, 0.9), AttentionNormBias: input(tensor.MustShape(1, 2), 0.02, -0.03),
		AttentionQ: input(tensor.MustShape(1, 2, 2), 0.03, -0.1), AttentionQBias: input(tensor.MustShape(1, 2), 0.02, -0.03),
		AttentionK: input(tensor.MustShape(1, 2, 2), 0.03, -0.1), AttentionKBias: input(tensor.MustShape(1, 2), 0.02, -0.03),
		AttentionV: input(tensor.MustShape(1, 2, 2), 0.03, -0.1), AttentionVBias: input(tensor.MustShape(1, 2), 0.02, -0.03),
		AttentionOutput: input(tensor.MustShape(1, 2, 2), 0.03, -0.1), AttentionOutBias: input(tensor.MustShape(1, 2), 0.02, -0.03),
	}
	weights.PosNet[5] = model.WavPosNetGraphWeights{
		AttentionNorm:     input(tensor.MustShape(1, 2), 0.03, 0.9),
		AttentionNormBias: input(tensor.MustShape(1, 2), 0.02, -0.03),
	}
	weights.ConvNext[0] = model.WavConvNextGraphWeights{
		Depthwise: input(tensor.MustShape(7, 1, 2), 0.03, -0.1), DepthwiseBias: input(tensor.MustShape(1, 2), 0.02, -0.03),
		Norm: input(tensor.MustShape(2), 0.03, 0.9), NormBias: input(tensor.MustShape(2), 0.02, -0.03),
		Pointwise1: input(tensor.MustShape(2, 4), 0.03, -0.1), Pointwise1Bias: input(tensor.MustShape(4), 0.02, -0.03),
		Pointwise2: input(tensor.MustShape(4, 2), 0.03, -0.1), Pointwise2Bias: input(tensor.MustShape(2), 0.02, -0.03),
		Gamma: input(tensor.MustShape(2), 0.03, 0.9),
	}
	embeddings := input(tensor.MustShape(2, 4), 0.1, -0.2)
	program := fixture.modelPlan(spec)
	output, err := program.BuildSequenceOutput(builder, embeddings, weights)
	if err != nil {
		t.Fatal(err)
	}
	fixture.requireMatch([]*tensor.Tensor{output}, 1e-3)
}

func TestExecutorDFlashPipelineMatchesReference(t *testing.T) {
	fixture := newCUDAReferenceFixture(t, 3)
	builder, input := fixture.builder, fixture.input
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "dflash", EmbeddingLength: 4, FeedForwardLength: 6,

		RMSNormEpsilon: 1e-5,
		TargetLayers:   []int32{1, 3}}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 2, ValueLength: 2,
		RopeDimensionCount: 2, RopeFrequencyBase: 10000,
		NonCausalAttention: true},
	}
	features := input("features", tensor.MustShape(8, 2), 0.08, -0.1)
	projection := input("fc", tensor.MustShape(8, 4), 0.03, -0.1)
	encoderNorm := input("enc_norm", tensor.MustShape(4), 0.03, 0.9)
	program := fixture.modelPlan(spec)
	fused, err := program.BuildFeatureProjection(builder, features, projection, encoderNorm)
	if err != nil {
		t.Fatal(err)
	}
	weights := model.LayerGraphWeights{
		AttentionNorm:   input("attn_norm", tensor.MustShape(4), 0.03, 0.9),
		AttentionQ:      input("q", tensor.MustShape(4, 4), 0.03, -0.1),
		AttentionK:      input("k", tensor.MustShape(4, 2), 0.03, -0.1),
		AttentionV:      input("v", tensor.MustShape(4, 2), 0.03, -0.1),
		AttentionOutput: input("o", tensor.MustShape(4, 4), 0.03, -0.1),
		AttentionQNorm:  input("q_norm", tensor.MustShape(2), 0.03, 0.9),
		AttentionKNorm:  input("k_norm", tensor.MustShape(2), 0.03, 0.9),
		FeedForwardNorm: input("ffn_norm", tensor.MustShape(4), 0.03, 0.9),
		FeedForwardGate: input("gate", tensor.MustShape(4, 6), 0.03, -0.1),
		FeedForwardUp:   input("up", tensor.MustShape(4, 6), 0.03, -0.1),
		FeedForwardDown: input("down", tensor.MustShape(6, 4), 0.03, -0.1),
	}
	key, value, err := program.BuildCacheProjection(builder, fused, weights, []uint32{0, 1}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	noise := input("noise", tensor.MustShape(4, 3), 0.08, -0.1)
	result, err := buildFixtureDenseBlockCachedForLayer(builder, noise, spec, weights, []uint32{2, 3, 4}, key, value, 0)
	if err != nil {
		t.Fatal(err)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	fixture.requireMatch(outputs, 1e-3)
}

func TestExecutorEagle3PipelineMatchesReference(t *testing.T) {
	fixture := newCUDAReferenceFixture(t, 3)
	builder, input := fixture.builder, fixture.input
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "eagle3", BlockCount: 1, EmbeddingLength: 4, TargetHiddenSize: 3,
		TargetLayers: []int32{1, 3, 5}, FeedForwardLength: 6,

		RMSNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 2, ValueLength: 2,
		RopeDimensionCount: 2, RopeFrequencyBase: 10000},
	}
	features := input("features", tensor.MustShape(9, 3), 0.08, -0.1)
	projection := input("fc", tensor.MustShape(9, 4), 0.03, -0.1)
	program := fixture.modelPlan(spec)
	fused, err := program.BuildFeatureProjection(builder, features, projection, nil)
	if err != nil {
		t.Fatal(err)
	}
	weights := model.LayerGraphWeights{
		AttentionNorm:   input("token_norm", tensor.MustShape(4), 0.03, 0.9),
		AttentionNorm2:  input("target_norm", tensor.MustShape(4), 0.03, 0.9),
		AttentionQ:      input("q", tensor.MustShape(8, 4), 0.03, -0.1),
		AttentionK:      input("k", tensor.MustShape(8, 2), 0.03, -0.1),
		AttentionV:      input("v", tensor.MustShape(8, 2), 0.03, -0.1),
		AttentionOutput: input("o", tensor.MustShape(4, 4), 0.03, -0.1),
		FeedForwardNorm: input("ffn_norm", tensor.MustShape(4), 0.03, 0.9),
		FeedForwardGate: input("gate", tensor.MustShape(4, 6), 0.03, -0.1),
		FeedForwardUp:   input("up", tensor.MustShape(4, 6), 0.03, -0.1),
		FeedForwardDown: input("down", tensor.MustShape(6, 4), 0.03, -0.1),
	}
	tokens := input("tokens", tensor.MustShape(4, 3), 0.08, -0.1)
	result := fixture.layer(fixture.modelPlan(spec), 0, spec, weights, model.CachedBlockContext{
		Input: tokens, Positions: []uint32{0, 1, 2}, PerLayerInput: fused,
		CacheWrite: tensor.CacheWriteConcat,
	})
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	fixture.requireMatch(outputs, 1e-3)
}

func TestExecutorGemma4AssistantPipelineMatchesReference(t *testing.T) {
	fixture := newCUDAReferenceFixture(t, 7)
	builder, input := fixture.builder, fixture.input
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "gemma4-assistant", BlockCount: 2, EmbeddingLength: 4,
		TargetHiddenSize: 6, FeedForwardLength: 6,

		RMSNormEpsilon: 1e-5,
		VocabularySize: 8}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, KeyLengthSWA: 2, ValueLengthSWA: 2,
		RopeDimensionCount: 4, RopeDimensionSWA: 2,
		RopeFrequencyBase: 10000, RopeFrequencySWA: 1000,
		SlidingWindow: 2, SlidingLayers: []bool{true, false}},
	}
	targetToken := input("target_token", tensor.MustShape(6, 1), 0.06, -0.1)
	targetHidden := input("target_hidden", tensor.MustShape(6, 1), 0.05, 0.2)
	pre := input("pre", tensor.MustShape(12, 4), 0.03, -0.1)
	program := fixture.modelPlan(spec)
	current, err := program.BuildFusedInput(builder, targetToken, targetHidden, pre)
	if err != nil {
		t.Fatal(err)
	}
	for layer := uint32(0); layer < spec.BlockCount; layer++ {
		keyWidth := uint64(spec.LayerKeyLength(layer))
		weights := model.LayerGraphWeights{
			AttentionNorm:       input(fmt.Sprintf("attn_norm_%d", layer), tensor.MustShape(4), 0.03, 0.9),
			AttentionQ:          input(fmt.Sprintf("q_%d", layer), tensor.MustShape(4, 2*keyWidth), 0.03, -0.1),
			AttentionQNorm:      input(fmt.Sprintf("q_norm_%d", layer), tensor.MustShape(keyWidth), 0.03, 0.9),
			AttentionOutput:     input(fmt.Sprintf("o_%d", layer), tensor.MustShape(2*keyWidth, 4), 0.03, -0.1),
			AttentionPostNorm:   input(fmt.Sprintf("attn_post_%d", layer), tensor.MustShape(4), 0.03, 0.9),
			FeedForwardNorm:     input(fmt.Sprintf("ffn_norm_%d", layer), tensor.MustShape(4), 0.03, 0.9),
			FeedForwardGate:     input(fmt.Sprintf("gate_%d", layer), tensor.MustShape(4, 6), 0.03, -0.1),
			FeedForwardUp:       input(fmt.Sprintf("up_%d", layer), tensor.MustShape(4, 6), 0.03, -0.1),
			FeedForwardDown:     input(fmt.Sprintf("down_%d", layer), tensor.MustShape(6, 4), 0.03, -0.1),
			FeedForwardPostNorm: input(fmt.Sprintf("ffn_post_%d", layer), tensor.MustShape(4), 0.03, 0.9),
			LayerOutputScale:    input(fmt.Sprintf("scale_%d", layer), tensor.MustShape(1), 0.01, 0.95),
		}
		sharedKey := input(fmt.Sprintf("shared_key_%d", layer), tensor.MustShape(keyWidth, 1, 3), 0.04, -0.1)
		sharedValue := input(fmt.Sprintf("shared_value_%d", layer), tensor.MustShape(keyWidth, 1, 3), 0.04, -0.1)
		block := fixture.layer(program, int(layer), spec, weights, model.CachedBlockContext{
			Input: current, Positions: []uint32{3}, PastKey: sharedKey, PastValue: sharedValue,
		})
		current = block.Output
	}
	outputNorm := input("output_norm", tensor.MustShape(4), 0.03, 0.9)
	output := input("output", tensor.MustShape(4, 8), 0.03, -0.1)
	post := input("post", tensor.MustShape(4, 6), 0.03, -0.1)
	logits, nextHidden, err := program.BuildProjectedOutputs(builder, current, outputNorm, output, post)
	if err != nil {
		t.Fatal(err)
	}
	outputs := []*tensor.Tensor{logits, nextHidden}
	fixture.requireMatch(outputs, 2e-3)
}

func TestExecutorGemma3nActiveStageMatchesReference(t *testing.T) {
	fixture := newCUDAReferenceFixture(t, 11)
	builder, input := fixture.builder, fixture.input
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "gemma3n", BlockCount: 21, EmbeddingLength: 4,
		FeedForwardLength: 6,

		RMSNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 2, ValueLength: 2, RopeDimensionCount: 2,
		RopeFrequencyBase: 10000}, MultimodalSpec: model.MultimodalSpec{KVFromStart: 20, SharedKVLayers: 1},
	}
	weights := model.LayerGraphWeights{
		AttentionNorm:       input("attn_norm", tensor.MustShape(4), 0.03, 0.9),
		AttentionQ:          input("q", tensor.MustShape(4, 4), 0.03, -0.1),
		AttentionK:          input("k", tensor.MustShape(4, 2), 0.03, -0.1),
		AttentionV:          input("v", tensor.MustShape(4, 2), 0.03, -0.1),
		AttentionOutput:     input("o", tensor.MustShape(4, 4), 0.03, -0.1),
		AttentionQNorm:      input("q_norm", tensor.MustShape(2), 0.03, 0.9),
		AttentionKNorm:      input("k_norm", tensor.MustShape(2), 0.03, 0.9),
		AttentionPostNorm:   input("attn_post", tensor.MustShape(4), 0.03, 0.9),
		FeedForwardNorm:     input("ffn_norm", tensor.MustShape(4), 0.03, 0.9),
		FeedForwardGate:     input("gate", tensor.MustShape(4, 6), 0.03, -0.1),
		FeedForwardUp:       input("up", tensor.MustShape(4, 6), 0.03, -0.1),
		FeedForwardDown:     input("down", tensor.MustShape(6, 4), 0.03, -0.1),
		FeedForwardPostNorm: input("ffn_post", tensor.MustShape(4), 0.03, 0.9),
		LaurelLeft:          input("laurel_l", tensor.MustShape(4, 2), 0.03, -0.1),
		LaurelRight:         input("laurel_r", tensor.MustShape(2, 4), 0.03, -0.1),
		LaurelPostNorm:      input("laurel_post", tensor.MustShape(4), 0.03, 0.9),
	}
	current := input("current", tensor.MustShape(4, 3), 0.08, -0.1)
	program, err := compileFixtureLayerProgram(spec, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	stage, err := program.BuildActivationProjection(model.CachedBlockContext{
		Builder: builder, Input: current, Positions: []uint32{0, 1, 2}, Layer: 0,
	}, weights)
	if err != nil {
		t.Fatal(err)
	}
	activated := builder.Multiply(builder.GELU(stage.Gate), stage.Up)
	output, err := program.BuildActivatedOutput(builder, stage.Residual, activated, weights)
	if err != nil {
		t.Fatal(err)
	}
	outputs := []*tensor.Tensor{output, stage.Key, stage.Value}
	fixture.requireMatch(outputs, 2e-3)
}

func TestExecutorUsesPersistentDeviceFeed(t *testing.T) {
	cudatest.Require(t)
	worker := newFixtureWorker(t)
	leftData := []float32{1, 2, 3, 4}
	pointer := copyFixtureDeviceBytes(t, worker, driver.Bytes(leftData))
	cuda := newFixtureExecutorWithWorker(t, worker)
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(4)
	left := builder.Input("left", dtype.F32, shape)
	right := builder.Input("right", dtype.F32, shape)
	output := builder.Add(left, right)
	rightValue, _ := reference.NewValue(shape, []float32{10, 20, 30, 40})
	got, err := cuda.ExecuteWithDeviceFeeds(
		context.Background(),
		[]*tensor.Tensor{output},
		map[*tensor.Tensor]reference.Value{right: rightValue},
		map[*tensor.Tensor]driver.DevicePtr{left: pointer},
	)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, []float32{11, 22, 33, 44}, 0)
}

func TestExecutorQ8DeviceEmbeddingAndMulMat(t *testing.T) {
	const (
		inputRows          = 5
		q8FixtureBlockSize = 32
	)
	cudatest.Require(t)
	worker := newFixtureWorker(t)
	storage := make([]byte, 68)
	binary.LittleEndian.PutUint16(storage[0:], 0x3800)  // 0.5
	binary.LittleEndian.PutUint16(storage[34:], 0x3800) // 0.5
	for index := 0; index < q8FixtureBlockSize; index++ {
		storage[2+index] = byte(int8(index - 16))
		storage[36+index] = 2
	}
	pointer := copyFixtureDeviceBytes(t, worker, storage)
	cuda := newFixtureExecutorWithWorker(t, worker)
	builder := tensor.NewBuilder()
	weights := builder.Input("weights", dtype.Q8_0, tensor.MustShape(q8FixtureBlockSize, 2))
	rows := builder.GetRows(weights, []uint32{1})
	indices := builder.Input("indices", dtype.F32, tensor.MustShape(1))
	dynamicRows := builder.GatherLast(weights, indices)
	input := builder.Input("input", dtype.F32, tensor.MustShape(q8FixtureBlockSize, inputRows))
	product := builder.MulMat(weights, input)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	inputData := make([]float32, q8FixtureBlockSize*inputRows)
	for row := range inputRows {
		for column := range q8FixtureBlockSize {
			inputData[row*q8FixtureBlockSize+column] = float32(row + 1)
		}
	}
	inputValue, _ := reference.NewValue(input.Shape, inputData)
	results, err := cuda.ExecuteWithDeviceFeeds(
		context.Background(),
		[]*tensor.Tensor{rows, dynamicRows, product},
		map[*tensor.Tensor]reference.Value{
			input:   inputValue,
			indices: {Shape: indices.Shape, Data: []float32{1}},
		},
		map[*tensor.Tensor]driver.DevicePtr{weights: pointer},
	)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, results[rows].Data, []float32{
		1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1,
	}, 0)
	compare(t, results[dynamicRows].Data, results[rows].Data, 0)
	compare(t, results[product].Data, []float32{
		-8, 32,
		-16, 64,
		-24, 96,
		-32, 128,
		-40, 160,
	}, 1e-5)
}

func TestExecutorQ6KDeviceEmbeddingAndMulMat(t *testing.T) {
	cudatest.Require(t)
	worker := newFixtureWorker(t)
	storage := make([]byte, 420)
	for row := 0; row < 2; row++ {
		offset := row * 210
		for index := 0; index < 16; index++ {
			storage[offset+192+index] = 1
		}
		binary.LittleEndian.PutUint16(storage[offset+208:], 0x3c00)
	}
	storage[0] = 0x0f
	storage[128] = 0xe4
	pointer := copyFixtureDeviceBytes(t, worker, storage)
	cuda := newFixtureExecutorWithWorker(t, worker)
	builder := tensor.NewBuilder()
	weights := builder.Input("weights", dtype.Q6K, tensor.MustShape(256, 2))
	rows := builder.GetRows(weights, []uint32{1})
	input := builder.Input("input", dtype.F32, tensor.MustShape(256, 1))
	product := builder.MulMat(weights, input)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	ones := make([]float32, 256)
	for index := range ones {
		ones[index] = 1
	}
	inputValue, _ := reference.NewValue(input.Shape, ones)
	results, err := cuda.ExecuteWithDeviceFeeds(
		context.Background(),
		[]*tensor.Tensor{rows, product},
		map[*tensor.Tensor]reference.Value{input: inputValue},
		map[*tensor.Tensor]driver.DevicePtr{weights: pointer},
	)
	if err != nil {
		t.Fatal(err)
	}
	wantRow := make([]float32, 256)
	for index := range wantRow {
		wantRow[index] = -32
	}
	compare(t, results[rows].Data, wantRow, 0)
	compare(t, results[product].Data, []float32{-8081, -8192}, 1e-5)
}

func TestExecutorQ4KDeviceEmbeddingAndMulMat(t *testing.T) {
	testExecutorKQuantEmbeddingAndMulMat(t, dtype.Q4K)
}

func TestExecutorQ5KDeviceEmbeddingAndMulMat(t *testing.T) {
	testExecutorKQuantEmbeddingAndMulMat(t, dtype.Q5K)
}

func TestExecutorQ2KDeviceEmbeddingAndMulMat(t *testing.T) {
	testExecutorKQuantEmbeddingAndMulMat(t, dtype.Q2K)
}

func TestExecutorQ3KDeviceEmbeddingAndMulMat(t *testing.T) {
	testExecutorKQuantEmbeddingAndMulMat(t, dtype.Q3K)
}

func TestExecutorIQ4XSDeviceEmbeddingAndMulMat(t *testing.T) {
	testExecutorKQuantEmbeddingAndMulMat(t, dtype.IQ4XS)
}

func TestExecutorIQCodebookDeviceEmbeddingAndMulMat(t *testing.T) {
	for _, dataType := range []dtype.Type{
		dtype.IQ2XXS,
		dtype.IQ2XS,
		dtype.IQ2S,
		dtype.IQ3XXS,
		dtype.IQ3S,
		dtype.IQ1S,
		dtype.IQ1M,
	} {
		t.Run(dataType.String(), func(t *testing.T) {
			testExecutorKQuantEmbeddingAndMulMat(t, dataType)
		})
	}
}

func TestExecutorTQ2_0DeviceEmbeddingAndMulMat(t *testing.T) {
	testExecutorKQuantEmbeddingAndMulMat(t, dtype.TQ2_0)
}

func TestExecutorTQ1_0DeviceEmbeddingAndMulMat(t *testing.T) {
	testExecutorKQuantEmbeddingAndMulMat(t, dtype.TQ1_0)
}

func TestExecutorQ8KDeviceEmbeddingAndMulMat(t *testing.T) {
	testExecutorKQuantEmbeddingAndMulMat(t, dtype.Q8K)
}

func TestExecutorClassicQuantDeviceEmbeddingAndMulMat(t *testing.T) {
	for _, dataType := range []dtype.Type{
		dtype.Q4_0,
		dtype.Q4_1,
		dtype.Q5_0,
		dtype.Q5_1,
		dtype.Q8_1,
		dtype.IQ4NL,
		dtype.MXFP4,
	} {
		t.Run(dataType.String(), func(t *testing.T) {
			testExecutorClassicQuantEmbeddingAndMulMat(t, dataType)
		})
	}
}

func TestExecutorSmallQuantDeviceEmbeddingAndMulMat(t *testing.T) {
	for _, dataType := range []dtype.Type{dtype.Q1_0, dtype.Q2_0, dtype.NVFP4} {
		t.Run(dataType.String(), func(t *testing.T) {
			testExecutorSmallQuantEmbeddingAndMulMat(t, dataType)
		})
	}
}

func testExecutorSmallQuantEmbeddingAndMulMat(t *testing.T, dataType dtype.Type) {
	t.Helper()
	cudatest.Require(t)
	traits, ok := dataType.Traits()
	if !ok {
		t.Fatalf("missing traits for %s", dataType)
	}
	storage := make([]byte, int(traits.TypeSize)*2)
	for row := 0; row < 2; row++ {
		offset := row * int(traits.TypeSize)
		if dataType == dtype.NVFP4 {
			for index := 0; index < 4; index++ {
				storage[offset+index] = byte(64 + row + index)
			}
			for index := 4; index < int(traits.TypeSize); index++ {
				storage[offset+index] = byte(index*29 + row)
			}
		} else {
			binary.LittleEndian.PutUint16(
				storage[offset:],
				uint16(0x3800+row*0x0400),
			)
			for index := 2; index < int(traits.TypeSize); index++ {
				storage[offset+index] = byte(index*29 + row)
			}
		}
	}
	expected, err := quant.Dequantize(dataType, storage, traits.BlockSize*2)
	if err != nil {
		t.Fatal(err)
	}

	worker := newFixtureWorker(t)
	pointer := copyFixtureDeviceBytes(t, worker, storage)
	cuda := newFixtureExecutorWithWorker(t, worker)

	width := traits.BlockSize
	builder := tensor.NewBuilder()
	weights := builder.Input("weights", dataType, tensor.MustShape(width, 2))
	rows := builder.GetRows(weights, []uint32{1})
	input := builder.Input("input", dtype.F32, tensor.MustShape(width, 1))
	product := builder.MulMat(weights, input)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	ones := make([]float32, int(width))
	for index := range ones {
		ones[index] = 1
	}
	inputValue, _ := reference.NewValue(input.Shape, ones)
	results, err := cuda.ExecuteWithDeviceFeeds(
		context.Background(),
		[]*tensor.Tensor{rows, product},
		map[*tensor.Tensor]reference.Value{input: inputValue},
		map[*tensor.Tensor]driver.DevicePtr{weights: pointer},
	)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, results[rows].Data, expected[width:], 1e-5)
	wantProduct := []float32{0, 0}
	for row := range 2 {
		for _, value := range expected[uint64(row)*width : uint64(row+1)*width] {
			wantProduct[row] += value
		}
	}
	compare(t, results[product].Data, wantProduct, 1e-4)
}

func testExecutorClassicQuantEmbeddingAndMulMat(t *testing.T, dataType dtype.Type) {
	t.Helper()
	cudatest.Require(t)
	traits, ok := dataType.Traits()
	if !ok {
		t.Fatalf("missing traits for %s", dataType)
	}
	storage := make([]byte, int(traits.TypeSize)*2)
	for row := 0; row < 2; row++ {
		offset := row * int(traits.TypeSize)
		binary.LittleEndian.PutUint16(storage[offset:], uint16(0x3800+row*0x0400))
		quantizedOffset := 2
		switch dataType {
		case dtype.MXFP4:
			storage[offset] = byte(128 + row)
			quantizedOffset = 1
		case dtype.Q8_1:
			binary.LittleEndian.PutUint16(storage[offset+2:], 0x4200)
			quantizedOffset = 4
		case dtype.Q4_1:
			binary.LittleEndian.PutUint16(storage[offset+2:], uint16(0xbc00+row*0x0400))
			quantizedOffset = 4
		case dtype.Q5_0:
			for index := 0; index < 4; index++ {
				storage[offset+2+index] = byte(index*37 + row)
			}
			quantizedOffset = 6
		case dtype.Q5_1:
			binary.LittleEndian.PutUint16(storage[offset+2:], uint16(0xbc00+row*0x0400))
			for index := 0; index < 4; index++ {
				storage[offset+4+index] = byte(index*37 + row)
			}
			quantizedOffset = 8
		}
		for index := 0; index < 16; index++ {
			storage[offset+quantizedOffset+index] = byte(index | (15-index)<<4)
		}
	}
	expected, err := quant.Dequantize(dataType, storage, 64)
	if err != nil {
		t.Fatal(err)
	}

	worker := newFixtureWorker(t)
	pointer := copyFixtureDeviceBytes(t, worker, storage)
	cuda := newFixtureExecutorWithWorker(t, worker)

	builder := tensor.NewBuilder()
	weights := builder.Input("weights", dataType, tensor.MustShape(32, 2))
	rows := builder.GetRows(weights, []uint32{1})
	input := builder.Input("input", dtype.F32, tensor.MustShape(32, 1))
	product := builder.MulMat(weights, input)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	ones := make([]float32, 32)
	for index := range ones {
		ones[index] = 1
	}
	inputValue, _ := reference.NewValue(input.Shape, ones)
	results, err := cuda.ExecuteWithDeviceFeeds(
		context.Background(),
		[]*tensor.Tensor{rows, product},
		map[*tensor.Tensor]reference.Value{input: inputValue},
		map[*tensor.Tensor]driver.DevicePtr{weights: pointer},
	)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, results[rows].Data, expected[32:], 1e-5)
	wantProduct := []float32{0, 0}
	for row := range 2 {
		for _, value := range expected[row*32 : (row+1)*32] {
			wantProduct[row] += value
		}
	}
	compare(t, results[product].Data, wantProduct, 1e-4)
}

func testExecutorKQuantEmbeddingAndMulMat(t *testing.T, dataType dtype.Type) {
	t.Helper()
	cudatest.Require(t)
	traits, ok := dataType.Traits()
	if !ok {
		t.Fatalf("missing traits for %s", dataType)
	}
	storage := make([]byte, int(traits.TypeSize)*2)
	for row := 0; row < 2; row++ {
		offset := row * int(traits.TypeSize)
		switch dataType {
		case dtype.TQ1_0:
			for index := 0; index < 52; index++ {
				storage[offset+index] = byte(index*13 + row)
			}
			binary.LittleEndian.PutUint16(
				storage[offset+52:],
				uint16(0x3800+row*0x0400),
			)
		case dtype.TQ2_0:
			for index := 0; index < 64; index++ {
				storage[offset+index] = byte(index*13 + row)
			}
			binary.LittleEndian.PutUint16(
				storage[offset+64:],
				uint16(0x3800+row*0x0400),
			)
		case dtype.Q8K:
			binary.LittleEndian.PutUint32(
				storage[offset:],
				math.Float32bits(0.25+float32(row)*0.25),
			)
			for index := 0; index < 256; index++ {
				storage[offset+4+index] = byte(int8((index+row)%127 - 63))
			}
		case dtype.IQ4XS:
			binary.LittleEndian.PutUint16(storage[offset:], 0x3800)
			binary.LittleEndian.PutUint16(storage[offset+2:], uint16(0xaaaa+row))
			for index := 0; index < 4; index++ {
				storage[offset+4+index] = byte(index*17 + row)
			}
			for index := 0; index < 128; index++ {
				storage[offset+8+index] = byte(index*11 + row)
			}
		case dtype.IQ2XXS, dtype.IQ2XS, dtype.IQ2S,
			dtype.IQ3XXS, dtype.IQ3S, dtype.IQ1S, dtype.IQ1M:
			for index := 0; index < int(traits.TypeSize); index++ {
				storage[offset+index] = byte(index*73 + 19 + row*11)
			}
			if dataType != dtype.IQ1M {
				binary.LittleEndian.PutUint16(
					storage[offset:],
					uint16(0x3800+row*0x0400),
				)
			} else {
				// Encode binary16 0.5/1.0 across high nibbles of
				// IQ1_M's four packed scale words
				scaleMiddleNibble := byte(0x80)
				if row == 1 {
					scaleMiddleNibble = 0xc0
				}
				storage[offset+49] &= 0x0f
				storage[offset+51] =
					storage[offset+51]&0x0f | scaleMiddleNibble
				storage[offset+53] = storage[offset+53]&0x0f | 0x30
				storage[offset+55] &= 0x0f
			}
		case dtype.Q2K:
			for index := 0; index < 16; index++ {
				storage[offset+index] = byte(0x21 + row)
			}
			for index := 0; index < 64; index++ {
				storage[offset+16+index] = byte((index + row) % 4)
			}
			binary.LittleEndian.PutUint16(storage[offset+80:], 0x3c00)
			binary.LittleEndian.PutUint16(storage[offset+82:], 0x3c00)
		case dtype.Q3K:
			for index := 0; index < 32; index++ {
				storage[offset+index] = byte(index*3 + row)
			}
			for index := 0; index < 64; index++ {
				storage[offset+32+index] = byte(index*5 + row)
			}
			for index := 0; index < 12; index++ {
				storage[offset+96+index] = byte(index*7 + row)
			}
			binary.LittleEndian.PutUint16(storage[offset+108:], 0x3c00)
		case dtype.Q4K, dtype.Q5K:
			binary.LittleEndian.PutUint16(storage[offset:], 0x3c00)
			binary.LittleEndian.PutUint16(storage[offset+2:], 0x3c00)
			storage[offset+4] = byte(row + 1)
			storage[offset+8] = byte(2 - row)
			quantizedOffset := 16
			if dataType == dtype.Q5K {
				quantizedOffset = 48
				storage[offset+16] = byte(row)
			}
			for index := 0; index < 32; index++ {
				storage[offset+quantizedOffset+index] = byte(index % 16)
			}
		}
	}
	expected, err := quant.Dequantize(dataType, storage, 512)
	if err != nil {
		t.Fatal(err)
	}

	worker := newFixtureWorker(t)
	pointer := copyFixtureDeviceBytes(t, worker, storage)
	cuda := newFixtureExecutorWithWorker(t, worker)

	builder := tensor.NewBuilder()
	weights := builder.Input("weights", dataType, tensor.MustShape(256, 2))
	rows := builder.GetRows(weights, []uint32{1})
	input := builder.Input("input", dtype.F32, tensor.MustShape(256, 1))
	product := builder.MulMat(weights, input)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	ones := make([]float32, 256)
	for index := range ones {
		ones[index] = 1
	}
	inputValue, _ := reference.NewValue(input.Shape, ones)
	results, err := cuda.ExecuteWithDeviceFeeds(
		context.Background(),
		[]*tensor.Tensor{rows, product},
		map[*tensor.Tensor]reference.Value{input: inputValue},
		map[*tensor.Tensor]driver.DevicePtr{weights: pointer},
	)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, results[rows].Data, expected[256:], 1e-5)
	wantProduct := []float32{0, 0}
	for row := range 2 {
		for _, value := range expected[row*256 : (row+1)*256] {
			wantProduct[row] += value
		}
	}
	compare(t, results[product].Data, wantProduct, 1e-4)
}
