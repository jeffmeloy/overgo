//go:build windows

package executor

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"testing"

	"llamacpp2go/internal/cuda/device"
	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/quant"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
)

func TestAllZeroFloat32(t *testing.T) {
	if !allZeroFloat32([]float32{0, 0, 0}) {
		t.Fatal("zero values were not recognized")
	}
	for _, values := range [][]float32{nil, {}, {0, 1}, {0, -1}} {
		if allZeroFloat32(values) {
			t.Fatalf("nonzero/empty values were recognized: %v", values)
		}
	}
}

func TestExecutorMatchesReference(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(8, 3)
	left := builder.Input("left", dtype.F32, shape)
	right := builder.Input("right", dtype.F32, shape)
	add := builder.Add(left, right)
	multiply := builder.Multiply(add, right)
	scale := builder.Scale(multiply, 0.25)
	layerNorm := builder.LayerNorm(scale, 1e-5)
	reluSquared := builder.ReLUSquared(scale)
	norm := builder.RMSNorm(scale, 1e-5)
	silu := builder.SiLU(norm)
	sigmoid := builder.Sigmoid(norm)
	softplus := builder.Softplus(builder.GELU(silu))
	xielu := builder.XIELU(scale, 0.8, 0.2, 0.5, -0.1)
	l2Norm := builder.L2Norm(builder.Add(sigmoid, softplus), 1e-6)
	output := builder.Softmax(l2Norm)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	leftData := make([]float32, 24)
	rightData := make([]float32, 24)
	for i := range leftData {
		leftData[i] = float32(i-11) / 3
		rightData[i] = float32((i%7)+1) / 5
	}
	leftValue, _ := reference.NewValue(shape, leftData)
	rightValue, _ := reference.NewValue(shape, rightData)
	feeds := map[*tensor.Tensor]reference.Value{left: leftValue, right: rightValue}
	want, err := reference.Execute([]*tensor.Tensor{output, layerNorm, reluSquared, xielu}, feeds)
	if err != nil {
		t.Fatal(err)
	}

	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(
		context.Background(),
		[]*tensor.Tensor{output, layerNorm, reluSquared, xielu},
		feeds,
	)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[output].Data, 2e-5)
	compare(t, got[layerNorm].Data, want[layerNorm].Data, 2e-5)
	compare(t, got[reluSquared].Data, want[reluSquared].Data, 2e-5)
	compare(t, got[xielu].Data, want[xielu].Data, 2e-5)
}

func TestExecutorMoEMatchesReference(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(2, 2))
	router := builder.Input("router", dtype.F32, tensor.MustShape(2, 3))
	gate := builder.Input("gate", dtype.F32, tensor.MustShape(2, 2, 3))
	up := builder.Input("up", dtype.F32, tensor.MustShape(2, 2, 3))
	down := builder.Input("down", dtype.F32, tensor.MustShape(2, 2, 3))
	output := builder.MoE(input, router, gate, up, down, 2, true, 1.25)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	value := func(shape tensor.Shape, data []float32) reference.Value {
		result, err := reference.NewValue(shape, data)
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: value(input.Shape, []float32{0.5, -1, 1.5, 0.25}),
		router: value(router.Shape, []float32{
			1, -0.5,
			-0.25, 0.75,
			0.5, 0.5,
		}),
		gate: value(gate.Shape, []float32{
			1, 0, 0, 1,
			0.5, -0.5, 1, 0.25,
			-1, 0.5, 0.25, 1,
		}),
		up: value(up.Shape, []float32{
			0.25, 1, -0.5, 0.75,
			1, 0.5, 0.5, -1,
			0.75, -0.25, 1, 0.5,
		}),
		down: value(down.Shape, []float32{
			1, -0.5, 0.25, 0.75,
			-0.25, 1, 0.5, -0.75,
			0.75, 0.25, -1, 0.5,
		}),
	}
	want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[output].Data, 5e-5)
}

func TestExecutorRetainedOutputLifetime(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(4)
	left := builder.Input("left", dtype.F32, shape)
	right := builder.Input("right", dtype.F32, shape)
	output := builder.Add(left, right)
	leftValue, _ := reference.NewValue(shape, []float32{1, 2, 3, 4})
	rightValue, _ := reference.NewValue(shape, []float32{10, 20, 30, 40})
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	before, err := cuda.worker.MemoryStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	retained, err := cuda.ExecuteRetainedWithDeviceFeeds(
		context.Background(),
		[]*tensor.Tensor{output},
		map[*tensor.Tensor]reference.Value{left: leftValue, right: rightValue},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	during, err := cuda.worker.MemoryStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if during.CurrentBytes-before.CurrentBytes != 16 {
		t.Fatalf("retained bytes = %d, want 16", during.CurrentBytes-before.CurrentBytes)
	}
	got, err := retained.CopyToHost(context.Background(), output)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got.Data, []float32{11, 22, 33, 44}, 0)
	if err := retained.Release(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := retained.Release(context.Background()); err != nil {
		t.Fatal(err)
	}
	after, err := cuda.worker.MemoryStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if after.CurrentBytes != before.CurrentBytes {
		t.Fatalf("bytes after release = %d, want %d", after.CurrentBytes, before.CurrentBytes)
	}
	if _, ok := retained.Value(output); ok {
		t.Fatal("released output remains accessible")
	}
}

func TestExecutorCopyDeviceValuesConcatenatesSegments(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(4)
	input := builder.Input("input", dtype.F32, shape)
	output := builder.Scale(input, 1)
	inputValue, _ := reference.NewValue(shape, []float32{1, 2, 3, 4})
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	retained, err := cuda.ExecuteRetainedWithDeviceFeeds(
		context.Background(),
		[]*tensor.Tensor{output},
		map[*tensor.Tensor]reference.Value{input: inputValue},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer retained.Release(context.Background())
	source, ok := retained.Value(output)
	if !ok {
		t.Fatal("retained output is unavailable")
	}
	copiedOwner, copied, err := cuda.CopyDeviceValues(
		context.Background(),
		[]DeviceCopy{{
			Shape: shape,
			Segments: []DeviceCopySegment{
				{Source: source.Pointer + 8, Bytes: 8},
				{Source: source.Pointer, Bytes: 8},
			},
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer copiedOwner.Release(context.Background())
	data := make([]float32, 4)
	err = cuda.worker.Do(context.Background(), func(state *device.State) error {
		return state.Driver.MemcpyDtoH(float32Bytes(data), copied[0].Pointer)
	})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, data, []float32{3, 4, 1, 2}, 0)
}

func TestExecutorMulMatMatchesReference(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	builder := tensor.NewBuilder()
	left := builder.Input("left", dtype.F32, tensor.MustShape(5, 3))
	right := builder.Input("right", dtype.F32, tensor.MustShape(5, 4))
	output := builder.MulMat(left, right)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	leftValue, _ := reference.NewValue(left.Shape, []float32{
		1, 2, 3, 4, 5,
		2, 3, 4, 5, 6,
		3, 4, 5, 6, 7,
	})
	rightValue, _ := reference.NewValue(right.Shape, []float32{
		1, 0, 0, 0, 0,
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		1, 1, 1, 1, 1,
	})
	feeds := map[*tensor.Tensor]reference.Value{left: leftValue, right: rightValue}
	want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[output].Data, 1e-5)
}

func TestExecutorSSMConvMatchesReference(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(6, 3, 2))
	weights := builder.Input("weights", dtype.F32, tensor.MustShape(4, 3))
	output := builder.SSMConv(input, weights)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	inputValue := patternedValue(input.Shape, 7, 0.2, -0.3)
	weightValue := patternedValue(weights.Shape, 11, 0.1, 0.05)
	feeds := map[*tensor.Tensor]reference.Value{
		input:   inputValue,
		weights: weightValue,
	}
	want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[output].Data, 1e-5)
}

func TestExecutorGatedDeltaNetMatchesReference(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	for _, gateWidth := range []uint64{1, 4} {
		t.Run(fmt.Sprintf("gate_width_%d", gateWidth), func(t *testing.T) {
			builder := tensor.NewBuilder()
			q := builder.Input("q", dtype.F32, tensor.MustShape(4, 1, 3, 2))
			k := builder.Input("k", dtype.F32, tensor.MustShape(4, 1, 3, 2))
			v := builder.Input("v", dtype.F32, tensor.MustShape(4, 2, 3, 2))
			gate := builder.Input("gate", dtype.F32, tensor.MustShape(gateWidth, 2, 3, 2))
			beta := builder.Input("beta", dtype.F32, tensor.MustShape(1, 2, 3, 2))
			state := builder.Input("state", dtype.F32, tensor.MustShape(4, 4, 2, 2))
			output := builder.GatedDeltaNet(q, k, v, gate, beta, state)
			if err := builder.Err(); err != nil {
				t.Fatal(err)
			}
			feeds := map[*tensor.Tensor]reference.Value{
				q:     patternedValue(q.Shape, 7, 0.08, -0.15),
				k:     patternedValue(k.Shape, 11, 0.06, -0.1),
				v:     patternedValue(v.Shape, 13, 0.09, 0.03),
				gate:  patternedValue(gate.Shape, 5, 0.02, -0.08),
				beta:  patternedValue(beta.Shape, 3, 0.03, 0.4),
				state: patternedValue(state.Shape, 17, 0.04, -0.07),
			}
			want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
			if err != nil {
				t.Fatal(err)
			}
			cuda, err := New(0)
			if err != nil {
				t.Fatal(err)
			}
			defer cuda.Close()
			got, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds)
			if err != nil {
				t.Fatal(err)
			}
			compare(t, got[output].Data, want[output].Data, 3e-5)
		})
	}
}

func TestExecutorQwen35LayoutOperationsMatchReference(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(16, 5))
	query := builder.GroupSlice(input, 0, 2, 4, 4)
	gate := builder.GroupSlice(input, 2, 2, 4, 4)
	transposed := builder.Transpose2D(input)
	state := builder.Input("state", dtype.F32, tensor.MustShape(3, 16))
	convInput := builder.Concat(state, transposed, 0)
	slice := builder.FlatSlice(convInput, 7, 35)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: patternedValue(input.Shape, 11, 0.13, -0.2),
		state: patternedValue(state.Shape, 7, 0.09, 0.05),
	}
	outputs := []*tensor.Tensor{query, gate, transposed, convInput, slice}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	for _, output := range outputs {
		compare(t, got[output].Data, want[output].Data, 0)
	}
}

func TestExecutorQwen35BlocksMatchReference(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	for _, recurrent := range []bool{false, true} {
		name := "attention"
		if recurrent {
			name = "recurrent"
		}
		t.Run(name, func(t *testing.T) {
			builder := tensor.NewBuilder()
			spec := qwen35ExecutorSpec()
			input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
			weights, feeds := qwen35ExecutorWeights(builder, spec, recurrent)
			feeds[input] = patternedValue(input.Shape, 17, 0.04, -0.08)
			var convState, ssmState *tensor.Tensor
			if recurrent {
				convState = builder.Input("conv_state", dtype.F32, tensor.MustShape(2, 8))
				ssmState = builder.Input("ssm_state", dtype.F32, tensor.MustShape(2, 2, 2, 1))
				feeds[convState] = patternedValue(convState.Shape, 7, 0.03, -0.02)
				feeds[ssmState] = patternedValue(ssmState.Shape, 11, 0.02, 0.01)
			}
			result, err := model.BuildQwen35BlockCached(
				builder,
				input,
				spec,
				weights,
				[]uint32{0, 1},
				recurrent,
				nil,
				nil,
				convState,
				ssmState,
			)
			if err != nil {
				t.Fatal(err)
			}
			outputs := []*tensor.Tensor{result.Output}
			if recurrent {
				outputs = append(outputs, result.ConvState, result.SSMState)
			} else {
				outputs = append(outputs, result.Key, result.Value)
			}
			want, err := reference.Execute(outputs, feeds)
			if err != nil {
				t.Fatal(err)
			}
			cuda, err := New(0)
			if err != nil {
				t.Fatal(err)
			}
			defer cuda.Close()
			got, err := cuda.Execute(context.Background(), outputs, feeds)
			if err != nil {
				t.Fatal(err)
			}
			for _, output := range outputs {
				compare(t, got[output].Data, want[output].Data, 4e-5)
			}
		})
	}
}

func TestExecutorRoPEMultiMatchesReference(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(12, 3, 4, 2))
	positions := [4][]uint32{
		{0, 1, 2, 3},
		{3, 5, 7, 9},
		{2, 4, 6, 8},
		{11, 13, 17, 19},
	}
	output := builder.RoPEMulti(
		input,
		positions,
		[4]int32{2, 1, 1, 0},
		8,
		1_000_000,
	)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: patternedValue(input.Shape, 17, 0.08, -0.2),
	}
	want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[output].Data, 2e-6)
}

func qwen35ExecutorSpec() model.Spec {
	return model.Spec{
		Architecture:          "qwen35",
		EmbeddingLength:       8,
		FeedForwardLength:     12,
		HeadCount:             2,
		HeadCountKV:           1,
		KeyLength:             4,
		ValueLength:           4,
		RopeFrequencyBase:     10000,
		RMSNormEpsilon:        1e-6,
		RopeDimensionCount:    4,
		RopeSections:          [4]int32{1, 1, 0, 0},
		SSMConvKernel:         3,
		SSMInnerSize:          4,
		SSMStateSize:          2,
		SSMTimeStepRank:       2,
		SSMGroupCount:         1,
		FullAttentionInterval: 4,
	}
}

func qwen35ExecutorWeights(
	builder *tensor.Builder,
	spec model.Spec,
	recurrent bool,
) (model.LayerGraphWeights, map[*tensor.Tensor]reference.Value) {
	feeds := make(map[*tensor.Tensor]reference.Value)
	seed := 1
	input := func(name string, shape tensor.Shape, scale, bias float32) *tensor.Tensor {
		node := builder.Input(name, dtype.F32, shape)
		feeds[node] = patternedValue(shape, seed, scale, bias)
		seed += 2
		return node
	}
	embedding := uint64(spec.EmbeddingLength)
	feedForward := uint64(spec.FeedForwardLength)
	result := model.LayerGraphWeights{
		AttentionNorm: input("attn_norm", tensor.MustShape(embedding), 0.03, 0.9),
		FeedForwardNorm: input(
			"post_attention_norm",
			tensor.MustShape(embedding),
			0.03,
			0.9,
		),
		FeedForwardGate: input(
			"ffn_gate",
			tensor.MustShape(embedding, feedForward),
			0.025,
			-0.04,
		),
		FeedForwardUp: input(
			"ffn_up",
			tensor.MustShape(embedding, feedForward),
			0.02,
			0.03,
		),
		FeedForwardDown: input(
			"ffn_down",
			tensor.MustShape(feedForward, embedding),
			0.02,
			-0.01,
		),
	}
	if !recurrent {
		headWidth := uint64(spec.KeyLength)
		result.AttentionQ = input(
			"attn_q",
			tensor.MustShape(embedding, 2*uint64(spec.HeadCount)*headWidth),
			0.02,
			-0.03,
		)
		result.AttentionK = input(
			"attn_k",
			tensor.MustShape(embedding, uint64(spec.HeadCountKV)*headWidth),
			0.02,
			0.01,
		)
		result.AttentionV = input(
			"attn_v",
			tensor.MustShape(embedding, uint64(spec.HeadCountKV)*uint64(spec.ValueLength)),
			0.025,
			-0.02,
		)
		result.AttentionOutput = input(
			"attn_output",
			tensor.MustShape(embedding, embedding),
			0.02,
			0.01,
		)
		result.AttentionQNorm = input("attn_q_norm", tensor.MustShape(headWidth), 0.02, 0.95)
		result.AttentionKNorm = input("attn_k_norm", tensor.MustShape(headWidth), 0.02, 0.95)
		return result, feeds
	}
	stateWidth := uint64(spec.SSMStateSize)
	keyDimension := stateWidth * uint64(spec.SSMGroupCount)
	valueDimension := uint64(spec.SSMInnerSize)
	channels := 2*keyDimension + valueDimension
	valueHeads := uint64(spec.SSMTimeStepRank)
	result.AttentionQKV = input(
		"attn_qkv",
		tensor.MustShape(embedding, channels),
		0.025,
		-0.03,
	)
	result.AttentionGate = input(
		"attn_gate",
		tensor.MustShape(embedding, valueDimension),
		0.02,
		0.01,
	)
	result.SSMConv1D = input(
		"ssm_conv1d",
		tensor.MustShape(uint64(spec.SSMConvKernel), channels),
		0.03,
		-0.02,
	)
	result.SSMTimeStep = input("ssm_dt", tensor.MustShape(valueHeads), 0.02, -0.1)
	result.SSMA = input("ssm_a", tensor.MustShape(valueHeads), 0.01, -0.5)
	result.SSMBeta = input(
		"ssm_beta",
		tensor.MustShape(embedding, valueHeads),
		0.025,
		0.02,
	)
	result.SSMAlpha = input(
		"ssm_alpha",
		tensor.MustShape(embedding, valueHeads),
		0.02,
		-0.01,
	)
	result.SSMNorm = input("ssm_norm", tensor.MustShape(stateWidth), 0.02, 0.95)
	result.SSMOutput = input(
		"ssm_out",
		tensor.MustShape(valueDimension, embedding),
		0.025,
		-0.01,
	)
	return result, feeds
}

func TestExecutorEmbeddingBroadcastSwiGLUAndRoPEMatchesReference(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	builder := tensor.NewBuilder()
	table := builder.Input("table", dtype.F32, tensor.MustShape(4, 3))
	weight := builder.Input("weight", dtype.F32, tensor.MustShape(4))
	up := builder.Input("up", dtype.F32, tensor.MustShape(4, 2))
	embedding := builder.GetRows(table, []uint32{2, 0})
	swiglu := builder.SwiGLU(builder.WeightedRMSNorm(embedding, weight, 1e-5), up)
	ropeInput := builder.Input("rope", dtype.F32, tensor.MustShape(4, 1, 2))
	ropeFactors := builder.Input("rope_factors", dtype.F32, tensor.MustShape(2))
	rope := builder.RoPENeoX(ropeInput, []uint32{0, 17}, 4, 1_000_000)
	normalRope := builder.RoPENormal(ropeInput, []uint32{0, 17}, 4, 1_000_000)
	factoredRope := builder.RoPENormalScaledWithFactors(
		ropeInput,
		[]uint32{0, 17},
		4,
		1_000_000,
		0.25,
		ropeFactors,
	)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	tableValue, _ := reference.NewValue(table.Shape, []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	})
	weightValue, _ := reference.NewValue(weight.Shape, []float32{1, 2, 3, 4})
	upValue, _ := reference.NewValue(up.Shape, []float32{1, 2, 3, 4, 4, 3, 2, 1})
	ropeValue, _ := reference.NewValue(ropeInput.Shape, []float32{
		1, 2, 3, 4,
		-1, -2, -3, -4,
	})
	ropeFactorValue, _ := reference.NewValue(ropeFactors.Shape, []float32{1, 8})
	feeds := map[*tensor.Tensor]reference.Value{
		table:       tableValue,
		weight:      weightValue,
		up:          upValue,
		ropeInput:   ropeValue,
		ropeFactors: ropeFactorValue,
	}
	outputs := []*tensor.Tensor{embedding, swiglu, rope, normalRope, factoredRope}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[embedding].Data, want[embedding].Data, 0)
	compare(t, got[swiglu].Data, want[swiglu].Data, 3e-5)
	compare(t, got[rope].Data, want[rope].Data, 3e-5)
	compare(t, got[normalRope].Data, want[normalRope].Data, 3e-5)
	compare(t, got[factoredRope].Data, want[factoredRope].Data, 3e-5)
}

func TestExecutorDenseQwen3BlockMatchesReference(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	builder := tensor.NewBuilder()
	spec := model.Spec{
		Architecture:      "qwen3",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 1_000_000,
		RMSNormEpsilon:    1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:   builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:      builder.Input("attn_q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:      builder.Input("attn_k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:      builder.Input("attn_v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput: builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:  builder.Input("attn_q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:  builder.Input("attn_k_norm", dtype.F32, tensor.MustShape(4)),
		AttentionQBias:  builder.Input("attn_q_bias", dtype.F32, tensor.MustShape(8)),
		AttentionKBias:  builder.Input("attn_k_bias", dtype.F32, tensor.MustShape(4)),
		AttentionVBias:  builder.Input("attn_v_bias", dtype.F32, tensor.MustShape(4)),
		AttentionOutputBias: builder.Input(
			"attn_output_bias",
			dtype.F32,
			tensor.MustShape(8),
		),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate: builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
		FeedForwardGateBias: builder.Input(
			"ffn_gate_bias",
			dtype.F32,
			tensor.MustShape(12),
		),
		FeedForwardUpBias: builder.Input(
			"ffn_up_bias",
			dtype.F32,
			tensor.MustShape(12),
		),
		FeedForwardDownBias: builder.Input(
			"ffn_down_bias",
			dtype.F32,
			tensor.MustShape(8),
		),
	}
	output, err := model.BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1, 2})
	if err != nil {
		t.Fatal(err)
	}
	feeds := make(map[*tensor.Tensor]reference.Value)
	feeds[input] = patternedValue(input.Shape, 1, 0.25, 0)
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ,
		weights.AttentionK,
		weights.AttentionV,
		weights.AttentionOutput,
		weights.FeedForwardGate,
		weights.FeedForwardUp,
		weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+2, 0.08, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm,
		weights.AttentionQNorm,
		weights.AttentionKNorm,
		weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+13, 0.03, 1)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQBias,
		weights.AttentionKBias,
		weights.AttentionVBias,
		weights.AttentionOutputBias,
		weights.FeedForwardGateBias,
		weights.FeedForwardUpBias,
		weights.FeedForwardDownBias,
	} {
		feeds[node] = patternedValue(node.Shape, index+20, 0.02, -0.01)
	}
	want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[output].Data, 3e-4)
}

func TestExecutorDenseOLMo2BlockMatchesReference(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	builder := tensor.NewBuilder()
	spec := model.Spec{
		Architecture:      "olmo2",
		BlockCount:        4,
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		RopeFrequencySWA:  10000,
		RMSNormEpsilon:    1e-6,
		SlidingWindow:     2,
		SlidingPattern:    4,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionQ:        builder.Input("attn_q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:        builder.Input("attn_k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:        builder.Input("attn_v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:   builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:    builder.Input("attn_q_norm", dtype.F32, tensor.MustShape(8)),
		AttentionKNorm:    builder.Input("attn_k_norm", dtype.F32, tensor.MustShape(4)),
		AttentionPostNorm: builder.Input("post_attention_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:   builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:     builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:   builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
		FeedForwardPostNorm: builder.Input(
			"post_ffw_norm",
			dtype.F32,
			tensor.MustShape(8),
		),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder,
		input,
		spec,
		weights,
		[]uint32{0, 1},
		nil,
		nil,
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: patternedValue(input.Shape, 1, 0.2, 0),
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ,
		weights.AttentionK,
		weights.AttentionV,
		weights.AttentionOutput,
		weights.FeedForwardGate,
		weights.FeedForwardUp,
		weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+2, 0.06, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQNorm,
		weights.AttentionKNorm,
		weights.AttentionPostNorm,
		weights.FeedForwardPostNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.03, 1)
	}
	want, err := reference.Execute([]*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 3e-4)
}

func TestExecutorDenseSmolLM3NoRoPEBlockMatchesReference(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	builder := tensor.NewBuilder()
	spec := model.Spec{
		Architecture:      "smollm3",
		BlockCount:        4,
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		RMSNormEpsilon:    1e-6,
		AttentionScale:    0.25,
		NoRopeLayerStep:   4,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:   builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:      builder.Input("attn_q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:      builder.Input("attn_k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:      builder.Input("attn_v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput: builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate: builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder,
		input,
		spec,
		weights,
		[]uint32{0, 1},
		nil,
		nil,
		3,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: patternedValue(input.Shape, 1, 0.2, 0),
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ,
		weights.AttentionK,
		weights.AttentionV,
		weights.AttentionOutput,
		weights.FeedForwardGate,
		weights.FeedForwardUp,
		weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+2, 0.06, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm,
		weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.03, 1)
	}
	want, err := reference.Execute([]*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 3e-4)
}

func TestExecutorDenseMiniCPMBlockMatchesReference(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	builder := tensor.NewBuilder()
	spec := model.Spec{
		Architecture:      "minicpm",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		RMSNormEpsilon:    1e-6,
		ResidualScale:     0.25,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:       builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:          builder.Input("attn_q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:          builder.Input("attn_k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:          builder.Input("attn_v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:     builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutputBias: builder.Input("attn_output_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardNorm:     builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:     builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:       builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:     builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
		FeedForwardGateBias: builder.Input("ffn_gate_bias", dtype.F32, tensor.MustShape(12)),
		FeedForwardUpBias:   builder.Input("ffn_up_bias", dtype.F32, tensor.MustShape(12)),
		FeedForwardDownBias: builder.Input("ffn_down_bias", dtype.F32, tensor.MustShape(8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder,
		input,
		spec,
		weights,
		[]uint32{0, 1},
		nil,
		nil,
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: patternedValue(input.Shape, 1, 0.2, 0),
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ,
		weights.AttentionK,
		weights.AttentionV,
		weights.AttentionOutput,
		weights.FeedForwardGate,
		weights.FeedForwardUp,
		weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+2, 0.06, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm,
		weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.03, 1)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionOutputBias,
		weights.FeedForwardGateBias,
		weights.FeedForwardUpBias,
		weights.FeedForwardDownBias,
	} {
		feeds[node] = patternedValue(node.Shape, index+20, 0.02, -0.01)
	}
	want, err := reference.Execute([]*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 3e-4)
}

func TestExecutorDenseGraniteNoRoPEBlockMatchesReference(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	builder := tensor.NewBuilder()
	spec := model.Spec{
		Architecture:      "granite",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		RMSNormEpsilon:    1e-6,
		AttentionScale:    0.25,
		ResidualScale:     0.5,
		RopeDisabled:      true,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:   builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:      builder.Input("attn_q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:      builder.Input("attn_k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:      builder.Input("attn_v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput: builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate: builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder,
		input,
		spec,
		weights,
		[]uint32{0, 1},
		nil,
		nil,
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: patternedValue(input.Shape, 1, 0.2, 0),
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ,
		weights.AttentionK,
		weights.AttentionV,
		weights.AttentionOutput,
		weights.FeedForwardGate,
		weights.FeedForwardUp,
		weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+2, 0.06, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm,
		weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.03, 1)
	}
	want, err := reference.Execute([]*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 3e-4)
}

func TestExecutorDenseMaincoderBlockMatchesReference(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	builder := tensor.NewBuilder()
	spec := model.Spec{
		Architecture:      "maincoder",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		RMSNormEpsilon:    1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:   builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:      builder.Input("attn_q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:      builder.Input("attn_k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:      builder.Input("attn_v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput: builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:  builder.Input("attn_q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:  builder.Input("attn_k_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate: builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder,
		input,
		spec,
		weights,
		[]uint32{0, 1},
		nil,
		nil,
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: patternedValue(input.Shape, 1, 0.2, 0),
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ,
		weights.AttentionK,
		weights.AttentionV,
		weights.AttentionOutput,
		weights.FeedForwardGate,
		weights.FeedForwardUp,
		weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+2, 0.06, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm,
		weights.AttentionQNorm,
		weights.AttentionKNorm,
		weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.03, 1)
	}
	want, err := reference.Execute([]*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 3e-4)
}

func TestExecutorDenseMistral3BlockMatchesReference(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	builder := tensor.NewBuilder()
	spec := model.Spec{
		Architecture:      "mistral3",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		RMSNormEpsilon:    1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:       builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:          builder.Input("attn_q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:          builder.Input("attn_k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:          builder.Input("attn_v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:     builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutputBias: builder.Input("attn_output_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardNorm:     builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:     builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:       builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:     builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
		FeedForwardGateBias: builder.Input("ffn_gate_bias", dtype.F32, tensor.MustShape(12)),
		FeedForwardUpBias:   builder.Input("ffn_up_bias", dtype.F32, tensor.MustShape(12)),
		FeedForwardDownBias: builder.Input("ffn_down_bias", dtype.F32, tensor.MustShape(8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder,
		input,
		spec,
		weights,
		[]uint32{0, 1},
		nil,
		nil,
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: patternedValue(input.Shape, 1, 0.2, 0),
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ,
		weights.AttentionK,
		weights.AttentionV,
		weights.AttentionOutput,
		weights.FeedForwardGate,
		weights.FeedForwardUp,
		weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+2, 0.06, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm,
		weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.03, 1)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionOutputBias,
		weights.FeedForwardGateBias,
		weights.FeedForwardUpBias,
		weights.FeedForwardDownBias,
	} {
		feeds[node] = patternedValue(node.Shape, index+20, 0.02, -0.01)
	}
	want, err := reference.Execute([]*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 3e-4)
}

func TestExecutorDenseOrionBlockMatchesReference(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	builder := tensor.NewBuilder()
	spec := model.Spec{
		Architecture:      "orion",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		LayerNormEpsilon:  1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:       builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionNormBias:   builder.Input("attn_norm_bias", dtype.F32, tensor.MustShape(8)),
		AttentionQ:          builder.Input("attn_q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:          builder.Input("attn_k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:          builder.Input("attn_v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:     builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:     builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardNormBias: builder.Input("ffn_norm_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:     builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:       builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:     builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder,
		input,
		spec,
		weights,
		[]uint32{0, 1},
		nil,
		nil,
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: patternedValue(input.Shape, 1, 0.2, 0),
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ,
		weights.AttentionK,
		weights.AttentionV,
		weights.AttentionOutput,
		weights.FeedForwardGate,
		weights.FeedForwardUp,
		weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+2, 0.06, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm,
		weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.03, 1)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNormBias,
		weights.FeedForwardNormBias,
	} {
		feeds[node] = patternedValue(node.Shape, index+20, 0.02, -0.01)
	}
	want, err := reference.Execute([]*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 5e-4)
}

func TestExecutorDenseStarCoder2BlockMatchesReference(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	builder := tensor.NewBuilder()
	spec := model.Spec{
		Architecture:      "starcoder2",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		LayerNormEpsilon:  1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:       builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionNormBias:   builder.Input("attn_norm_bias", dtype.F32, tensor.MustShape(8)),
		AttentionQ:          builder.Input("attn_q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:          builder.Input("attn_k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:          builder.Input("attn_v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:     builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutputBias: builder.Input("attn_output_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardNorm:     builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardNormBias: builder.Input("ffn_norm_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardUp:       builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUpBias:   builder.Input("ffn_up_bias", dtype.F32, tensor.MustShape(12)),
		FeedForwardDown:     builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
		FeedForwardDownBias: builder.Input("ffn_down_bias", dtype.F32, tensor.MustShape(8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder,
		input,
		spec,
		weights,
		[]uint32{0, 1},
		nil,
		nil,
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: patternedValue(input.Shape, 1, 0.2, 0),
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ,
		weights.AttentionK,
		weights.AttentionV,
		weights.AttentionOutput,
		weights.FeedForwardUp,
		weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+2, 0.06, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm,
		weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.03, 1)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNormBias,
		weights.AttentionOutputBias,
		weights.FeedForwardNormBias,
		weights.FeedForwardUpBias,
		weights.FeedForwardDownBias,
	} {
		feeds[node] = patternedValue(node.Shape, index+20, 0.02, -0.01)
	}
	want, err := reference.Execute([]*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 5e-4)
}

func TestExecutorCachedAttentionMatchesReference(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	builder := tensor.NewBuilder()
	query := builder.Input("query", dtype.F32, tensor.MustShape(4, 2, 1))
	pastKey := builder.Input("past_key", dtype.F32, tensor.MustShape(4, 1, 3))
	newKey := builder.Input("new_key", dtype.F32, tensor.MustShape(4, 1, 1))
	pastValue := builder.Input("past_value", dtype.F32, tensor.MustShape(4, 1, 3))
	newValue := builder.Input("new_value", dtype.F32, tensor.MustShape(4, 1, 1))
	key := builder.Concat(pastKey, newKey, 2)
	value := builder.Concat(pastValue, newValue, 2)
	output := builder.AttentionWithOffset(query, key, value, 0.5, true, 3)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		query:     patternedValue(query.Shape, 1, 0.2, 0),
		pastKey:   patternedValue(pastKey.Shape, 2, 0.15, 0),
		newKey:    patternedValue(newKey.Shape, 3, 0.15, 0),
		pastValue: patternedValue(pastValue.Shape, 4, 0.2, 0),
		newValue:  patternedValue(newValue.Shape, 5, 0.2, 0),
	}
	outputs := []*tensor.Tensor{key, value, output}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[key].Data, want[key].Data, 0)
	compare(t, got[value].Data, want[value].Data, 0)
	compare(t, got[output].Data, want[output].Data, 3e-5)
}

func TestExecutorSoftcappedWindowAttentionMatchesReference(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	builder := tensor.NewBuilder()
	query := builder.Input("query", dtype.F32, tensor.MustShape(4, 2, 2))
	key := builder.Input("key", dtype.F32, tensor.MustShape(4, 1, 5))
	value := builder.Input("value", dtype.F32, tensor.MustShape(4, 1, 5))
	output := builder.AttentionWindowSoftcappedWithOffset(
		query, key, value, 0.5, 1.75, true, 3, 3,
	)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		query: patternedValue(query.Shape, 13, 0.8, 0),
		key:   patternedValue(key.Shape, 17, 0.7, 0),
		value: patternedValue(value.Shape, 19, 0.2, 0),
	}
	want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[output].Data, 3e-5)
}

func TestExecutorT5RelativeBiasAttentionMatchesReference(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	builder := tensor.NewBuilder()
	query := builder.Input("query", dtype.F32, tensor.MustShape(3, 2, 4))
	key := builder.Input("key", dtype.F32, tensor.MustShape(3, 2, 4))
	value := builder.Input("value", dtype.F32, tensor.MustShape(3, 2, 4))
	bias := builder.Input("bias", dtype.F32, tensor.MustShape(2, 32))
	output := builder.AttentionWithRelativeBias(query, key, value, bias, 1)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		query: patternedValue(query.Shape, 3, 0.08, 0),
		key:   patternedValue(key.Shape, 5, 0.07, 0),
		value: patternedValue(value.Shape, 7, 0.1, 0),
		bias:  patternedValue(bias.Shape, 11, 0.04, -0.2),
	}
	want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[output].Data, 3e-5)
}

func TestExecutorUsesPersistentDeviceFeed(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	var pointer driver.DevicePtr
	leftData := []float32{1, 2, 3, 4}
	err = worker.Do(context.Background(), func(state *device.State) error {
		var allocateErr error
		pointer, allocateErr = state.Driver.MemAlloc(uint64(len(leftData) * 4))
		if allocateErr != nil {
			return allocateErr
		}
		return state.Driver.MemcpyHtoD(pointer, float32Bytes(leftData))
	})
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Do(context.Background(), func(state *device.State) error {
		return state.Driver.MemFree(pointer)
	})
	cuda, err := NewWithWorker(worker)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
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
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	storage := make([]byte, 68)
	binary.LittleEndian.PutUint16(storage[0:], 0x3800)  // 0.5
	binary.LittleEndian.PutUint16(storage[34:], 0x3800) // 0.5
	for index := 0; index < 32; index++ {
		storage[2+index] = byte(int8(index - 16))
		storage[36+index] = 2
	}
	var pointer driver.DevicePtr
	err = worker.Do(context.Background(), func(state *device.State) error {
		var allocateErr error
		pointer, allocateErr = state.Driver.MemAlloc(uint64(len(storage)))
		if allocateErr != nil {
			return allocateErr
		}
		return state.Driver.MemcpyHtoD(pointer, storage)
	})
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Do(context.Background(), func(state *device.State) error {
		return state.Driver.MemFree(pointer)
	})
	cuda, err := NewWithWorker(worker)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	builder := tensor.NewBuilder()
	weights := builder.Input("weights", dtype.Q8_0, tensor.MustShape(32, 2))
	rows := builder.GetRows(weights, []uint32{1})
	input := builder.Input("input", dtype.F32, tensor.MustShape(32, 1))
	product := builder.MulMat(weights, input)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	inputValue, _ := reference.NewValue(input.Shape, []float32{
		1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1,
	})
	results, err := cuda.ExecuteWithDeviceFeeds(
		context.Background(),
		[]*tensor.Tensor{rows, product},
		map[*tensor.Tensor]reference.Value{input: inputValue},
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
	compare(t, results[product].Data, []float32{-8, 32}, 1e-5)
}

func TestExecutorQ6KDeviceEmbeddingAndMulMat(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
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
	var pointer driver.DevicePtr
	err = worker.Do(context.Background(), func(state *device.State) error {
		var allocateErr error
		pointer, allocateErr = state.Driver.MemAlloc(uint64(len(storage)))
		if allocateErr != nil {
			return allocateErr
		}
		return state.Driver.MemcpyHtoD(pointer, storage)
	})
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Do(context.Background(), func(state *device.State) error {
		return state.Driver.MemFree(pointer)
	})
	cuda, err := NewWithWorker(worker)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
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
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
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

	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	var pointer driver.DevicePtr
	err = worker.Do(context.Background(), func(state *device.State) error {
		var allocateErr error
		pointer, allocateErr = state.Driver.MemAlloc(uint64(len(storage)))
		if allocateErr != nil {
			return allocateErr
		}
		return state.Driver.MemcpyHtoD(pointer, storage)
	})
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Do(context.Background(), func(state *device.State) error {
		return state.Driver.MemFree(pointer)
	})
	cuda, err := NewWithWorker(worker)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()

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
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
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

	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	var pointer driver.DevicePtr
	err = worker.Do(context.Background(), func(state *device.State) error {
		var allocateErr error
		pointer, allocateErr = state.Driver.MemAlloc(uint64(len(storage)))
		if allocateErr != nil {
			return allocateErr
		}
		return state.Driver.MemcpyHtoD(pointer, storage)
	})
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Do(context.Background(), func(state *device.State) error {
		return state.Driver.MemFree(pointer)
	})
	cuda, err := NewWithWorker(worker)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()

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
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
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
				// Encode binary16 0.5/1.0 across the high nibbles of
				// IQ1_M's four packed scale words.
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

	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	var pointer driver.DevicePtr
	err = worker.Do(context.Background(), func(state *device.State) error {
		var allocateErr error
		pointer, allocateErr = state.Driver.MemAlloc(uint64(len(storage)))
		if allocateErr != nil {
			return allocateErr
		}
		return state.Driver.MemcpyHtoD(pointer, storage)
	})
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Do(context.Background(), func(state *device.State) error {
		return state.Driver.MemFree(pointer)
	})
	cuda, err := NewWithWorker(worker)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()

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

func patternedValue(shape tensor.Shape, seed int, scale, bias float32) reference.Value {
	elements, _ := shape.Elements()
	data := make([]float32, int(elements))
	for i := range data {
		data[i] = bias + scale*float32(((i*7+seed*3)%19)-9)
	}
	value, _ := reference.NewValue(shape, data)
	return value
}

func compare(t *testing.T, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if difference := math.Abs(float64(got[i] - want[i])); difference > tolerance {
			t.Fatalf("value[%d] = %v, want %v (difference %g)", i, got[i], want[i], difference)
		}
	}
}
