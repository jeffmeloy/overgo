package inference

import (
	"context"
	"math"
	"strings"
	"testing"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

func TestCogVLMVisualEmbeddingAdmission(t *testing.T) {
	complete := []EmbeddingOverride{
		{TokenIndex: 1, Embedding: []float32{1}},
		{TokenIndex: 0, Embedding: []float32{2}},
	}
	if err := validateCogVLMVisualOverrides(2, complete); err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name      string
		overrides []EmbeddingOverride
		contains  string
	}{
		{"partial", complete[:1], "one projected embedding per token"},
		{"duplicate", []EmbeddingOverride{complete[0], complete[0]}, "duplicate"},
		{"range", []EmbeddingOverride{complete[0], {TokenIndex: 2}}, "out of range"},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := validateCogVLMVisualOverrides(2, test.overrides)
			if err == nil || !strings.Contains(err.Error(), test.contains) {
				t.Fatalf("error = %v, want containing %q", err, test.contains)
			}
		})
	}
}

func TestCogVLMVisualGraphWeightSelection(t *testing.T) {
	builder := tensor.NewBuilder()
	normal := builder.Input("text_qkv", dtype.F32, tensor.MustShape(2, 6))
	weights := model.LayerGraphWeights{
		AttentionQKV: normal, AttentionQ: normal, AttentionK: normal, AttentionV: normal,
		AttentionOutput: normal, FeedForwardGate: normal,
		FeedForwardUp: normal, FeedForwardDown: normal,
	}
	visual := []*tensor.Tensor{
		builder.Input("visual_qkv", dtype.F32, tensor.MustShape(2, 6)),
		builder.Input("visual_output", dtype.F32, tensor.MustShape(2, 2)),
		builder.Input("visual_gate", dtype.F32, tensor.MustShape(2, 3)),
		builder.Input("visual_up", dtype.F32, tensor.MustShape(2, 3)),
		builder.Input("visual_down", dtype.F32, tensor.MustShape(3, 2)),
	}
	if err := selectCogVLMVisualGraphWeights(&weights, visual); err != nil {
		t.Fatal(err)
	}
	if weights.AttentionQ != nil || weights.AttentionK != nil || weights.AttentionV != nil ||
		weights.AttentionQKV != visual[0] || weights.AttentionOutput != visual[1] ||
		weights.FeedForwardGate != visual[2] || weights.FeedForwardUp != visual[3] ||
		weights.FeedForwardDown != visual[4] {
		t.Fatalf("visual graph weights were not selected: %+v", weights)
	}
}

func TestSelectedModelTensorsIncludesCogVLMVisualBank(t *testing.T) {
	names := []string{
		"token_embd.weight", "blk.0.vis_attn_qkv.weight", "blk.0.vis_attn_output.weight",
		"blk.0.vis_gate.weight", "blk.0.vis_up.weight", "blk.0.vis_down.weight",
	}
	infos := make([]gguf.TensorInfo, len(names))
	for index, name := range names {
		infos[index] = gguf.TensorInfo{Name: name}
	}
	visual := func(index int) *gguf.TensorInfo { return &infos[index] }
	selected := selectedModelTensors(&gguf.File{Tensors: infos}, model.Weights{
		TokenEmbedding: infos[0],
		Layers: []model.LayerWeights{{
			VisualAttentionQKV: visual(1), VisualAttentionOutput: visual(2),
			VisualFeedForwardGate: visual(3), VisualFeedForwardUp: visual(4),
			VisualFeedForwardDown: visual(5),
		}},
	})
	got := make(map[string]bool, len(selected))
	for _, info := range selected {
		got[info.Name] = true
	}
	for _, name := range names {
		if !got[name] {
			t.Fatalf("selected tensors omit %q", name)
		}
	}
}

func TestF32RequiredModelTensorsIncludesConvolutionKernels(t *testing.T) {
	infos := []gguf.TensorInfo{
		{Name: "ssm"}, {Name: "q"}, {Name: "k"}, {Name: "v"}, {Name: "short"},
	}
	pointer := func(index int) *gguf.TensorInfo { return &infos[index] }
	names := f32RequiredModelTensors(model.Weights{Layers: []model.LayerWeights{{
		SSMConv1D: pointer(0), SSMQueryConv: pointer(1), SSMKeyConv: pointer(2),
		SSMValueConv: pointer(3), ShortConvKernel: pointer(4),
	}}})
	for _, info := range infos {
		if _, ok := names[info.Name]; !ok {
			t.Fatalf("F32-required tensors omit %q", info.Name)
		}
	}
}

func TestApplyEmbeddingOverrides(t *testing.T) {
	activation := reference.Value{
		Shape: tensor.MustShape(3, 3),
		Data:  []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
	}
	err := applyEmbeddingOverrides(&activation, []EmbeddingOverride{
		{TokenIndex: 1, Embedding: []float32{10, 11, 12}},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 2, 3, 10, 11, 12, 7, 8, 9}
	for index := range want {
		if activation.Data[index] != want[index] {
			t.Fatalf("embedding[%d] = %v, want %v", index, activation.Data[index], want[index])
		}
	}
}

func TestApplyEmbeddingOverridesRejectsInvalidInput(t *testing.T) {
	tests := []struct {
		name      string
		overrides []EmbeddingOverride
		contains  string
	}{
		{"index", []EmbeddingOverride{{TokenIndex: 2, Embedding: []float32{1, 2}}}, "out of range"},
		{"width", []EmbeddingOverride{{TokenIndex: 0, Embedding: []float32{1}}}, "width"},
		{"duplicate", []EmbeddingOverride{
			{TokenIndex: 0, Embedding: []float32{1, 2}},
			{TokenIndex: 0, Embedding: []float32{3, 4}},
		}, "duplicate"},
		{"non-finite", []EmbeddingOverride{{TokenIndex: 0, Embedding: []float32{1, float32(math.NaN())}}}, "non-finite"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			activation := reference.Value{Shape: tensor.MustShape(2, 2), Data: make([]float32, 4)}
			err := applyEmbeddingOverrides(&activation, test.overrides)
			if err == nil || !strings.Contains(err.Error(), test.contains) {
				t.Fatalf("error = %v, want containing %q", err, test.contains)
			}
		})
	}
}

func TestMultimodalInputAdmission(t *testing.T) {
	for _, architecture := range []string{"qwen3vl", "glm4", "hunyuan-dense"} {
		supported := &Runner{spec: model.Spec{
			Architecture: architecture, RopeSections: [4]int32{2, 2, 0, 0},
		}}
		_, _, err := supported.ForwardCachedWithMultimodalInputs(
			context.Background(), nil, nil, MultiAxisPositions{}, nil,
		)
		if err == nil || !strings.Contains(err.Error(), "token sequence is empty") {
			t.Fatalf("%s multimodal error = %v", architecture, err)
		}
	}

	unsupported := &Runner{spec: model.Spec{Architecture: "llama"}}
	_, _, err := unsupported.ForwardCachedWithMultimodalInputs(
		context.Background(), nil, nil, MultiAxisPositions{}, nil,
	)
	if err == nil || !strings.Contains(err.Error(), "does not support") {
		t.Fatalf("unsupported multimodal error = %v", err)
	}
}

func TestMultimodalInputRejectsIncompleteAxes(t *testing.T) {
	runner := &Runner{spec: model.Spec{
		Architecture: "qwen3vl", ContextLength: 4,
		RopeSections: [4]int32{1, 1, 0, 0},
	}}
	positions := MultiAxisPositions{{0}, {0}, nil, {0}}
	_, _, err := runner.ForwardCachedWithMultimodalInputs(
		context.Background(), []tokenizer.TokenID{0}, nil, positions, nil,
	)
	if err == nil || !strings.Contains(err.Error(), "position 2") {
		t.Fatalf("incomplete multi-axis error = %v", err)
	}
}

func TestProjectedInputDeepstackAdmission(t *testing.T) {
	runner := &Runner{spec: model.Spec{
		Architecture: "granite", EmbeddingLength: 2, DeepstackLayerCount: 1,
		DeepstackMapping: []int32{0, 1},
	}}
	_, _, err := runner.ForwardCachedWithProjectedInputs(
		context.Background(), nil, nil,
		ProjectedInputs{DeepstackEmbeddings: []reference.Value{{}}},
	)
	if err == nil || !strings.Contains(err.Error(), "token sequence is empty") {
		t.Fatalf("Granite projected input error = %v", err)
	}
}

func TestDeepstackLayerMapping(t *testing.T) {
	base := reference.Value{Shape: tensor.MustShape(1, 1), Data: []float32{10}}
	streams := []reference.Value{
		{Shape: base.Shape, Data: []float32{20}},
		{Shape: base.Shape, Data: []float32{30}},
	}
	granite := model.Spec{
		Architecture: "granite", DeepstackLayerCount: 2,
		DeepstackMapping: []int32{0, 2, -1, 1},
	}
	for _, test := range []struct {
		layer uint32
		want  float32
		ok    bool
	}{
		{layer: 0},
		{layer: 1, want: 30, ok: true},
		{layer: 2},
		{layer: 3, want: 20, ok: true},
	} {
		got := deepstackInputForLayer(granite, test.layer, false, base, streams)
		if (got != nil) != test.ok || got != nil && got.Data[0] != test.want {
			t.Fatalf("Granite layer %d stream = %v, want %v/%v", test.layer, got, test.want, test.ok)
		}
	}
	qwen := model.Spec{Architecture: "qwen3vl", DeepstackLayerCount: 2}
	for layer, want := range []float32{20, 30} {
		got := deepstackInputForLayer(qwen, uint32(layer), true, base, streams)
		if got == nil || got.Data[0] != want {
			t.Fatalf("Qwen layer %d stream = %v, want %v", layer, got, want)
		}
	}
	if got := deepstackInputForLayer(qwen, 0, false, base, streams); got != nil {
		t.Fatalf("Qwen pre-layer stream = %v", got)
	}
}

func TestAddDeepstackEmbedding(t *testing.T) {
	shape := tensor.MustShape(2, 2)
	got, err := addDeepstackEmbedding(
		reference.Value{Shape: shape, Data: []float32{1, 2, 3, 4}},
		reference.Value{Shape: shape, Data: []float32{10, 20, 30, 40}},
	)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{11, 22, 33, 44}
	for index := range want {
		if got.Data[index] != want[index] {
			t.Fatalf("deepstack sum[%d] = %v, want %v", index, got.Data[index], want[index])
		}
	}
}
