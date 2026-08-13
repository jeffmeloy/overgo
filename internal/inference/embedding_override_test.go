package inference

import (
	"context"
	"math"
	"slices"
	"strings"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
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

func TestCogVLMMixedVisualExpertBlockAdmission(t *testing.T) {
	overrides := []EmbeddingOverride{
		{TokenIndex: 1, Embedding: []float32{1}},
		{TokenIndex: 2, Embedding: []float32{2}},
		{TokenIndex: 4, Embedding: []float32{3}},
	}
	blocks, err := validateVisualExpertBlocks(6, []AttentionBlock{{Start: 4, End: 5}, {Start: 1, End: 3}}, overrides)
	if err != nil {
		t.Fatal(err)
	}
	if len(blocks) != 2 || blocks[0] != (AttentionBlock{Start: 1, End: 3}) {
		t.Fatalf("ordered blocks = %v", blocks)
	}
	for _, test := range []struct {
		name      string
		blocks    []AttentionBlock
		overrides []EmbeddingOverride
	}{
		{name: "overlap", blocks: []AttentionBlock{{Start: 1, End: 3}, {Start: 2, End: 4}}, overrides: overrides},
		{name: "missing", blocks: []AttentionBlock{{Start: 1, End: 3}}, overrides: overrides[:1]},
		{name: "outside", blocks: []AttentionBlock{{Start: 1, End: 3}}, overrides: []EmbeddingOverride{{TokenIndex: 0, Embedding: []float32{1}}, {TokenIndex: 1, Embedding: []float32{2}}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := validateVisualExpertBlocks(6, test.blocks, test.overrides); err == nil {
				t.Fatal("expected validation error")
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

func TestApplyGemmaRawEmbeddingOverridesScalesOnlyTokens(t *testing.T) {
	activation := reference.Value{
		Shape: tensor.MustShape(2, 2),
		Data:  []float32{1, 2, 3, 4},
	}
	err := applyScaledRawEmbeddingOverrides(&activation, []EmbeddingOverride{
		{TokenIndex: 1, Embedding: []float32{5, 6}},
	}, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{2, 4, 5, 6}
	for index, value := range want {
		if activation.Data[index] != value {
			t.Fatalf("activation[%d] = %g, want %g", index, activation.Data[index], value)
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
		supported := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: architecture}, AttentionSpec: model.AttentionSpec{RopeSections: [4]int32{2, 2, 0, 0}}}}}
		supported = attachFixtureProgram(supported)
		_, _, err := supported.ForwardCachedWithMultimodalInputs(
			context.Background(), nil, nil, MultiAxisPositions{}, nil,
		)
		if err == nil || !strings.Contains(err.Error(), "token sequence is empty") {
			t.Fatalf("%s multimodal error = %v", architecture, err)
		}
	}

	unsupported := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "llama"}}}}
	unsupported = attachFixtureProgram(unsupported)
	_, _, err := unsupported.ForwardCachedWithMultimodalInputs(
		context.Background(), nil, nil, MultiAxisPositions{}, nil,
	)
	if err == nil || !strings.Contains(err.Error(), "does not support") {
		t.Fatalf("unsupported multimodal error = %v", err)
	}
}

func TestMultimodalInputRejectsIncompleteAxes(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "qwen3vl", ContextLength: 4}, AttentionSpec: model.AttentionSpec{RopeSections: [4]int32{1, 1, 0, 0}}}}}
	runner = attachFixtureProgram(runner)
	positions := MultiAxisPositions{{0}, {0}, nil, {0}}
	_, _, err := runner.ForwardCachedWithMultimodalInputs(
		context.Background(), []tokenizer.TokenID{0}, nil, positions, nil,
	)
	if err == nil || !strings.Contains(err.Error(), "position 2") {
		t.Fatalf("incomplete multi-axis error = %v", err)
	}
}

func TestProjectedInputDeepstackAdmission(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "granite", EmbeddingLength: 2}, MultimodalSpec: model.MultimodalSpec{DeepstackLayerCount: 1,
		DeepstackMapping: []int32{0, 1}},
	}}}
	runner = attachFixtureProgram(runner)
	_, _, err := runner.ForwardCachedWithProjectedInputs(
		context.Background(), nil, nil,
		ProjectedInputs{DeepstackEmbeddings: []reference.Value{{}}},
	)
	if err == nil || !strings.Contains(err.Error(), "token sequence is empty") {
		t.Fatalf("Granite projected input error = %v", err)
	}
}

func TestProjectedAttentionBlockIDs(t *testing.T) {
	blocks := []AttentionBlock{{Start: 1, End: 3}, {Start: 4, End: 5}}
	ids, err := projectedAttentionBlockIDs(
		model.AttentionBlocksUncached, 6, false, blocks,
	)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{-1, 0, 0, -1, 1, -1}
	if !slices.Equal(ids, want) {
		t.Fatalf("attention block IDs = %v, want %v", ids, want)
	}
	if _, err = projectedAttentionBlockIDs(
		model.AttentionBlocksNone, 6, false, blocks,
	); err == nil || !strings.Contains(err.Error(), "does not support") {
		t.Fatalf("unsupported block error = %v", err)
	}
	if _, err = projectedAttentionBlockIDs(
		model.AttentionBlocksUncached, 6, true, blocks,
	); err == nil || !strings.Contains(err.Error(), "uncached") {
		t.Fatalf("cached block error = %v", err)
	}
	if _, err = projectedAttentionBlockIDs(
		model.AttentionBlocksUncached, 6, false,
		[]AttentionBlock{{Start: 1, End: 4}, {Start: 3, End: 5}},
	); err == nil || !strings.Contains(err.Error(), "overlaps") {
		t.Fatalf("overlap error = %v", err)
	}
}

func TestCompileProjectedRequestPlan(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{
		CommonSpec: model.CommonSpec{Architecture: "gemma4", ContextLength: 2, EmbeddingLength: 4},
	}}}
	runner = attachFixtureProgram(runner)
	plan, err := runner.compileProjectedRequestPlan(2, false, ProjectedInputs{
		BidirectionalAttentionBlocks: []AttentionBlock{{Start: 0, End: 2}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.overridePolicy != model.EmbeddingOverrideRawScaled || plan.multiPositions != nil ||
		!slices.Equal(plan.attentionBlockIDs, []float32{0, 0}) || plan.visualMode {
		t.Fatalf("projected request plan = %+v", plan)
	}
}

func TestProjectedInputsSignatureIncludesMediaPayload(t *testing.T) {
	first := ProjectedInputs{EmbeddingOverrides: []EmbeddingOverride{{
		TokenIndex: 2, Embedding: []float32{1, 2},
	}}}
	second := ProjectedInputs{EmbeddingOverrides: []EmbeddingOverride{{
		TokenIndex: 2, Embedding: []float32{1, 3},
	}}}
	if projectedInputsSignature(first) == projectedInputsSignature(second) {
		t.Fatal("distinct projected media produced one cache signature")
	}
}

func TestDeepstackLayerMapping(t *testing.T) {
	base := reference.Value{Shape: tensor.MustShape(1, 1), Data: []float32{10}}
	streams := []reference.Value{
		{Shape: base.Shape, Data: []float32{20}},
		{Shape: base.Shape, Data: []float32{30}},
	}
	granite := model.Spec{CommonSpec: model.CommonSpec{Architecture: "granite"}, MultimodalSpec: model.MultimodalSpec{DeepstackLayerCount: 2,
		DeepstackMapping: []int32{0, 2, -1, 1}},
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
		got := deepstackInputForLayer(granite.PlanLayer(test.layer, false).DeepstackBefore, base, streams)
		if (got != nil) != test.ok || got != nil && got.Data[0] != test.want {
			t.Fatalf("Granite layer %d stream = %v, want %v/%v", test.layer, got, test.want, test.ok)
		}
	}
	for _, architecture := range []string{"qwen3vl", "qwen3vlmoe"} {
		qwen := model.Spec{CommonSpec: model.CommonSpec{Architecture: architecture}, MultimodalSpec: model.MultimodalSpec{DeepstackLayerCount: 2}}
		for layer, want := range []float32{20, 30} {
			got := deepstackInputForLayer(qwen.PlanLayer(uint32(layer), false).DeepstackAfter, base, streams)
			if got == nil || got.Data[0] != want {
				t.Fatalf("%s layer %d stream = %v, want %v", architecture, layer, got, want)
			}
		}
		if got := deepstackInputForLayer(qwen.PlanLayer(0, false).DeepstackBefore, base, streams); got != nil {
			t.Fatalf("%s pre-layer stream = %v", architecture, got)
		}
	}
	if got := deepstackInputForLayer(model.DeepstackSource(-3), base, streams); got != nil {
		t.Fatalf("invalid deepstack source = %v", got)
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
