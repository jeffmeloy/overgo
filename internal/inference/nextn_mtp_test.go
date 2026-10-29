package inference

import (
	"slices"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/model"
)

func TestSelectedModelTensorsIncludesGenericNextN(t *testing.T) {
	info := func(name string) gguf.TensorInfo { return gguf.TensorInfo{Name: name} }
	outputNorm := info("blk.1.nextn.shared_head_norm.weight")
	layerOutputNorm := info("blk.1.layer_output_norm.weight")
	mtp := model.AppendedDraftWeights{
		Layer: model.LayerWeights{
			AttentionNorm: new(info("blk.1.attn_norm.weight")),
			AttentionQKV:  new(info("blk.1.attn_qkv.weight")),
		},
		EHProjection:    info("blk.1.nextn.eh_proj.weight"),
		EmbeddingNorm:   info("blk.1.nextn.enorm.weight"),
		HiddenNorm:      info("blk.1.nextn.hnorm.weight"),
		LayerOutputNorm: &layerOutputNorm,
		OutputNorm:      &outputNorm,
	}
	weights := model.Weights{TokenEmbedding: info("token_embd.weight"), AppendedSingleDraft: []model.AppendedDraftWeights{mtp}}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		weights.TokenEmbedding, *mtp.Layer.AttentionNorm, *mtp.Layer.AttentionQKV,
		mtp.EHProjection, mtp.EmbeddingNorm, mtp.HiddenNorm, layerOutputNorm, outputNorm,
	}}
	selected := selectedModelTensors(file, weights)
	names := make(map[string]bool, len(selected))
	for _, item := range selected {
		names[item.Name] = true
	}
	for _, item := range file.Tensors {
		if !names[item.Name] {
			t.Fatalf("NextN tensor %q was not selected", item.Name)
		}
	}
}

func TestValidateGLM4NextNAvailability(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "glm4", NextNPredictLayers: 1}},
		weights: model.Weights{AppendedSingleDraft: []model.AppendedDraftWeights{{}}}},
	}
	runner = attachFixtureProgram(runner)
	if err := runner.validateNextNMTP(); err != nil {
		t.Fatal(err)
	}
	runner.weights.AppendedSingleDraft = nil
	if err := runner.validateNextNMTP(); err == nil {
		t.Fatal("GLM4 without loaded NextN block was accepted")
	}
}

// TestOwnDraftsRefusedWhereFewAreKept: the model's own next-token layer stops drafting by default once the share of its
// first ownDraftJudgedDrafts its trunk kept falls under ownDraftPaysShare, and only then; the judgement is made once.
func TestOwnDraftsRefusedWhereFewAreKept(t *testing.T) {
	t.Parallel()
	const depth = 3
	for _, test := range []struct {
		name    string
		kept    []int
		refused bool
	}{
		{"a layer keeping a third of its drafts, as Qwen3.8-27B's keeps on prose", []int{1}, false},
		{"a layer keeping none, as Qwen3.5-4B's unshifted norms keep", []int{0}, true},
		{"one draft kept in sixteen windows", append([]int{1}, make([]int, ownDraftJudgedDrafts/depth)...), true},
		{"a layer judged before its drafts were refused", append(slices.Repeat([]int{1}, ownDraftJudgedDrafts/depth), make([]int, ownDraftJudgedDrafts)...), false},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			runner := &Runner{}
			for window := range 2 * ownDraftJudgedDrafts / depth {
				runner.observeOwnDrafts(depth, test.kept[min(window, len(test.kept)-1)])
			}
			if runner.ownDraftsRefused.Load() != test.refused {
				t.Fatalf("refused %v after %d of %d drafts kept, want %v", runner.ownDraftsRefused.Load(), runner.ownKept, runner.ownDrafted, test.refused)
			}
			if test.refused && runner.ownDraftLayer() {
				t.Fatal("a refused layer still drafts by default")
			}
		})
	}
}
