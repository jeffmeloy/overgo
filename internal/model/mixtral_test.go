package model

import (
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func TestMixtralMetadataCatalogAndGraph(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "llama"),
		metadata("llama.block_count", gguf.ValueTypeUint32, uint32(1)),
		metadata("llama.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("llama.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("llama.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("llama.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("llama.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("llama.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("llama.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("llama.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("llama.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("llama.vocab_size", gguf.ValueTypeUint32, uint32(32)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.ExpertFeedForward != 12 || spec.ExpertCount != 4 || spec.ExpertUsedCount != 2 {
		t.Fatalf("unexpected Mixtral spec: %+v", spec)
	}
	file.Tensors = []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4), tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.ffn_gate_inp.weight", 8, 4), tensorInfo("blk.0.ffn_gate_exps.weight", 8, 12, 4),
		tensorInfo("blk.0.ffn_up_exps.weight", 8, 12, 4), tensorInfo("blk.0.ffn_down_exps.weight", 12, 8, 4),
	}
	weights, err := readFixtureWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.Layers[0].FeedForwardRouter == nil {
		t.Fatal("Mixtral expert catalog was not loaded")
	}
	b := tensor.NewBuilder()
	input := b.Input("input", dtype.F32, tensor.MustShape(8, 2))
	graphWeights := denseBlockInputs(b, spec)
	graphWeights.FeedForwardGate = nil
	graphWeights.FeedForwardUp = nil
	graphWeights.FeedForwardDown = nil
	graphWeights.FeedForwardRouter = b.Input("router", dtype.F32, tensor.MustShape(8, 4))
	graphWeights.FeedForwardGateExperts = b.Input("gate_exps", dtype.F32, tensor.MustShape(8, 12, 4))
	graphWeights.FeedForwardUpExperts = b.Input("up_exps", dtype.F32, tensor.MustShape(8, 12, 4))
	graphWeights.FeedForwardDownExperts = b.Input("down_exps", dtype.F32, tensor.MustShape(12, 8, 4))
	output, err := buildFixtureDenseBlock(b, input, spec, graphWeights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, _ := tensor.Topological(output)
	for _, node := range nodes {
		if node.Op == tensor.OpMoE && node.Attrs.(tensor.MoEAttributes).NormalizeTopKProb {
			return
		}
	}
	t.Fatal("Mixtral graph lacks normalized routed experts")
}

func TestMixtralRejectsInvalidExpertMetadata(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "llama"),
		metadata("llama.block_count", gguf.ValueTypeUint32, uint32(1)),
		metadata("llama.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("llama.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("llama.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("llama.expert_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("llama.expert_used_count", gguf.ValueTypeUint32, uint32(3)),
		metadata("llama.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("llama.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("llama.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("llama.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("llama.vocab_size", gguf.ValueTypeUint32, uint32(32)),
	}}
	if _, err := ReadSpec(file); err == nil {
		t.Fatal("invalid Mixtral expert metadata was accepted")
	}
}
