package model_test

import (
	"fmt"
	"os"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/testevidence"
)

// Qwen3.5-9B Q8_0 CPU structural oracle. The 9B is the same qwen35 hybrid as
// the served 4B, with one delta: an UNTIED lm_head (4B ties embeddings). This
// opens the real container READ-ONLY and asserts the read path — metadata dims,
// hybrid layer-type synthesis from full_attention_interval, tensor-inventory
// completeness, and the distinct untied head — without any forward/decode.
//
// Skips when the artifact is absent (worktree-safe). Override the location with
// OVERGO_QWEN35_9B_GGUF; otherwise the local-models.json models root is used.
const qwen35_9BDefaultPath = `C:\Users\jeffm\adaptive_new\models\Qwen3.5-9B-Q8_0.gguf`

func qwen35_9BPath(t *testing.T) string {
	t.Helper()
	path := os.Getenv("OVERGO_QWEN35_9B_GGUF")
	if path == "" {
		path = qwen35_9BDefaultPath
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("Qwen3.5-9B GGUF absent (%s); set OVERGO_QWEN35_9B_GGUF to run", path)
	}
	return path
}

func TestQwen35_9BUntiedHeadGGUFStructuralOracle(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	file, err := gguf.Open(qwen35_9BPath(t))
	if err != nil {
		t.Fatalf("open GGUF: %v", err)
	}
	defer file.Close()

	spec, err := model.ReadSpec(file)
	if err != nil {
		t.Fatalf("ReadSpec: %v", err)
	}

	// --- Metadata-derived dimensions (all synthesized from GGUF metadata; the
	// 9B ships no HF config/tokenizer sidecar). ---
	type dim struct {
		name string
		got  uint32
		want uint32
	}
	for _, d := range []dim{
		{"BlockCount", spec.BlockCount, 32},
		{"EmbeddingLength", spec.EmbeddingLength, 4096},
		{"FeedForwardLength", spec.FeedForwardLength, 12288},
		{"HeadCount", spec.HeadCount, 16},
		{"HeadCountKV", spec.HeadCountKV, 4},
		{"KeyLength", spec.KeyLength, 256},
		{"ValueLength", spec.ValueLength, 256},
		{"VocabularySize", spec.VocabularySize, 248320},
		{"FullAttentionInterval", spec.FullAttentionInterval, 4},
		{"RopeDimensionCount", spec.RopeDimensionCount, 64},
		{"SSMConvKernel", spec.SSMConvKernel, 4},
		{"SSMStateSize", spec.SSMStateSize, 128},
		{"SSMGroupCount", spec.SSMGroupCount, 16},
		{"SSMTimeStepRank", spec.SSMTimeStepRank, 32},
		{"SSMInnerSize", spec.SSMInnerSize, 4096},
	} {
		if d.got != d.want {
			t.Errorf("spec.%s = %d, want %d", d.name, d.got, d.want)
		}
	}
	if spec.Architecture != "qwen35" {
		t.Errorf("spec.Architecture = %q, want %q", spec.Architecture, "qwen35")
	}

	// --- Hybrid layer types synthesized from full_attention_interval, verified
	// against the physical tensor inventory. This Unsloth container ships no
	// qwen35.attention.recurrent_layers array, so the cadence must synthesize
	// each layer's kind from full_attention_interval (recurrent iff
	// (block+1)%interval != 0) and every synthesized kind must match the
	// tensors actually present. Both directions are asserted, so a wrong
	// synthesis OR a missing tensor family fails loudly. ---
	blk := func(block uint32, suffix string) (gguf.TensorInfo, bool) {
		return file.Tensor(fmt.Sprintf("blk.%d.%s", block, suffix))
	}
	for block := uint32(0); block < spec.BlockCount; block++ {
		recurrent := spec.IsRecurrentLayer(block)
		wantRecurrent := (block+1)%spec.FullAttentionInterval != 0
		if recurrent != wantRecurrent {
			t.Errorf("block %d: IsRecurrentLayer=%v, want %v (interval synthesis)", block, recurrent, wantRecurrent)
		}
		_, hasSSM := blk(block, "ssm_a")
		_, hasQKV := blk(block, "attn_qkv.weight")
		_, hasAttnQ := blk(block, "attn_q.weight")
		if recurrent {
			if !hasSSM || !hasQKV {
				t.Errorf("recurrent block %d: ssm_a=%v attn_qkv=%v, want both present", block, hasSSM, hasQKV)
			}
			if hasAttnQ {
				t.Errorf("recurrent block %d: attn_q.weight present but layer is linear-attention", block)
			}
		} else {
			if !hasAttnQ {
				t.Errorf("full-attention block %d: attn_q.weight missing", block)
			}
			if hasSSM {
				t.Errorf("full-attention block %d: ssm_a present but layer is full-attention", block)
			}
		}
	}

	// --- Untied head. The 9B stores a distinct output.weight; the read path
	// MUST map it as its own head and MUST NOT silently reuse token_embd as a
	// tied head. A silent tied fallback (Weights.Output == nil -> executor
	// projects through the embedding) still produces plausible text from the
	// wrong matrix, which is worse than a load error -- so assert the mapping
	// rather than trust the fallback. ---
	embd, ok := file.Tensor("token_embd.weight")
	if !ok {
		t.Fatal("token_embd.weight absent")
	}
	head, ok := file.Tensor("output.weight")
	if !ok {
		t.Fatal("output.weight absent: this container has an UNTIED head; losing it would substitute the embedding silently")
	}
	// GGML row-major shape is [embedding, vocab].
	for _, ti := range []gguf.TensorInfo{embd, head} {
		if ti.Shape[0] != uint64(spec.EmbeddingLength) || ti.Shape[1] != uint64(spec.VocabularySize) {
			t.Errorf("%s shape = [%d %d], want [%d %d]", ti.Name, ti.Shape[0], ti.Shape[1], spec.EmbeddingLength, spec.VocabularySize)
		}
	}
	if head.Offset == embd.Offset {
		t.Errorf("output.weight and token_embd.weight share offset %d: head is not distinct (tied), but the 9B is untied", head.Offset)
	}

	weights, err := model.ReadWeights(file, spec)
	if err != nil {
		t.Fatalf("ReadWeights: %v", err)
	}
	if weights.Output == nil {
		t.Fatal("Weights.Output is nil: read path fell back to the tied embedding head instead of mapping the distinct output.weight")
	}
	if weights.Output.Name != "output.weight" {
		t.Errorf("Weights.Output.Name = %q, want %q", weights.Output.Name, "output.weight")
	}
	if weights.Output.Offset == weights.TokenEmbedding.Offset {
		t.Errorf("mapped head shares the embedding offset %d: silent tied fallback", weights.Output.Offset)
	}

	// --- GGML numeric-convention wiring. The three qwen35 conventions
	// (ssm_a = -exp(A_log), V-head grouped->tiled reorder, 1+w on norm
	// weights) are import-time transforms (internal/hfgguf qwen35TensorTransforms,
	// gated to the GGML target). A native GGUF stores tensors already in GGML
	// convention, so the read path consumes them AS-IS with no re-application
	// (no double transform) -- mirroring adaptive's importedValueConventions
	// gated on TensorEncodingGGML. The CPU-verifiable proxy is that the
	// convention targets are present with the promoted F32 element width the
	// transforms produce, and that the GDN group norm (which stays raw) is
	// distinguished from the 1+w norms. ---
	var recurrentBlock uint32
	for block := uint32(0); block < spec.BlockCount; block++ {
		if spec.IsRecurrentLayer(block) {
			recurrentBlock = block
			break
		}
	}
	f32 := func(name string, want [2]uint64) {
		ti, ok := file.Tensor(name)
		if !ok {
			t.Errorf("%s absent", name)
			return
		}
		if ti.Type != gguf.DTypeF32 {
			t.Errorf("%s type = %v, want F32 (convention transform output)", name, ti.Type)
		}
		if want[0] != 0 && ti.Shape[0] != want[0] {
			t.Errorf("%s shape[0] = %d, want %d", name, ti.Shape[0], want[0])
		}
	}
	// ssm_a: -exp(A_log) target, per-head F32 (time_step_rank entries).
	f32(fmt.Sprintf("blk.%d.ssm_a", recurrentBlock), [2]uint64{uint64(spec.SSMTimeStepRank), 0})
	// ssm_conv1d: V-head reorder produces the [conv_kernel, ...] F32 layout.
	f32(fmt.Sprintf("blk.%d.ssm_conv1d.weight", recurrentBlock), [2]uint64{uint64(spec.SSMConvKernel), 0})
	// attn_norm / output_norm: 1+w convention, F32.
	f32(fmt.Sprintf("blk.%d.attn_norm.weight", recurrentBlock), [2]uint64{uint64(spec.EmbeddingLength), 0})
	f32("output_norm.weight", [2]uint64{uint64(spec.EmbeddingLength), 0})
	// GDN group norm stays raw (state_size wide), distinct from the 1+w norms.
	f32(fmt.Sprintf("blk.%d.ssm_norm.weight", recurrentBlock), [2]uint64{uint64(spec.SSMStateSize), 0})
}
