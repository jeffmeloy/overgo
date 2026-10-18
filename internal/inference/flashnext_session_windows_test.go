//go:build windows

package inference

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/gguf"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// The Flash-Next fixture is the 3:1 gated-delta to attention cadence at hermetic scale.
//   - layer 1 carries the lexical injection and layer 3 the block-sparse attention, as the full model does
//   - the indexer budget of six tokens in blocks of three makes a dozen decode steps cross it and two capacity pages
const (
	flashNextContext     = uint32(64)
	flashNextVocab       = uint64(32)
	flashNextEmbedding   = uint64(8)
	flashNextStreams     = uint64(4)
	flashNextRank        = uint64(3)
	flashNextHeads       = uint64(2)
	flashNextHeadWidth   = uint64(4)
	flashNextConv        = uint64(4)
	flashNextInner       = uint64(8)
	flashNextState       = uint64(4)
	flashNextGroups      = uint64(2)
	flashNextTimeRank    = uint64(2)
	flashNextExperts     = uint64(4)
	flashNextUsed        = uint32(2)
	flashNextExpertWidth = uint64(6)
	flashNextShared      = uint64(4)
	flashNextIndexHeads  = uint64(2)
	flashNextIndexWidth  = uint64(4)
	flashNextBudget      = uint32(12)
	flashNextBlock       = int32(3)
	flashNextNGram       = uint32(3)
	flashNextNGramHeads  = uint32(2)
	flashNextRowWidth    = uint64(2)
	flashNextLexicalEOS  = uint32(31)
	flashNextLexicalRows = uint64(102)
	flashNextLexical     = int32(1)
	flashNextAttention   = 3
	flashNextBlocks      = 4
	flashNextRopeBase    = float32(10_000)
	flashNextEpsilon     = float32(1e-6)
	// flashNextAttentionGain lifts the attention layer's values and output projection over the other layers' contribution.
	flashNextAttentionGain = float32(30)
)

// writeHermeticFlashNextGGUF emits a four-layer qwen4exp model with routed experts, hyper-connection streams, the lexical table and block-sparse attention.
func writeHermeticFlashNextGGUF(t *testing.T) string {
	t.Helper()
	architecture := "qwen4exp."
	u32 := func(key string, value uint32) gguf.Metadata {
		return testutil.GGUFScalar(architecture+key, gguf.ValueTypeUint32, value)
	}
	tokens := make([]string, flashNextVocab)
	types := make([]int32, flashNextVocab)
	for index := range tokens {
		tokens[index], types[index] = fmt.Sprintf("t%d", index), 1
	}
	tokens[0], tokens[1], tokens[2] = "<unk>", "<s>", "</s>"
	types[0], types[1], types[2] = 2, 3, 3
	ratios := make([]int32, flashNextBlocks)
	ratios[flashNextAttention] = flashNextBlock
	metadata := []gguf.Metadata{
		testutil.GGUFScalar("general.architecture", gguf.ValueTypeString, "qwen4exp"),
		testutil.GGUFScalar("general.name", gguf.ValueTypeString, "hermetic-flash-next"),
		u32("block_count", flashNextBlocks),
		u32("context_length", flashNextContext),
		u32("embedding_length", uint32(flashNextEmbedding)),
		u32("vocab_size", uint32(flashNextVocab)),
		u32("attention.head_count", uint32(flashNextHeads)),
		u32("attention.head_count_kv", 1),
		u32("attention.key_length", uint32(flashNextHeadWidth)),
		u32("attention.value_length", uint32(flashNextHeadWidth)),
		testutil.GGUFScalar(architecture+"rope.freq_base", gguf.ValueTypeFloat32, flashNextRopeBase),
		u32("rope.dimension_count", uint32(flashNextHeadWidth)),
		testutil.GGUFArray(architecture+"rope.dimension_sections", gguf.ValueTypeInt32, []int32{1, 1, 0, 0}),
		testutil.GGUFScalar(architecture+"attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, flashNextEpsilon),
		u32("ssm.conv_kernel", uint32(flashNextConv)),
		u32("ssm.inner_size", uint32(flashNextInner)),
		u32("ssm.state_size", uint32(flashNextState)),
		u32("ssm.time_step_rank", uint32(flashNextTimeRank)),
		u32("ssm.group_count", uint32(flashNextGroups)),
		u32("full_attention_interval", flashNextBlocks),
		u32("expert_count", uint32(flashNextExperts)),
		u32("expert_used_count", flashNextUsed),
		u32("expert_feed_forward_length", uint32(flashNextExpertWidth)),
		u32("expert_shared_feed_forward_length", uint32(flashNextShared)),
		u32("hyper_connection.count", uint32(flashNextStreams)),
		u32("hyper_connection.low_rank", uint32(flashNextRank)),
		u32("attention.indexer.head_count", uint32(flashNextIndexHeads)),
		u32("attention.indexer.key_length", uint32(flashNextIndexWidth)),
		u32("attention.indexer.top_k", flashNextBudget),
		testutil.GGUFArray(architecture+"attention.compress_ratios", gguf.ValueTypeInt32, ratios),
		testutil.GGUFArray(architecture+"ple.layers", gguf.ValueTypeInt32, []int32{flashNextLexical}),
		u32("ple.ngram_size", flashNextNGram),
		u32("ple.heads_per_ngram", flashNextNGramHeads),
		u32("ple.conv_kernel", uint32(flashNextConv)),
		u32("ple.eos_token_id", flashNextLexicalEOS),
		u32("embedding_length_per_layer_input", uint32(flashNextRowWidth)),
		testutil.GGUFArray(architecture+"ple.layer_multipliers", gguf.ValueTypeUint64, []uint64{3, 5, 7}),
		testutil.GGUFArray(architecture+"ple.head_offsets", gguf.ValueTypeUint64, []uint64{0, 19, 42, 71}),
		testutil.GGUFArray(architecture+"ple.head_vocab_sizes", gguf.ValueTypeUint64, []uint64{19, 23, 29, 31}),
		testutil.GGUFScalar("tokenizer.ggml.model", gguf.ValueTypeString, "llama"),
		testutil.GGUFArray("tokenizer.ggml.tokens", gguf.ValueTypeString, tokens),
		testutil.GGUFArray("tokenizer.ggml.scores", gguf.ValueTypeFloat32, make([]float32, flashNextVocab)),
		testutil.GGUFArray("tokenizer.ggml.token_type", gguf.ValueTypeInt32, types),
		testutil.GGUFScalar("tokenizer.ggml.bos_token_id", gguf.ValueTypeUint32, uint32(1)),
		testutil.GGUFScalar("tokenizer.ggml.eos_token_id", gguf.ValueTypeUint32, uint32(2)),
	}
	wide, keyWidth := flashNextEmbedding*flashNextStreams, flashNextState*flashNextGroups
	var tensors []gguf.TensorData
	seed := 1
	tensor := func(name string, shape ...uint64) {
		tensors = append(tensors, testutil.GGUFTensorF32(name, shape, seed))
		seed++
	}
	// The attention layer's values and output projection are scaled up, so its block selection decides the selected token.
	amplified := func(name string, shape ...uint64) {
		tensor(name, shape...)
		data := &tensors[len(tensors)-1]
		values := make([]float32, data.Data.(*bytes.Reader).Len()/binary.Size(float32(0)))
		if err := binary.Read(data.Data, binary.LittleEndian, values); err != nil {
			t.Fatal(err)
		}
		for index := range values {
			values[index] *= flashNextAttentionGain
		}
		var scaled bytes.Buffer
		if err := binary.Write(&scaled, binary.LittleEndian, values); err != nil {
			t.Fatal(err)
		}
		data.Data = bytes.NewReader(scaled.Bytes())
	}
	tensor("token_embd.weight", flashNextEmbedding, flashNextVocab)
	tensor("output.weight", flashNextEmbedding, flashNextVocab)
	tensor("output_hc_norm.weight", wide)
	tensor("output_hc_down.weight", wide, flashNextRank)
	tensor("output_hc_up.weight", flashNextRank, wide)
	tensor("per_layer_token_embd.weight", flashNextRowWidth, flashNextLexicalRows)
	for layer := range flashNextBlocks {
		block := fmt.Sprintf("blk.%d.", layer)
		if layer == flashNextAttention {
			tensor(block+"attn_q.weight", flashNextEmbedding, flashNextHeads*flashNextHeadWidth*2)
			tensor(block+"attn_k.weight", flashNextEmbedding, flashNextHeadWidth)
			amplified(block+"attn_v.weight", flashNextEmbedding, flashNextHeadWidth)
			amplified(block+"attn_output.weight", flashNextHeads*flashNextHeadWidth, flashNextEmbedding)
			tensor(block+"attn_q_norm.weight", flashNextHeadWidth)
			tensor(block+"attn_k_norm.weight", flashNextHeadWidth)
			tensor(block+"indexer.q_proj.weight", flashNextEmbedding, flashNextIndexHeads*flashNextIndexWidth)
			tensor(block+"indexer.k_proj.weight", flashNextEmbedding, flashNextIndexWidth)
			tensor(block+"indexer.q_norm.weight", flashNextIndexWidth)
			tensor(block+"indexer.k_norm.weight", flashNextIndexWidth)
		} else {
			tensor(block+"attn_qkv.weight", flashNextEmbedding, 2*keyWidth+flashNextInner)
			tensor(block+"attn_gate.weight", flashNextEmbedding, flashNextInner)
			tensor(block+"ssm_beta.weight", flashNextEmbedding, flashNextTimeRank)
			tensor(block+"ssm_alpha.weight", flashNextEmbedding, flashNextTimeRank)
			tensor(block+"ssm_conv1d.weight", flashNextConv, 2*keyWidth+flashNextInner)
			tensor(block+"ssm_dt.bias", flashNextTimeRank)
			tensor(block+"ssm_a", flashNextTimeRank)
			tensor(block+"ssm_norm.weight", flashNextState)
			tensor(block+"ssm_out.weight", flashNextInner, flashNextEmbedding)
		}
		if layer == int(flashNextLexical) {
			tensor(block+"ple_key.weight", flashNextEmbedding, wide)
			tensor(block+"ple_value.weight", flashNextEmbedding, flashNextEmbedding)
			tensor(block+"ple_norm_key.weight", wide)
			tensor(block+"ple_norm_query.weight", wide)
			tensor(block+"ple_norm_conv.weight", wide)
			tensor(block+"ple_conv1d.weight", flashNextConv, wide)
		}
		for _, stage := range []string{"attn", "ffn"} {
			tensor(block+"hc_"+stage+"_norm.weight", wide)
			tensor(block+"hc_"+stage+"_down.weight", wide, flashNextRank)
			tensor(block+"hc_"+stage+"_up.weight", flashNextRank, wide)
			tensor(block+"hc_"+stage+"_inject.weight", wide, flashNextStreams)
		}
		tensor(block+"ffn_gate_inp.weight", flashNextEmbedding, flashNextExperts)
		tensor(block+"ffn_gate_exps.weight", flashNextEmbedding, flashNextExpertWidth, flashNextExperts)
		tensor(block+"ffn_up_exps.weight", flashNextEmbedding, flashNextExpertWidth, flashNextExperts)
		tensor(block+"ffn_down_exps.weight", flashNextExpertWidth, flashNextEmbedding, flashNextExperts)
		tensor(block+"ffn_gate_inp_shexp.weight", flashNextEmbedding)
		tensor(block+"ffn_gate_shexp.weight", flashNextEmbedding, flashNextShared)
		tensor(block+"ffn_up_shexp.weight", flashNextEmbedding, flashNextShared)
		tensor(block+"ffn_down_shexp.weight", flashNextShared, flashNextEmbedding)
	}
	path := filepath.Join(t.TempDir(), "hermetic-flash-next.gguf")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err = gguf.Write(file, metadata, tensors, gguf.WriteOptions{}); err != nil {
		_ = file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestGreedySessionReplaysAcrossCapacityClasses holds a replayed Flash-Next decode step to the tokens a freshly compiled step selects.
//   - the steps cross the indexer budget and the 4-, 8- and 16-token pages, so the lexical rows, the block selection and every capacity class change under replay
//   - the replaying generation compiles nothing, and its steps onto a new page bind pasts another session made
func TestGreedySessionReplaysAcrossCapacityClasses(t *testing.T) {
	// Serial: the generations share one runner and its kept sessions on the one CUDA device.
	requireIntegration(t)
	cudatest.Require(t)
	const steps = 14
	runner, err := openF32FixtureRunner(writeHermeticFlashNextGGUF(t), OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	ctx := t.Context()
	runner.mu.Lock()
	defer runner.mu.Unlock()
	prompt := []tokenizer.TokenID{4, 9, 17}
	type generation struct {
		selected []tokenizer.TokenID
		logits   [][]float32
		sessions []*deviceDecodeSession
		cache    *KVCache
	}
	// generate decodes the steps after the prompt, each step compiled afresh or served by the kept sessions.
	//   - forced = nil => each step feeds the token the last one selected, else forced[step]
	generate := func(mode deviceOutputMode, fresh bool, forced []tokenizer.TokenID) generation {
		t.Helper()
		past, err := runner.forwardDeviceCachedModeLocked(ctx, mode, prompt, nil)
		if err != nil {
			t.Fatal(err)
		}
		result := generation{selected: []tokenizer.TokenID{past.Selected}}
		for step := range steps {
			if fresh {
				past.session, runner.decodeSessions = nil, nil
			}
			token := past.Selected
			if forced != nil {
				token = forced[step]
			}
			next, err := runner.forwardDeviceCachedModeLocked(ctx, mode, []tokenizer.TokenID{token}, past)
			if releaseErr := past.Release(ctx); err != nil || releaseErr != nil {
				t.Fatalf("decode step %d: %v %v", step, err, releaseErr)
			}
			past = next
			result.selected, result.sessions = append(result.selected, past.Selected), append(result.sessions, past.session)
			result.logits = append(result.logits, past.Logits)
		}
		if result.cache, err = runner.downloadDeviceCache(ctx, past); err != nil {
			t.Fatal(err)
		}
		if err := past.Release(ctx); err != nil {
			t.Fatal(err)
		}
		return result
	}
	// replay generates afresh, then twice from the kept sessions, and holds the second, which compiles nothing, to the fresh one.
	replay := func(mode deviceOutputMode, forced []tokenizer.TokenID) generation {
		t.Helper()
		compiled := generate(mode, true, forced)
		generate(mode, false, forced)
		kept := make(map[*deviceDecodeSession]uint64, len(runner.decodeSessions))
		for _, session := range runner.decodeSessions {
			kept[session] = session.replays
		}
		replayed := generate(mode, false, forced)
		if !slices.Equal(replayed.selected, compiled.selected) {
			t.Fatalf("mode %d selected tokens fresh/replayed = %v/%v", mode, compiled.selected, replayed.selected)
		}
		if !reflect.DeepEqual(replayed.logits, compiled.logits) || !reflect.DeepEqual(replayed.cache, compiled.cache) {
			t.Fatalf("mode %d replayed logits or cache differ from the fresh steps", mode)
		}
		var replays uint64
		for step, session := range replayed.sessions {
			if _, ok := kept[session]; !ok {
				t.Fatalf("mode %d replaying step %d compiled a session", mode, step)
			}
		}
		for session, before := range kept {
			replays += session.replays - before
		}
		if replays != steps {
			t.Fatalf("mode %d replaying generation replayed %d of its %d steps", mode, replays, steps)
		}
		return compiled
	}
	greedy := replay(deviceOutputGreedy, nil)
	if len(slices.Compact(slices.Clone(greedy.selected))) == 1 {
		t.Fatalf("the fixture selects one token throughout (%v), so the replay is not exercised", greedy.selected)
	}
	// The logits hold the block selection's effect bit for bit, which a selected token can hide.
	replay(deviceOutputLogits, greedy.selected)
}
