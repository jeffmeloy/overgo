//go:build windows

package inference

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/gguf"
	"overgo/internal/quant"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// writeHermeticFlashNextMTPGGUF emits the draft layer of the hermetic Flash-Next in the checkpoint's names, its norms stored without their unit offset.
//   - copyTrunk => the layer passes the trunk's streams through unchanged (no embedding, no attention or expert output) and mixes them with the trunk's own head mix, so each draft is the token the trunk selected from the same streams
func writeHermeticFlashNextMTPGGUF(t *testing.T, trunkPath string, copyTrunk bool) string {
	t.Helper()
	wide, streams, rank := flashNextEmbedding*flashNextStreams, flashNextStreams, flashNextRank
	var tensors []gguf.TensorData
	seed := 101
	values := func(name string, data []float32, shape ...uint64) {
		var storage bytes.Buffer
		if err := binary.Write(&storage, binary.LittleEndian, data); err != nil {
			t.Fatal(err)
		}
		tensors = append(tensors, gguf.TensorData{Name: name, Shape: shape, Type: gguf.DTypeF32, Data: bytes.NewReader(storage.Bytes())})
	}
	// random stores a seeded tensor, a one-dimensional one as a raw norm near zero.
	random := func(name string, shape ...uint64) {
		data := testutil.GGUFTensorF32(name, shape, seed)
		seed++
		raw := make([]float32, data.Data.(*bytes.Reader).Len()/4)
		if err := binary.Read(data.Data, binary.LittleEndian, raw); err != nil {
			t.Fatal(err)
		}
		if len(shape) == 1 {
			for index := range raw {
				raw[index]--
			}
		}
		values(name, raw, shape...)
	}
	zeros := func(name string, shape ...uint64) {
		elements := uint64(1)
		for _, extent := range shape {
			elements *= extent
		}
		values(name, make([]float32, elements), shape...)
	}
	trunk, err := gguf.Open(trunkPath)
	if err != nil {
		t.Fatal(err)
	}
	defer trunk.Close()
	// trunkTensor copies the trunk's tensor, less one for a norm.
	trunkTensor := func(name, source string, norm bool, shape ...uint64) {
		info, ok := trunk.Tensor(source)
		if !ok {
			t.Fatalf("trunk tensor %s is missing", source)
		}
		raw := make([]byte, info.Size)
		if err := trunk.ReadTensorData(info, raw); err != nil {
			t.Fatal(err)
		}
		data, err := quant.Dequantize(info.Type, raw, info.Size/4)
		if err != nil {
			t.Fatal(err)
		}
		if norm {
			for index := range data {
				data[index]--
			}
		}
		values(name, data, shape...)
	}
	p, l := "mtp.", "mtp.layers.0."
	if copyTrunk {
		zeros(p+"fc_embedding.weight", flashNextEmbedding, flashNextEmbedding)
		identity := make([]float32, flashNextEmbedding*flashNextEmbedding)
		for index := range flashNextEmbedding {
			identity[index*flashNextEmbedding+index] = 1
		}
		values(p+"fc_hidden.weight", identity, flashNextEmbedding, flashNextEmbedding)
		zeros(p+"pre_fc_norm_hidden.weight", wide)
		trunkTensor(p+"hyper_connection_mixer.hc_norm.weight", "output_hc_norm.weight", true, wide)
		trunkTensor(p+"hyper_connection_mixer.input_mix_weight_down.weight", "output_hc_down.weight", false, wide, rank)
		trunkTensor(p+"hyper_connection_mixer.input_mix_weight_up.weight", "output_hc_up.weight", false, rank, wide)
	} else {
		random(p+"fc_embedding.weight", flashNextEmbedding, flashNextEmbedding)
		random(p+"fc_hidden.weight", flashNextEmbedding, flashNextEmbedding)
		random(p+"pre_fc_norm_hidden.weight", wide)
		random(p+"hyper_connection_mixer.hc_norm.weight", wide)
		random(p+"hyper_connection_mixer.input_mix_weight_down.weight", wide, rank)
		random(p+"hyper_connection_mixer.input_mix_weight_up.weight", rank, wide)
	}
	random(p+"pre_fc_norm_embedding.weight", flashNextEmbedding)
	random(l+"self_attn.q_proj.weight", flashNextEmbedding, flashNextHeads*flashNextHeadWidth*2)
	random(l+"self_attn.k_proj.weight", flashNextEmbedding, flashNextHeadWidth)
	random(l+"self_attn.v_proj.weight", flashNextEmbedding, flashNextHeadWidth)
	random(l+"self_attn.q_norm.weight", flashNextHeadWidth)
	random(l+"self_attn.k_norm.weight", flashNextHeadWidth)
	for _, stage := range []string{"attn", "mlp"} {
		random(l+stage+"_hyper_connection.hc_norm.weight", wide)
		random(l+stage+"_hyper_connection.input_mix_weight_down.weight", wide, rank)
		random(l+stage+"_hyper_connection.input_mix_weight_up.weight", rank, wide)
		random(l+stage+"_hyper_connection.block_inject_weight.weight", wide, streams)
	}
	random(l+"mlp.gate.weight", flashNextEmbedding, flashNextExperts)
	random(l+"mlp.experts.gate_up_proj", flashNextEmbedding, 2*flashNextExpertWidth, flashNextExperts)
	random(l+"mlp.shared_expert_gate.weight", flashNextEmbedding, 1)
	random(l+"mlp.shared_expert.gate_proj.weight", flashNextEmbedding, flashNextShared)
	random(l+"mlp.shared_expert.up_proj.weight", flashNextEmbedding, flashNextShared)
	if copyTrunk {
		zeros(l+"self_attn.o_proj.weight", flashNextHeads*flashNextHeadWidth, flashNextEmbedding)
		zeros(l+"mlp.experts.down_proj", flashNextExpertWidth, flashNextEmbedding, flashNextExperts)
		zeros(l+"mlp.shared_expert.down_proj.weight", flashNextShared, flashNextEmbedding)
	} else {
		random(l+"self_attn.o_proj.weight", flashNextHeads*flashNextHeadWidth, flashNextEmbedding)
		random(l+"mlp.experts.down_proj", flashNextExpertWidth, flashNextEmbedding, flashNextExperts)
		random(l+"mlp.shared_expert.down_proj.weight", flashNextShared, flashNextEmbedding)
	}
	metadata := []gguf.Metadata{testutil.GGUFScalar("general.architecture", gguf.ValueTypeString, "qwen4exp-mtp")}
	path := filepath.Join(t.TempDir(), "hermetic-flash-next-mtp.gguf")
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

// copyDraftAcceptance replays generateDeviceSpeculativeGreedy's rounds over the plain greedy tokens when every draft repeats the last emitted token: what the copying draft layer drafts.
func copyDraftAcceptance(plain []tokenizer.TokenID, generated int) (drafted, accepted int) {
	emitted := 1
	pending := 1
	haveSeed := false
	for emitted < generated {
		count := 0
		if haveSeed && pending+hyperDraftTokens <= q8InputSpanColumnsWindow && (hyperDraftTokens > 1 || pending == 1) {
			// A window drafts no more than the output has left after its own token.
			count = min(hyperDraftTokens, generated-emitted-1)
		}
		last := plain[emitted-1]
		run := 0
		for run < count && plain[emitted+run] == last {
			run++
		}
		drafted, accepted = drafted+count, accepted+run
		emitted += run + 1
		haveSeed = true
		// A window whose drafts were all accepted, or rejected at the first, leaves only its correction pending.
		if run == count || run == 0 {
			pending = 1
		} else {
			pending += run + 1
		}
	}
	return drafted, accepted
}

// TestFlashNextMTPKeepsThePromptCache speculates two turns of one conversation with the prompt cache on: the second
// turn reads only what follows the first turn's last boundary, and both keep the trunk's plain greedy tokens, with a
// mostly rejected draft layer (whose last boundary lags its tokens) and a copying one.
func TestFlashNextMTPKeepsThePromptCache(t *testing.T) {
	// Serial: the generations share the one CUDA device.
	requireIntegration(t)
	cudatest.Require(t)
	trunk := writeHermeticFlashNextGGUF(t)
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	const generated = 12
	first := []tokenizer.TokenID{4, 9, 17}
	// The follow-up turn adds a user's tokens after the first turn's answer.
	followUp := []tokenizer.TokenID{21, 30}
	plainRunner, err := openFixtureRunnerWithResidency(trunk, OpenOptions{}, recipe.ResidencyDeviceNative)
	if err != nil {
		t.Fatal(err)
	}
	plain := func(prompt []tokenizer.TokenID) []tokenizer.TokenID {
		t.Helper()
		ids, _, err := plainRunner.Generate(t.Context(), "", GenerateOptions{
			MaxNewTokens: generated, Sampler: greedy, DeviceGreedy: true, PromptTokenIDs: prompt, ContinueAfterEOG: true,
		})
		if err != nil {
			t.Fatal(err)
		}
		return ids
	}
	plainFirst := plain(first)
	plainSecond := plain(append(slices.Clone(plainFirst), followUp...))
	if err := plainRunner.Close(); err != nil {
		t.Fatal(err)
	}
	for _, copyTrunk := range []bool{false, true} {
		runner, err := openFixtureRunnerWithResidency(trunk,
			OpenOptions{Draft: writeHermeticFlashNextMTPGGUF(t, trunk, copyTrunk), PromptCacheEntries: 1}, recipe.ResidencyDeviceNative)
		if err != nil {
			t.Fatal(err)
		}
		turn := func(prompt []tokenizer.TokenID) ([]tokenizer.TokenID, int) {
			t.Helper()
			cached := -1
			ids, _, err := runner.Generate(t.Context(), "", GenerateOptions{
				MaxNewTokens: generated, Sampler: greedy, DeviceGreedy: true, SpeculativeDecode: true, CachePrompt: true,
				PromptTokenIDs: prompt, ContinueAfterEOG: true,
				OnPromptEvaluated: func(evaluation PromptEvaluation) { cached = evaluation.Cached },
			})
			if err != nil {
				_ = runner.Close()
				t.Fatal(err)
			}
			return ids, cached
		}
		one, _ := turn(first)
		two, cached := turn(append(slices.Clone(one), followUp...))
		if closeErr := runner.Close(); closeErr != nil {
			t.Fatal(closeErr)
		}
		if !slices.Equal(one, plainFirst) || !slices.Equal(two, plainSecond) {
			t.Fatalf("copy=%t: speculative turns %v and %v differ from plain %v and %v", copyTrunk, one, two, plainFirst, plainSecond)
		}
		if cached <= len(first) || cached >= len(one) {
			t.Fatalf("copy=%t: the second turn reused %d cached tokens of the first turn's %d", copyTrunk, cached, len(one))
		}
		t.Logf("copy=%t: the second turn reused %d of %d tokens", copyTrunk, cached, len(two)-generated)
	}
}

// TestFlashNextMTPKeepsGreedyText drafts with a Flash-Next-shaped draft layer attached from its own file and holds speculative greedy generation to plain greedy generation token for token.
//   - a random draft layer is mostly rejected, so verify spans re-span from their boundary
//   - a copying draft layer drafts the token the trunk selected from the same streams, so its accepted drafts are exactly the plain text's repeats
//   - both under F32 and native residency, and the runner without the draft declares no draft span
func TestFlashNextMTPKeepsGreedyText(t *testing.T) {
	// Serial: the generations share the one CUDA device.
	requireIntegration(t)
	cudatest.Require(t)
	trunk := writeHermeticFlashNextGGUF(t)
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	const generated = 24
	prompt := []tokenizer.TokenID{4, 9, 17}
	for _, residency := range []recipe.ResidencyPolicy{recipe.ResidencyDeviceF32, recipe.ResidencyDeviceNative} {
		plainRunner, err := openFixtureRunnerWithResidency(trunk, OpenOptions{}, residency)
		if err != nil {
			t.Fatal(err)
		}
		if tokens := plainRunner.SpeculativeDraftTokens(); tokens != 0 {
			t.Fatalf("%s: the trunk alone declares %d draft tokens", residency, tokens)
		}
		// The lookahead past the generation covers drafts accepted beyond its last token.
		lookahead, _, err := plainRunner.Generate(t.Context(), "", GenerateOptions{
			MaxNewTokens: generated + hyperDraftTokens + 1, Sampler: greedy, DeviceGreedy: true,
			PromptTokenIDs: prompt, ContinueAfterEOG: true,
		})
		if closeErr := plainRunner.Close(); err != nil || closeErr != nil {
			t.Fatal(err, closeErr)
		}
		lookahead = lookahead[len(prompt):]
		if len(slices.Compact(slices.Clone(lookahead))) == 1 {
			t.Fatalf("%s: the fixture selects one token throughout (%v)", residency, lookahead)
		}
		for _, copyTrunk := range []bool{false, true} {
			runner, err := openFixtureRunnerWithResidency(trunk, OpenOptions{Draft: writeHermeticFlashNextMTPGGUF(t, trunk, copyTrunk)}, residency)
			if err != nil {
				t.Fatal(err)
			}
			if tokens := runner.SpeculativeDraftTokens(); tokens != hyperDraftTokens {
				_ = runner.Close()
				t.Fatalf("%s: the attached draft declares %d draft tokens", residency, tokens)
			}
			var drafted, accepted int
			options := GenerateOptions{
				MaxNewTokens: generated, Sampler: greedy, DeviceGreedy: true, SpeculativeDecode: true,
				PromptTokenIDs: prompt, ContinueAfterEOG: true,
				OnDraftSpan: func(count, kept int) { drafted, accepted = drafted+count, accepted+kept },
			}
			speculated, speculatedText, err := runner.Generate(t.Context(), "", options)
			if err != nil {
				_ = runner.Close()
				t.Fatal(err)
			}
			// A draft step after the generation's replays its captured graph: one graph launch and no kernel launch.
			replayed := func() (uint64, uint64, error) {
				runner.mu.Lock()
				defer runner.mu.Unlock()
				width := runner.hyperDraftWidth()
				session, err := runner.newDeviceMTPSessionLocked(reference.Value{Shape: tensor.MustShape(width, 1), Data: make([]float32, width)}, 5)
				if err != nil {
					return 0, 0, err
				}
				before, err := runner.DeviceExecutionStats(t.Context())
				if err != nil {
					return 0, 0, err
				}
				if _, _, err := runner.draftDeviceTokenLocked(t.Context(), 4, session); err != nil {
					return 0, 0, err
				}
				after, err := runner.DeviceExecutionStats(t.Context())
				return after.GraphLaunches - before.GraphLaunches, after.KernelLaunches - before.KernelLaunches, err
			}
			graphs, kernels, err := replayed()
			if err != nil || graphs != 1 || kernels != 0 {
				_ = runner.Close()
				t.Fatalf("%s copy=%t: a later draft step launched %d graphs and %d kernels (%v), want one replay", residency, copyTrunk, graphs, kernels, err)
			}
			options.SpeculativeDecode, options.OnDraftSpan = false, nil
			plain, plainText, err := runner.Generate(t.Context(), "", options)
			if closeErr := runner.Close(); err != nil || closeErr != nil {
				t.Fatal(err, closeErr)
			}
			if !slices.Equal(speculated, plain) || speculatedText != plainText {
				t.Fatalf("%s copy=%t: speculative tokens %v differ from plain %v", residency, copyTrunk, speculated, plain)
			}
			if !slices.Equal(plain[len(prompt):], lookahead[:generated]) {
				t.Fatalf("%s copy=%t: the runner with the draft generates %v, the trunk alone %v", residency, copyTrunk, plain[len(prompt):], lookahead[:generated])
			}
			if drafted == 0 {
				t.Fatalf("%s copy=%t: no span drafted", residency, copyTrunk)
			}
			if copyTrunk {
				wantDrafted, wantAccepted := copyDraftAcceptance(lookahead, generated)
				if drafted != wantDrafted || accepted != wantAccepted {
					t.Fatalf("%s: the copying draft layer had %d of %d drafts accepted, the plain text's repeats give %d of %d", residency, accepted, drafted, wantAccepted, wantDrafted)
				}
			}
			t.Logf("%s copy=%t: %d tokens, drafts accepted %d of %d", residency, copyTrunk, generated, accepted, drafted)
		}
	}
}
