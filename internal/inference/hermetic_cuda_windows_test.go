//go:build windows

package inference

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/gguf"
	"overgo/internal/tokenizer"
)

const (
	hermeticEmbedding = uint64(8)
	hermeticHeads     = uint64(2)
	hermeticKVHeads   = uint64(1)
	hermeticHeadWidth = hermeticEmbedding / hermeticHeads
	hermeticFFN       = uint64(12)
	hermeticVocab     = uint64(8)
	hermeticContext   = uint32(16)
)

func TestHermeticCUDAContinuousCacheParity(t *testing.T) {
	cudatest.Require(t)
	path := writeHermeticLlamaGGUF(t)
	runner, err := openFixtureRunnerWithOptions(path, OpenOptions{PreloadDeviceWeights: true, CachePageTokens: 4})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	deviceBatch, err := runner.NewContinuousBatch(ContinuousBatchOptions{
		MaxSequences: 2, Device: true, PageTokens: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer deviceBatch.Close(context.Background())
	hostBatch, err := runner.NewContinuousBatch(ContinuousBatchOptions{
		MaxSequences: 2, PageTokens: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer hostBatch.Close(context.Background())

	steps := [][]tokenizer.TokenID{{1, 4}, {5}, {6}}
	for step, tokens := range steps {
		device, deviceErr := deviceBatch.Step(context.Background(), []SequenceBatchInput{{ID: 1, Tokens: tokens}})
		host, hostErr := hostBatch.Step(context.Background(), []SequenceBatchInput{{ID: 1, Tokens: tokens}})
		if deviceErr != nil || hostErr != nil {
			t.Fatalf("step %d device/host errors = %v/%v", step, deviceErr, hostErr)
		}
		if len(device) != 1 || len(host) != 1 || device[0].Tokens != host[0].Tokens ||
			device[0].Position != host[0].Position || len(device[0].Logits) != len(host[0].Logits) {
			t.Fatalf("step %d states = %+v/%+v", step, device, host)
		}
		for index := range device[0].Logits {
			if delta := math.Abs(float64(device[0].Logits[index] - host[0].Logits[index])); delta > 2e-4 {
				t.Fatalf("step %d logit %d delta = %g", step, index, delta)
			}
		}
	}
	deviceBatch.mu.Lock()
	session := deviceBatch.sequences[1].device.session
	deviceBatch.mu.Unlock()
	if session == nil || session.replays != 1 || session.program.identity.output.mode != deviceOutputLogits {
		t.Fatalf("generic decode session = %+v", session)
	}
	beforeFailure := deviceBatch.Snapshot()
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := deviceBatch.Step(canceled, []SequenceBatchInput{{
		ID: 1, Tokens: []tokenizer.TokenID{7},
	}}); err == nil {
		t.Fatal("canceled device append succeeded")
	}
	if afterFailure := deviceBatch.Snapshot(); !reflect.DeepEqual(afterFailure, beforeFailure) {
		t.Fatalf("canceled append changed cache state: before=%+v after=%+v", beforeFailure, afterFailure)
	}
	if err := deviceBatch.Fork(1, 2); err != nil {
		t.Fatal(err)
	}
	branched, err := deviceBatch.Step(context.Background(), []SequenceBatchInput{
		{ID: 1, Tokens: []tokenizer.TokenID{2}},
		{ID: 2, Tokens: []tokenizer.TokenID{3}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(branched) != 2 || branched[0].Tokens != branched[1].Tokens ||
		branched[0].Position != branched[1].Position {
		t.Fatalf("forked states = %+v", branched)
	}
	deviceBatch.mu.Lock()
	cohortSession := deviceBatch.sequences[1].device.session
	var cohortReplays uint64
	if cohortSession != nil {
		cohortReplays = cohortSession.replays
	}
	deviceBatch.mu.Unlock()
	if cohortSession == nil {
		t.Fatal("forked cohort has no reusable session")
	}
	if _, err = deviceBatch.Step(context.Background(), []SequenceBatchInput{
		{ID: 1, Tokens: []tokenizer.TokenID{4}},
		{ID: 2, Tokens: []tokenizer.TokenID{5}},
	}); err != nil {
		t.Fatal(err)
	}
	deviceBatch.mu.Lock()
	first := deviceBatch.sequences[1].device
	second := deviceBatch.sequences[2].device
	deviceBatch.mu.Unlock()
	if first.session == nil || first.session != second.session || first.session != cohortSession ||
		first.session.replays != cohortReplays+1 ||
		first.sessionBranch != 0 || second.sessionBranch != 1 ||
		first.session.program.identity.branches != 2 {
		t.Fatalf("multi-branch decode sessions = %+v/%+v", first.session, second.session)
	}
}

// TestHermeticCUDACapacityCachePageBoundary drives the capacity decode session
// across a KV-cache page boundary (page=4: boundary at pastTokens=4) and beyond,
// comparing the device append path against the proven-correct host concat path at
// every step. The steps reach pastTokens 2..7 so the session is replayed while the
// storage lives in a GROWN page (Cap=8) -- the exact condition that exposed the
// keyShape/keyCapacity off-by-one on the 12B.
func TestHermeticCUDACapacityCachePageBoundary(t *testing.T) {
	cudatest.Require(t)
	path := writeHermeticLlamaGGUF(t)
	runner, err := openFixtureRunnerWithOptions(path, OpenOptions{PreloadDeviceWeights: true, CachePageTokens: 4})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	deviceBatch, err := runner.NewContinuousBatch(ContinuousBatchOptions{
		MaxSequences: 1, Device: true, PageTokens: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer deviceBatch.Close(context.Background())
	hostBatch, err := runner.NewContinuousBatch(ContinuousBatchOptions{
		MaxSequences: 1, PageTokens: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer hostBatch.Close(context.Background())

	// prefill 2 -> decode single tokens; cache.Tokens after each step = 2,3,4,5,6,7,8.
	// pastTokens=4 crosses page 0->1 (Cap 4->8); pastTokens=5,6,7 replay inside the
	// grown page -- where the stale source capacity struck.
	steps := [][]tokenizer.TokenID{{1, 4}, {5}, {6}, {7}, {4}, {5}, {6}}
	for step, tokens := range steps {
		device, deviceErr := deviceBatch.Step(context.Background(), []SequenceBatchInput{{ID: 1, Tokens: tokens}})
		host, hostErr := hostBatch.Step(context.Background(), []SequenceBatchInput{{ID: 1, Tokens: tokens}})
		if deviceErr != nil || hostErr != nil {
			t.Fatalf("step %d device/host errors = %v/%v", step, deviceErr, hostErr)
		}
		if len(device) != 1 || len(host) != 1 || device[0].Tokens != host[0].Tokens ||
			device[0].Position != host[0].Position || len(device[0].Logits) != len(host[0].Logits) {
			t.Fatalf("step %d states = %+v/%+v", step, device, host)
		}
		for index := range device[0].Logits {
			if delta := math.Abs(float64(device[0].Logits[index] - host[0].Logits[index])); delta > 2e-4 {
				t.Fatalf("step %d (tokens=%d) logit %d device=%g host=%g delta=%g",
					step, device[0].Tokens, index, device[0].Logits[index], host[0].Logits[index], delta)
			}
		}
	}
}

func writeHermeticLlamaGGUF(t *testing.T) string {
	t.Helper()
	metadata := []gguf.Metadata{
		hermeticScalar("general.architecture", gguf.ValueTypeString, "llama"),
		hermeticScalar("general.name", gguf.ValueTypeString, "hermetic-cuda"),
		hermeticScalar("llama.block_count", gguf.ValueTypeUint32, uint32(1)),
		hermeticScalar("llama.context_length", gguf.ValueTypeUint32, hermeticContext),
		hermeticScalar("llama.embedding_length", gguf.ValueTypeUint32, uint32(hermeticEmbedding)),
		hermeticScalar("llama.feed_forward_length", gguf.ValueTypeUint32, uint32(hermeticFFN)),
		hermeticScalar("llama.attention.head_count", gguf.ValueTypeUint32, uint32(hermeticHeads)),
		hermeticScalar("llama.attention.head_count_kv", gguf.ValueTypeUint32, uint32(hermeticKVHeads)),
		hermeticScalar("llama.rope.freq_base", gguf.ValueTypeFloat32, float32(10_000)),
		hermeticScalar("llama.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		hermeticScalar("tokenizer.ggml.model", gguf.ValueTypeString, "llama"),
		hermeticArray("tokenizer.ggml.tokens", gguf.ValueTypeString,
			[]string{"<unk>", "<s>", "</s>", "â–", "a", "b", "c", "d"}),
		hermeticArray("tokenizer.ggml.scores", gguf.ValueTypeFloat32, make([]float32, hermeticVocab)),
		hermeticArray("tokenizer.ggml.token_type", gguf.ValueTypeInt32,
			[]int32{2, 3, 3, 1, 1, 1, 1, 1}),
		hermeticScalar("tokenizer.ggml.bos_token_id", gguf.ValueTypeUint32, uint32(1)),
		hermeticScalar("tokenizer.ggml.eos_token_id", gguf.ValueTypeUint32, uint32(2)),
	}
	tensors := []gguf.TensorData{
		hermeticTensor("token_embd.weight", []uint64{hermeticEmbedding, hermeticVocab}, 1),
		hermeticTensor("output_norm.weight", []uint64{hermeticEmbedding}, 2),
		hermeticTensor("output.weight", []uint64{hermeticEmbedding, hermeticVocab}, 3),
		hermeticTensor("blk.0.attn_norm.weight", []uint64{hermeticEmbedding}, 4),
		hermeticTensor("blk.0.attn_q.weight", []uint64{hermeticEmbedding, hermeticEmbedding}, 5),
		hermeticTensor("blk.0.attn_k.weight", []uint64{hermeticEmbedding, hermeticHeadWidth}, 6),
		hermeticTensor("blk.0.attn_v.weight", []uint64{hermeticEmbedding, hermeticHeadWidth}, 7),
		hermeticTensor("blk.0.attn_output.weight", []uint64{hermeticEmbedding, hermeticEmbedding}, 8),
		hermeticTensor("blk.0.ffn_norm.weight", []uint64{hermeticEmbedding}, 9),
		hermeticTensor("blk.0.ffn_gate.weight", []uint64{hermeticEmbedding, hermeticFFN}, 10),
		hermeticTensor("blk.0.ffn_up.weight", []uint64{hermeticEmbedding, hermeticFFN}, 11),
		hermeticTensor("blk.0.ffn_down.weight", []uint64{hermeticFFN, hermeticEmbedding}, 12),
	}
	path := filepath.Join(t.TempDir(), "hermetic.gguf")
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

func hermeticScalar(key string, valueType gguf.ValueType, data any) gguf.Metadata {
	return gguf.Metadata{Key: key, Value: gguf.Value{Type: valueType, Data: data}}
}

func hermeticArray(key string, valueType gguf.ValueType, data any) gguf.Metadata {
	return gguf.Metadata{Key: key, Value: gguf.Value{
		Type: gguf.ValueTypeArray, ArrayType: valueType, Data: data,
	}}
}

func hermeticTensor(name string, shape []uint64, seed int) gguf.TensorData {
	elements := uint64(1)
	for _, dimension := range shape {
		elements *= dimension
	}
	values := make([]float32, elements)
	for index := range values {
		if len(shape) == 1 {
			values[index] = 1 + float32((index+seed)%3)*0.01
		} else {
			values[index] = float32((index*17+seed*13)%29-14) * 0.01
		}
	}
	var storage bytes.Buffer
	_ = binary.Write(&storage, binary.LittleEndian, values)
	return gguf.TensorData{Name: name, Shape: shape, Type: gguf.DTypeF32, Data: bytes.NewReader(storage.Bytes())}
}
