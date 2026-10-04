package hfconvert

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/peft"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
)

const (
	loRAQuery = "model.layers.0.self_attn.q_proj"
	loRAKey   = "model.layers.1.self_attn.k_proj"
	loRADown  = "model.layers.0.mlp.down_proj"
)

// loRACheckpoint is a two-layer llama checkpoint of two query heads and one key
// head, small enough that converting it is instant and large enough that its
// query and key rows are permuted.
type loRACheckpoint struct {
	directory string
	weights   map[string][]float32
	spec      testutil.DenseCausalSpec
}

func writeLoRACheckpoint(t *testing.T, modelType string) loRACheckpoint {
	t.Helper()
	directory := t.TempDir()
	spec := testutil.DenseCausalSpec{Vocab: 3, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 2, Seed: 3}
	weights, shapes := testutil.DenseCausalWeights(t, spec)
	if err := safetensors.Save(filepath.Join(directory, "model.safetensors"), weights, shapes, nil); err != nil {
		t.Fatal(err)
	}
	config, err := json.Marshal(map[string]any{
		"model_type": modelType, "hidden_size": spec.Hidden, "intermediate_size": spec.Intermediate, "num_attention_heads": spec.Heads,
		"num_key_value_heads": spec.KVHeads, "head_dim": spec.HeadDim, "num_hidden_layers": spec.Layers, "vocab_size": spec.Vocab,
		"max_position_embeddings": 16, "rms_norm_eps": 1e-5, "rope_theta": 10000, "tie_word_embeddings": true,
	})
	if err != nil {
		t.Fatal(err)
	}
	testutil.WriteTextFile(t, directory, "config.json", string(config))
	testutil.WriteTextFile(t, directory, "tokenizer.json", `{"model":{"vocab":{"a":0,"<s>":1,"</s>":2},"merges":[]},"added_tokens":[{"id":1,"content":"<s>","special":true},{"id":2,"content":"</s>","special":true}]}`)
	testutil.WriteTextFile(t, directory, "generation_config.json", `{"bos_token_id":1,"eos_token_id":2}`)
	return loRACheckpoint{directory: directory, weights: weights, spec: spec}
}

// loRAAdapter has a query pair at the config's rank and alpha, a key pair of its
// own rank and alpha, and a down-projection pair of a third rank: the three
// differ in what the conversion must carry exactly.
func loRAAdapter(rank bool) peft.Adapter {
	random := rand.New(rand.NewSource(11))
	values := func(count int) []float32 {
		result := make([]float32, count)
		for index := range result {
			result[index] = float32(random.NormFloat64()) / 4
		}
		return result
	}
	return peft.Adapter{
		Config: peft.Config{Rank: 2, Alpha: 8, RSLoRA: rank},
		Modules: map[string]peft.Module{
			loRAQuery: {Rank: 2, In: 8, Out: 8, Alpha: 8, Down: values(2 * 8), Up: values(8 * 2)},
			loRAKey:   {Rank: 1, In: 8, Out: 4, Alpha: 2, Down: values(1 * 8), Up: values(4 * 1)},
			loRADown:  {Rank: 3, In: 16, Out: 8, Alpha: 8, Down: values(3 * 16), Up: values(8 * 3)},
		},
	}
}

// rotaryRows moves each head's rows as llama.cpp does for interleaved rotary
// weights: the row at half s, pair i goes to row 2i+s.
func rotaryRows(matrix []float32, rows, columns, heads int) []float32 {
	headRows := rows / heads
	half := headRows / 2
	result := make([]float32, len(matrix))
	for head := range heads {
		for pair := range half {
			for split := range 2 {
				source := head*headRows + split*half + pair
				destination := head*headRows + 2*pair + split
				copy(result[destination*columns:(destination+1)*columns], matrix[source*columns:(source+1)*columns])
			}
		}
	}
	return result
}

func readFloat32(t *testing.T, file *gguf.File, name string) ([]float32, []uint64) {
	t.Helper()
	for _, info := range file.Tensors {
		if info.Name != name {
			continue
		}
		data := make([]byte, info.Size)
		if err := file.ReadTensorData(info, data); err != nil {
			t.Fatal(err)
		}
		values := make([]float32, len(data)/4)
		for index := range values {
			values[index] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*index:]))
		}
		return values, info.Extents()
	}
	t.Fatalf("tensor %q is not in the file", name)
	return nil, nil
}

// TestConvertLoRALoadsAndCarriesEveryUpdate: the adapter converts to a GGUF file
// inference loads against the converted base, and each pair, as inference
// scales it (the file's alpha over the pair's rank), makes exactly the update
// the PEFT adapter defines, in the base's own row order: for a rotary query or
// key weight, the PEFT update with its rows permuted as the converter permutes
// the weight. Rank-stabilized scales are carried the same way.
func TestConvertLoRALoadsAndCarriesEveryUpdate(t *testing.T) {
	checkpoint := writeLoRACheckpoint(t, "llama")
	basePath := filepath.Join(t.TempDir(), "base.gguf")
	if _, err := Convert(Options{Directory: checkpoint.directory, OutputPath: basePath}); err != nil {
		t.Fatal(err)
	}
	base, err := gguf.Open(basePath)
	if err != nil {
		t.Fatal(err)
	}
	defer base.Close()
	// The premise of the row permutation: the converted base holds the checkpoint's query rows permuted.
	baseQuery, _ := readFloat32(t, base, "blk.0.attn_q.weight")
	wantQuery := rotaryRows(checkpoint.weights[loRAQuery+".weight"], 8, 8, 2)
	for index := range baseQuery {
		if baseQuery[index] != wantQuery[index] {
			t.Fatalf("converted query weight[%d] = %v, want the permuted checkpoint's %v", index, baseQuery[index], wantQuery[index])
		}
	}
	spec, err := model.ReadSpec(base)
	if err != nil {
		t.Fatal(err)
	}
	for _, stabilized := range []bool{false, true} {
		adapter := loRAAdapter(stabilized)
		output := filepath.Join(t.TempDir(), "adapter.gguf")
		report, err := ConvertLoRA(adapter, LoRAOptions{Directory: checkpoint.directory, OutputPath: output})
		if err != nil || report.Modules != 3 || report.Tensors != 6 {
			t.Fatalf("stabilized=%t: conversion = %+v, %v, want three modules as six tensors", stabilized, report, err)
		}
		loaded, err := model.LoadLoRA(t.Context(), output, base, spec)
		if err != nil {
			t.Fatalf("stabilized=%t: inference refuses the converted adapter: %v", stabilized, err)
		}
		if loaded.Alpha != 8 || len(loaded.Weights) != 3 {
			t.Fatalf("stabilized=%t: loaded alpha %v with %d weights, want 8 and three", stabilized, loaded.Alpha, len(loaded.Weights))
		}
		file, err := gguf.Open(output)
		if err != nil {
			t.Fatal(err)
		}
		for module, destination := range map[string]string{loRAQuery: "blk.0.attn_q.weight", loRAKey: "blk.1.attn_k.weight", loRADown: "blk.0.ffn_down.weight"} {
			factors := adapter.Modules[module]
			down, downExtents := readFloat32(t, file, destination+".lora_a")
			up, upExtents := readFloat32(t, file, destination+".lora_b")
			if len(downExtents) != 2 || downExtents[0] != uint64(factors.In) || downExtents[1] != uint64(factors.Rank) ||
				len(upExtents) != 2 || upExtents[0] != uint64(factors.Rank) || upExtents[1] != uint64(factors.Out) {
				t.Fatalf("%s: factor extents %v and %v, want GGUF's reversed in-by-rank and rank-by-out", module, downExtents, upExtents)
			}
			want, err := adapter.Delta(module)
			if err != nil {
				t.Fatal(err)
			}
			switch module {
			case loRAQuery:
				want = rotaryRows(want, factors.Out, factors.In, 2)
			case loRAKey:
				want = rotaryRows(want, factors.Out, factors.In, 1)
			}
			inferenceScale := float64(loaded.Alpha) / float64(factors.Rank)
			for row := range factors.Out {
				for column := range factors.In {
					var sum float64
					for rank := range factors.Rank {
						sum += float64(up[row*factors.Rank+rank]) * float64(down[rank*factors.In+column])
					}
					if got := inferenceScale * sum; math.Abs(got-float64(want[row*factors.In+column])) > 1e-5 {
						t.Fatalf("stabilized=%t %s: update[%d,%d] = %v as inference applies it, want %v", stabilized, module, row, column, got, want[row*factors.In+column])
					}
				}
			}
		}
		_ = file.Close()
	}
}

// TestConvertLoRARefusesWhatItCannotName: a module that is not a linear
// projection of a layer the checkpoint has, a model type with no GGUF layout,
// and an output that exists.
func TestConvertLoRARefusesWhatItCannotName(t *testing.T) {
	checkpoint := writeLoRACheckpoint(t, "llama")
	for name, scenario := range map[string]struct {
		module string
		want   string
	}{
		"norm":       {"model.layers.0.input_layernorm", "not a linear projection"},
		"embedding":  {"model.embed_tokens", "not a linear projection"},
		"past layer": {"model.layers.5.self_attn.q_proj", "past the checkpoint's 2 layers"},
		"unknown":    {"model.layers.0.self_attn.rotary", "not a linear projection"},
	} {
		adapter := loRAAdapter(false)
		adapter.Modules = map[string]peft.Module{scenario.module: adapter.Modules[loRAQuery]}
		_, err := ConvertLoRA(adapter, LoRAOptions{Directory: checkpoint.directory, OutputPath: filepath.Join(t.TempDir(), "adapter.gguf")})
		if err == nil || !strings.Contains(err.Error(), scenario.want) {
			t.Errorf("%s: err = %v, want it to say %q", name, err, scenario.want)
		}
	}
	unsupported := writeLoRACheckpoint(t, "gemma")
	if _, err := ConvertLoRA(loRAAdapter(false), LoRAOptions{Directory: unsupported.directory, OutputPath: filepath.Join(t.TempDir(), "adapter.gguf")}); err == nil || !strings.Contains(err.Error(), "unsupported") {
		t.Errorf("a gemma checkpoint: err = %v, want it unsupported", err)
	}
	existing := filepath.Join(t.TempDir(), "adapter.gguf")
	if err := os.WriteFile(existing, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := ConvertLoRA(loRAAdapter(false), LoRAOptions{Directory: checkpoint.directory, OutputPath: existing}); err == nil {
		t.Error("an existing output was overwritten")
	}
}
