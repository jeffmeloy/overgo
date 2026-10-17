package model

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"slices"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestFlashNextNGram(t *testing.T) { RunFlashNextNGram(t, reference.Execute) }

// RunFlashNextNGram shares original-model cases between host and CUDA execution.
func RunFlashNextNGram(t *testing.T, run func([]*tensor.Tensor, map[*tensor.Tensor]reference.Value) (map[*tensor.Tensor]reference.Value, error)) {
	t.Helper()
	data, err := os.ReadFile("testdata/flashnext_ngram.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Tolerance float64
		Config    struct {
			Hidden  uint32  `json:"hidden_size"`
			HC      uint32  `json:"hc_count"`
			Size    uint32  `json:"ngram_size"`
			Heads   uint32  `json:"heads_per_ngram"`
			Conv    uint32  `json:"ple_conv_kernel_size"`
			Vocab   uint32  `json:"vocab_size"`
			EOS     uint32  `json:"eos_token_id"`
			Epsilon float32 `json:"rms_norm_eps"`
		}
		Multipliers, Offsets []uint64
		VocabSizes           []uint64 `json:"vocab_sizes"`
		Weights              map[string]struct {
			Shape []uint64
			Data  []float32
		}
		Cases []struct {
			Name                           string
			Start                          uint32
			Tokens, Rows, History          []uint32
			Input, Output, Embedding, Conv []float32
		}
	}
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	spec := qwen35TestSpec()
	spec.Architecture, spec.BlockCount = "qwen4exp", 1
	spec.EmbeddingLength, spec.HyperConnectionCount, spec.HyperConnectionRank = f.Config.Hidden, f.Config.HC, 3
	spec.VocabularySize, spec.RMSNormEpsilon = f.Config.Vocab, f.Config.Epsilon
	spec.NGram = &NGramSpec{Size: f.Config.Size, HeadsPerOrder: f.Config.Heads, RowWidth: f.Config.Hidden / uint32(len(f.Offsets)), ConvKernel: f.Config.Conv, EOS: f.Config.EOS, ImageToken: f.Config.EOS, Multipliers: f.Multipliers, Offsets: f.Offsets, VocabSizes: f.VocabSizes}
	spec = bindFixtureSpec(spec)
	table := f.Weights["per_layer_token_embd.weight"]
	var contents, encoded bytes.Buffer
	if err := binary.Write(&contents, binary.LittleEndian, table.Data); err != nil {
		t.Fatal(err)
	}
	if err := gguf.Write(&encoded, nil, []gguf.TensorData{{Name: "per_layer_token_embd.weight", Shape: table.Shape, Type: dtype.F32, Data: &contents}}, gguf.WriteOptions{}); err != nil {
		t.Fatal(err)
	}
	file, err := gguf.Parse(bytes.NewReader(encoded.Bytes()), uint64(encoded.Len()), gguf.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	var history []uint32
	var conv reference.Value
	for _, test := range f.Cases {
		if !t.Run(fmt.Sprintf("%s/%d", test.Name, test.Start), func(t *testing.T) {
			if test.Start == 0 {
				history = nil
				conv = reference.ZeroValue(tensor.MustShape(uint64(f.Config.Hidden*f.Config.HC), uint64((f.Config.Conv-1)*f.Config.Size)))
			}
			rows, next, err := spec.NGramRows(test.Tokens, history)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(rows, test.Rows) || !slices.Equal(next, test.History) {
				t.Fatalf("rows/history = %v/%v, original %v/%v", rows, next, test.Rows, test.History)
			}
			selected, err := LoadHostRows(t.Context(), file, file.Tensors[0], rows)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(selected.Data, test.Embedding) {
				t.Fatal("selected embedding differs from original")
			}
			b := tensor.NewBuilder()
			feeds := map[*tensor.Tensor]reference.Value{}
			input := func(name string, value reference.Value) *tensor.Tensor {
				node := b.Input(name, dtype.F32, value.Shape)
				feeds[node] = value
				return node
			}
			weight := func(name string) *tensor.Tensor {
				value := f.Weights[name+".weight"]
				return input(name, reference.Value{Shape: tensor.MustShape(value.Shape...), Data: value.Data})
			}
			tokens := uint64(len(test.Tokens))
			selected.Shape = tensor.MustShape(uint64(f.Config.Hidden), tokens)
			nextValues := make([]float32, len(next))
			for i, token := range next {
				nextValues[i] = float32(token)
			}
			c := CachedBlockContext{Builder: b, Sequences: 1, Positions: make([]uint32, tokens),
				PerLayerInput: input("selected", selected),
				NGramHistory:  input("history", reference.Value{Shape: tensor.MustShape(uint64(len(nextValues))), Data: nextValues}),
				PastStates:    CacheStates[*tensor.Tensor]{cacheStateNGramConvolution: {Mode: CacheStateFixed, Value: input("past", conv)}},
			}
			activation := input("input", reference.Value{Shape: tensor.MustShape(uint64(f.Config.Hidden), uint64(f.Config.HC), tokens), Data: test.Input})
			result, err := buildNGramInjection(blockDispatchOptions{Context: c, Spec: spec, Weights: LayerGraphWeights{
				NGramKey: weight("ple_key"), NGramValue: weight("ple_value"), NGramKeyNorm: weight("ple_norm_key"), NGramQueryNorm: weight("ple_norm_query"), NGramConvNorm: weight("ple_norm_conv"), NGramConv: weight("ple_conv1d"),
			}}, activation)
			if err != nil {
				t.Fatal(err)
			}
			values, err := run(result.States.AppendValues([]*tensor.Tensor{result.Output}), feeds)
			if err != nil {
				t.Fatal(err)
			}
			for _, check := range []struct {
				name string
				node *tensor.Tensor
				want []float32
			}{
				{"output", result.Output, test.Output}, {"convolution", result.States[cacheStateNGramConvolution].Value, test.Conv}, {"history", result.States[CacheStateNGramHistory].Value, nextValues},
			} {
				got := values[check.node].Data
				if len(got) != len(check.want) {
					t.Fatalf("%s length %d, want %d", check.name, len(got), len(check.want))
				}
				for i, want := range check.want {
					difference := math.Abs(float64(got[i] - want))
					if math.IsNaN(difference) || difference > f.Tolerance*(1+math.Abs(float64(want))) {
						t.Fatalf("%s[%d] = %.9g, original %.9g", check.name, i, got[i], want)
					}
				}
			}
			conv, history = values[result.States[cacheStateNGramConvolution].Value], next
		}) {
			return
		}
	}
	for _, test := range []struct {
		name            string
		tokens, history []uint32
	}{
		{"empty", nil, nil}, {"token-range", []uint32{spec.VocabularySize}, nil},
		{"history-range", []uint32{0}, []uint32{spec.VocabularySize, 0}}, {"history-size", []uint32{0}, []uint32{0}},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, _, err := spec.NGramRows(test.tokens, test.history); err == nil {
				t.Fatal("invalid token history accepted")
			}
		})
	}
	t.Run("signed-root", func(t *testing.T) {
		b := tensor.NewBuilder()
		data := []float32{-4, -ngramGateFloor, -ngramGateFloor / 2, 0, ngramGateFloor / 2, ngramGateFloor, 4}
		node := b.Input("gate", dtype.F32, tensor.MustShape(uint64(len(data))))
		out := b.SignedSqrt(node, ngramGateFloor)
		values, err := run([]*tensor.Tensor{out}, map[*tensor.Tensor]reference.Value{node: {Shape: node.Shape, Data: data}})
		if err != nil {
			t.Fatal(err)
		}
		for i, want := range []float32{-2, -.001, -.001, 0, .001, .001, 2} {
			if math.Abs(float64(values[out].Data[i]-want)) > 1e-7 {
				t.Fatalf("root[%d] = %g, want %g", i, values[out].Data[i], want)
			}
		}
	})
}
