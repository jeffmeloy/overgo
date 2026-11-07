package hybridtrain

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"math"
	"slices"
	"strings"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestLoadStackFeedForward(t *testing.T) {
	for _, storageType := range []dtype.Type{dtype.F32, dtype.BF16} {
		t.Run(storageType.String(), func(t *testing.T) {
			names := []string{"gate", "up", "down", "norm"}
			var tensors []gguf.TensorData
			var expected [][]float32
			for index, name := range names {
				values := []float32{float32(index+1) * 10, float32(index+1)*10 + .25, float32(index+1)*10 + .5, float32(index+1)*10 + .75}
				var raw []byte
				for _, value := range values {
					if storageType == dtype.F32 {
						raw = binary.LittleEndian.AppendUint32(raw, math.Float32bits(value))
					} else {
						raw = binary.LittleEndian.AppendUint16(raw, dtype.Float32ToBF16(value))
					}
				}
				tensors = append(tensors, gguf.TensorData{Name: name, Shape: []uint64{2, 2}, Type: storageType, Data: bytes.NewReader(raw)})
				expected = append(expected, values)
			}
			file := testcheck.Must(gguf.Open(testutil.TempGGUF(t, "layer.gguf", nil, tensors)))(t)
			defer file.Close()
			info := model.LayerWeights{FeedForwardGate: &file.Tensors[0], FeedForwardUp: &file.Tensors[1], FeedForwardDown: &file.Tensors[2], AttentionNorm: &file.Tensors[3]}
			m := &Model{matW: make([]float32, 2, 14)}
			m.matW[0], m.matW[1] = 101, 102
			masters := m.matW[len(m.matW):cap(m.matW)]
			var scratch []byte
			host := testcheck.Must(loadStackLayer(t.Context(), file, info, masters, len(expected[0]), &scratch))(t)
			b := &builder{m: m}
			var bound [3][]float32
			for index, value := range []*reference.Value{host.FeedForwardGate, host.FeedForwardUp, host.FeedForwardDown} {
				if value == nil || !slices.Equal(value.Data, expected[index]) || &value.Data[0] != &masters[index*4] {
					t.Fatalf("matrix %d is not decoded directly into its master slot: %+v", index, value)
				}
				testcheck.NoError(t, b.matValues(&bound[index], names[index], 2, 2, value.Data))
			}
			b.rebindAliases()
			want := slices.Concat([]float32{101, 102}, expected[0], expected[1], expected[2])
			if !slices.Equal(m.matW, want) || !slices.Equal(host.AttentionNorm.Data, expected[3]) {
				t.Fatalf("master order or remaining layer tensor changed: %v / %v", m.matW, host.AttentionNorm.Data)
			}
			for index := range bound {
				if &bound[index][0] != &masters[index*4] {
					t.Fatalf("binding %d copied out of the master slab", index)
				}
			}
			bad := *info.FeedForwardGate
			bad.Shape[0]++
			info.FeedForwardGate = &bad
			if _, err := loadStackLayer(t.Context(), file, info, masters, len(expected[0]), &scratch); err == nil {
				t.Fatal("wrong matrix extent accepted")
			}
			info.FeedForwardGate = &file.Tensors[0]
			info.AttentionNorm = nil
			cancelled, cancel := context.WithCancel(t.Context())
			cancel()
			if _, err := loadStackLayer(cancelled, file, info, masters, len(expected[0]), &scratch); !errors.Is(err, context.Canceled) {
				t.Fatalf("cancelled decode: %v", err)
			}
		})
	}
}

// TestAGatedArtifactIsRefusedForWeightTraining writes a one-layer Qwen3.5 file, its attention's query projection fused
// with a per-head output gate as every Qwen3.5 file's is. Training its stack is refused for the gate before any weight
// is read, a cancelled load refused the same; walked frozen, its layer binds each head's query rows and gate rows from
// the fused projection.
func TestAGatedArtifactIsRefusedForWeightTraining(t *testing.T) {
	t.Parallel()
	const hidden, ffn, heads, kvHeads, width, vocab = 8, 16, 2, 1, 4, 6
	metadata := []gguf.Metadata{
		testutil.GGUFScalar("general.architecture", gguf.ValueTypeString, "qwen35"),
		testutil.GGUFScalar("qwen35.block_count", gguf.ValueTypeUint32, uint32(1)),
		testutil.GGUFScalar("qwen35.full_attention_interval", gguf.ValueTypeUint32, uint32(1)),
		testutil.GGUFScalar("qwen35.context_length", gguf.ValueTypeUint32, uint32(64)),
		testutil.GGUFScalar("qwen35.embedding_length", gguf.ValueTypeUint32, uint32(hidden)),
		testutil.GGUFScalar("qwen35.feed_forward_length", gguf.ValueTypeUint32, uint32(ffn)),
		testutil.GGUFScalar("qwen35.attention.head_count", gguf.ValueTypeUint32, uint32(heads)),
		testutil.GGUFScalar("qwen35.attention.head_count_kv", gguf.ValueTypeUint32, uint32(kvHeads)),
		testutil.GGUFScalar("qwen35.attention.key_length", gguf.ValueTypeUint32, uint32(width)),
		testutil.GGUFScalar("qwen35.attention.value_length", gguf.ValueTypeUint32, uint32(width)),
		testutil.GGUFScalar("qwen35.rope.freq_base", gguf.ValueTypeFloat32, float32(10_000)),
		testutil.GGUFScalar("qwen35.rope.dimension_count", gguf.ValueTypeUint32, uint32(width)),
		testutil.GGUFArray("qwen35.rope.dimension_sections", gguf.ValueTypeInt32, []int32{1, 1, 0, 0}),
		testutil.GGUFScalar("qwen35.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		testutil.GGUFScalar("qwen35.ssm.conv_kernel", gguf.ValueTypeUint32, uint32(4)),
		testutil.GGUFScalar("qwen35.ssm.inner_size", gguf.ValueTypeUint32, uint32(hidden)),
		testutil.GGUFScalar("qwen35.ssm.state_size", gguf.ValueTypeUint32, uint32(width)),
		testutil.GGUFScalar("qwen35.ssm.time_step_rank", gguf.ValueTypeUint32, uint32(heads)),
		testutil.GGUFScalar("qwen35.ssm.group_count", gguf.ValueTypeUint32, uint32(kvHeads)),
		testutil.GGUFArray("tokenizer.ggml.tokens", gguf.ValueTypeString, []string{"a", "b", "c", "d", "e", "f"}),
	}
	var tensors []gguf.TensorData
	for index, tensor := range []struct {
		name  string
		shape []uint64
	}{
		{"token_embd.weight", []uint64{hidden, vocab}}, {"output_norm.weight", []uint64{hidden}}, {"output.weight", []uint64{hidden, vocab}},
		{"blk.0.attn_norm.weight", []uint64{hidden}}, {"blk.0.attn_q.weight", []uint64{hidden, 2 * heads * width}},
		{"blk.0.attn_k.weight", []uint64{hidden, kvHeads * width}}, {"blk.0.attn_v.weight", []uint64{hidden, kvHeads * width}},
		{"blk.0.attn_output.weight", []uint64{heads * width, hidden}}, {"blk.0.attn_q_norm.weight", []uint64{width}},
		{"blk.0.attn_k_norm.weight", []uint64{width}}, {"blk.0.post_attention_norm.weight", []uint64{hidden}},
		{"blk.0.ffn_gate.weight", []uint64{hidden, ffn}}, {"blk.0.ffn_up.weight", []uint64{hidden, ffn}}, {"blk.0.ffn_down.weight", []uint64{ffn, hidden}},
	} {
		tensors = append(tensors, testutil.GGUFTensorF32(tensor.name, tensor.shape, index+1))
	}
	gated := testutil.TempGGUF(t, "gated.gguf", metadata, tensors)
	tokens := []uint32{1, 2, 3}
	cancelled, cancel := context.WithCancel(t.Context())
	cancel()
	for _, ctx := range []context.Context{t.Context(), cancelled} {
		if _, err := LoadStackArtifact(ctx, gated, tokens); err == nil || !strings.Contains(err.Error(), "output gate") {
			t.Fatalf("training a gated stack was not refused for its gate before reading a weight: %v", err)
		}
	}
	frozen := testcheck.Must(LoadFrozenLanguageModel(t.Context(), gated, len(tokens)-1))(t)
	defer frozen.Close()
	weights := testcheck.Must(model.ReadWeights(frozen.file, testcheck.Must(model.ReadSpec(frozen.file))(t)))(t)
	fused := testcheck.Must(model.LoadHostTensor(t.Context(), frozen.file, *weights.Layers[0].AttentionQ))(t).Data
	head := width * hidden
	for h := range heads {
		attention := frozen.Layers[0].Attn
		if !slices.Equal(attention.Wq[h*head:(h+1)*head], fused[2*h*head:(2*h+1)*head]) || !slices.Equal(attention.Gate[h*head:(h+1)*head], fused[(2*h+1)*head:(2*h+2)*head]) {
			t.Fatalf("head %d's query and gate rows are not the fused projection's [query | gate] rows", h)
		}
	}
}
