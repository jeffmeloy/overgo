package latentimage

import (
	"encoding/binary"
	"math"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
)

// TestAMediaAdapterReadsItsThreeFormats holds the adapter reader (media-lora/unmerged) to the three formats the MOSS
// Ultra workflow's adapters use: a kohya pair with its alpha, scaled by alpha over its rank, under ComfyUI's
// diffusion_model. prefix; a PEFT pair, scaled by 1, with no alpha; a joint block's fused qkv pair, read as the graph's
// q, k and v, each a third of the up rows, its third's payload those rows, while a text embedder's fused qkv, which the
// graph reads fused, stays one; and the full differences of a norm's weight and a bias. A tensor no format names, a
// pair missing a half, and a file of no patch are refused.
func TestAMediaAdapterReadsItsThreeFormats(t *testing.T) {
	t.Parallel()
	up := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	path := writeAdapter(t, "formats", map[string][]float32{
		"diffusion_model.layers.0.proj.q.lora_down.weight":   make([]float32, 8),
		"diffusion_model.layers.0.proj.q.lora_up.weight":     make([]float32, 6),
		"diffusion_model.layers.0.proj.q.alpha":              {4},
		"layers.0.proj.o.lora_A.weight":                      make([]float32, 8),
		"layers.0.proj.o.lora_B.weight":                      make([]float32, 8),
		"diffusion_model.layers.1.proj.qkv.lora_down.weight": make([]float32, 8),
		"diffusion_model.layers.1.proj.qkv.lora_up.weight":   up,
		"text_embedder.blocks.0.qkv.lora_A.weight":           make([]float32, 8),
		"text_embedder.blocks.0.qkv.lora_B.weight":           make([]float32, 12),
		"diffusion_model.layers.0.norm1.diff":                make([]float32, 4),
		"diffusion_model.text_embedder.proj.diff_b":          make([]float32, 4),
	}, map[string][]int{
		"diffusion_model.layers.0.proj.q.lora_down.weight":   {2, 4},
		"diffusion_model.layers.0.proj.q.lora_up.weight":     {3, 2},
		"diffusion_model.layers.0.proj.q.alpha":              {},
		"layers.0.proj.o.lora_A.weight":                      {2, 4},
		"layers.0.proj.o.lora_B.weight":                      {4, 2},
		"diffusion_model.layers.1.proj.qkv.lora_down.weight": {2, 4},
		"diffusion_model.layers.1.proj.qkv.lora_up.weight":   {6, 2},
		"text_embedder.blocks.0.qkv.lora_A.weight":           {2, 4},
		"text_embedder.blocks.0.qkv.lora_B.weight":           {6, 2},
		"diffusion_model.layers.0.norm1.diff":                {4},
		"diffusion_model.text_embedder.proj.diff_b":          {4},
	})
	adapter := testcheck.Must(readMediaAdapter(path))(t)
	if q := adapter.lowRank["layers.0.proj.q"]; q.scale != 2 || q.rank != 2 || q.in != 4 || q.out != 3 || q.split {
		t.Errorf("the kohya pair reads %+v; want rank 2, scale alpha 4 over rank 2, 4 in and 3 out", q)
	}
	if o := adapter.lowRank["layers.0.proj.o"]; o.scale != 1 || o.rank != 2 || o.out != 4 {
		t.Errorf("the PEFT pair reads %+v; want rank 2, scale 1 and 4 out", o)
	}
	for index, projection := range []string{"q", "k", "v"} {
		third := adapter.lowRank["layers.1.proj."+projection]
		if !third.split || third.rows != 2 || third.from != uint64(2*index) || third.down != "diffusion_model.layers.1.proj.qkv.lora_down.weight" {
			t.Errorf("the fused qkv's %s reads %+v; want rows %d and %d of its up tensor beside its down", projection, third, 2*index, 2*index+1)
		}
	}
	if _, fused := adapter.lowRank["layers.1.proj.qkv"]; fused {
		t.Error("a joint block's fused qkv pair stayed fused, where the graph reads q, k and v apart")
	}
	if embedder := adapter.lowRank["text_embedder.blocks.0.qkv"]; embedder.split || embedder.out != 6 {
		t.Errorf("the text embedder's fused qkv reads %+v; want it whole, as the graph reads it", embedder)
	}
	if adapter.diffs["layers.0.norm1.weight"] != "diffusion_model.layers.0.norm1.diff" || adapter.diffs["text_embedder.proj.bias"] != "diffusion_model.text_embedder.proj.diff_b" {
		t.Errorf("the full differences read %v", adapter.diffs)
	}
	source := testcheck.Must(safetensors.OpenFile(path))(t)
	defer source.Close()
	third := adapter.lowRank["layers.1.proj.k"]
	node := tensor.NewBuilder().Input(third.upInputName(), dtype.F32, tensor.MustShape(third.rank, third.rows))
	payload := testcheck.Must(testcheck.Must(adapterPayload(source, node))(t)())(t)
	got := make([]float32, len(payload)/4)
	for index := range got {
		got[index] = math.Float32frombits(binary.LittleEndian.Uint32(payload[4*index:]))
	}
	if !slices.Equal(got, up[4:8]) {
		t.Errorf("the fused qkv's k third binds %v; want its up rows 2 and 3, %v", got, up[4:8])
	}

	for name, refused := range map[string]struct {
		tensors map[string][]float32
		shapes  map[string][]int
		says    string
	}{
		"a replaced weight": {map[string][]float32{"video_embedder.weight": make([]float32, 4)}, map[string][]int{"video_embedder.weight": {2, 2}}, "no patch this reader applies"},
		"a half pair":       {map[string][]float32{"layers.0.proj.q.lora_A.weight": make([]float32, 4)}, map[string][]int{"layers.0.proj.q.lora_A.weight": {2, 2}}, "no up half"},
	} {
		if _, err := readMediaAdapter(writeAdapter(t, strings.ReplaceAll(name, " ", "-"), refused.tensors, refused.shapes)); err == nil || !strings.Contains(err.Error(), refused.says) {
			t.Errorf("an adapter with %s read with %v; want a refusal saying %q", name, err, refused.says)
		}
	}
}

// writeAdapter writes an adapter file of the tensors under a test's directory.
func writeAdapter(t *testing.T, name string, tensors map[string][]float32, shapes map[string][]int) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name+".safetensors")
	testcheck.NoError(t, safetensors.Save(path, tensors, shapes, nil))
	return path
}
