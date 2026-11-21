package latentimage

import (
	"maps"
	"testing"

	"overgo/internal/safetensors"
	"overgo/internal/testcheck"
)

// TestAFileDeclaresTheLayersThatTakeTheInt8Attention holds int8AttentionLayers to ComfyUI's reading of a file's
// attention records (ComfyAttention): a layer whose comfy_attention.config names comfy_kitchen_int8, alone or first of
// a list, takes the int8 attention, as the MOSS Ultra workflow's file declares for each joint layer; a layer with no
// record keeps the BF16 attention; and a layer that declares another method is refused, since overgo runs no other.
func TestAFileDeclaresTheLayersThatTakeTheInt8Attention(t *testing.T) {
	t.Parallel()
	tensors := map[string]safetensors.Tensor{}
	declare := func(name, record string) {
		tensors[name] = testcheck.Must(safetensors.NewTensor(name, "U8", []uint64{uint64(len(record))}, &countedReads{data: []byte(record)}, 0, int64(len(record))))(t)
	}
	declare("layers.0.comfy_attention.config", `{"attention": "comfy_kitchen_int8"}`)
	declare("layers.2.comfy_attention.config", `[{"attention": "comfy_kitchen_int8"}, {"attention": "comfy_kitchen_sol"}]`)
	got := testcheck.Must(int8AttentionLayers(&safetensors.Source{Tensors: tensors}, 3))(t)
	if want := map[int]bool{0: true, 2: true}; !maps.Equal(got, want) {
		t.Fatalf("the file declares the int8 attention for layers %v; want %v", got, want)
	}
	declare("layers.1.comfy_attention.config", `{"attention": "comfy_kitchen_sol"}`)
	if _, err := int8AttentionLayers(&safetensors.Source{Tensors: tensors}, 3); err == nil {
		t.Fatal("a layer that declares comfy_kitchen_sol was read as taking an attention overgo runs")
	}
}
