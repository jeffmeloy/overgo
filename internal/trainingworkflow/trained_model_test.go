package trainingworkflow

import (
	"bytes"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/testutil"
)

// TestTrainedModelKeepsUnchangedTensorsAsStored writes a trained model over
// a base holding an f16 tensor training left as it was, an f32 tensor
// training changed and a tensor training never held: only the changed
// tensor is rewritten, in f32; the others keep the base's stored bytes and
// type. A trained tensor set the base does not hold refuses.
func TestTrainedModelKeepsUnchangedTensorsAsStored(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	base := filepath.Join(root, "base.gguf")
	half := []byte{0x00, 0x3c, 0x00, 0x40, 0x00, 0x42, 0x00, 0x44} // 1, 2, 3, 4
	if err := gguf.WriteFileExclusive(base, []gguf.Metadata{testutil.GGUFScalar("general.architecture", gguf.ValueTypeString, "fixture")}, []gguf.TensorData{
		{Name: "unchanged.weight", Shape: []uint64{4}, Type: gguf.DTypeF16, Data: bytes.NewReader(half)},
		testutil.GGUFTensorF32("changed.weight", []uint64{2}, 1),
		testutil.GGUFTensorF32("untrained.weight", []uint64{2}, 2),
	}, gguf.WriteOptions{}); err != nil {
		t.Fatal(err)
	}
	trained := map[string][]float32{"unchanged.weight": {1, 2, 3, 4}, "changed.weight": {5, 6}}
	lookup := func(name string) ([]float32, bool) { values, ok := trained[name]; return values, ok }
	path := filepath.Join(root, "trained.gguf")
	if err := WriteTrainedModel(base, path, len(trained), lookup); err != nil {
		t.Fatal(err)
	}
	original, err := gguf.Open(base)
	if err != nil {
		t.Fatal(err)
	}
	defer original.Close()
	written, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer written.Close()
	stored := func(file *gguf.File, name string) (gguf.DType, []byte) {
		info, ok := file.Tensor(name)
		data := make([]byte, info.Size)
		if !ok || file.ReadTensorData(info, data) != nil {
			t.Fatalf("tensor %s is absent", name)
		}
		return info.Type, data
	}
	for _, name := range []string{"unchanged.weight", "untrained.weight"} {
		baseType, baseData := stored(original, name)
		writtenType, writtenData := stored(written, name)
		if writtenType != baseType || !bytes.Equal(writtenData, baseData) {
			t.Fatalf("%s = %s %x, want the base's %s %x", name, writtenType, writtenData, baseType, baseData)
		}
	}
	changedType, changedData := stored(written, "changed.weight")
	if changedType != gguf.DTypeF32 || !slices.Equal(changedData, []byte{0, 0, 0xa0, 0x40, 0, 0, 0xc0, 0x40}) {
		t.Fatalf("changed.weight = %s %x, want f32 5 and 6", changedType, changedData)
	}
	trained["absent.weight"] = []float32{1}
	if err := WriteTrainedModel(base, filepath.Join(root, "refused.gguf"), len(trained), lookup); err == nil {
		t.Fatal("a trained tensor the base does not hold was accepted")
	}
}
