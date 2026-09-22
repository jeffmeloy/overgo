package gguf

import "testing"

// TestCatalogLooksUpWhatItWasGiven holds the in-memory catalog to answering
// by key and by name as a read file does, so a converter can hand what it is
// about to write to the reader that will consume it, and to refusing a
// duplicate key or name as the reader refuses one.
func TestCatalogLooksUpWhatItWasGiven(t *testing.T) {
	t.Parallel()
	file, err := Catalog(
		[]Metadata{StringMetadata("general.architecture", "clip"), Uint32Metadata("clip.vision.block_count", 24)},
		[]TensorInfo{{Name: "v.patch_embd.weight", Type: DTypeF32, Dimensions: 1, Shape: [MaxDimensions]uint64{7}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	value, found := file.MetadataValue("general.architecture")
	if !found || value.Data != "clip" {
		t.Fatalf("architecture = %v, %t", value.Data, found)
	}
	if _, found := file.MetadataValue("clip.projector_type"); found {
		t.Fatal("a key the catalog never carried was found")
	}
	tensor, found := file.Tensor("v.patch_embd.weight")
	if !found || tensor.Shape[0] != 7 {
		t.Fatalf("tensor = %+v, %t", tensor, found)
	}
	if _, err := Catalog([]Metadata{StringMetadata("k", "a"), StringMetadata("k", "b")}, nil); err == nil {
		t.Fatal("duplicate metadata key accepted")
	}
	if _, err := Catalog(nil, []TensorInfo{{Name: "t"}, {Name: "t"}}); err == nil {
		t.Fatal("duplicate tensor name accepted")
	}
	if _, err := Catalog([]Metadata{StringMetadata("", "a")}, nil); err == nil {
		t.Fatal("empty metadata key accepted")
	}
}
