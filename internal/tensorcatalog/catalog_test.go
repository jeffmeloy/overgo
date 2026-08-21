package tensorcatalog

import (
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/tensor/dtype"
)

func TestValidateCatalog(t *testing.T) {
	info := gguf.TensorInfo{Name: "blk.weight", Dimensions: 2, Shape: [gguf.MaxDimensions]uint64{2, 3}}
	if err := Validate(
		map[string]gguf.TensorInfo{"blk.weight": info}, "blk.",
		[]Requirement{{Name: "weight", Shapes: [][]uint64{{2, 3}}}},
	); err != nil {
		t.Fatal(err)
	}
	if err := ValidateInfo(info, Requirement{Name: info.Name, Shapes: [][]uint64{{3, 2}}}); err == nil {
		t.Fatal("invalid shape accepted")
	}
	info.Type = dtype.F16
	if err := ValidateInfo(info, Requirement{
		Name: info.Name, Shapes: [][]uint64{{3, 2}, {2, 3}}, Storages: []dtype.Type{dtype.F32, dtype.F16},
	}); err != nil {
		t.Fatal(err)
	}
}

func TestValidateRelationalRequirement(t *testing.T) {
	requirement := Requirement{Rank: 1, NonEmpty: true, Storages: []dtype.Type{dtype.I64}}
	valid := gguf.TensorInfo{Name: "mapping", Type: dtype.I64, Dimensions: 1, Shape: [gguf.MaxDimensions]uint64{3}}
	if err := ValidateInfo(valid, requirement); err != nil {
		t.Fatal(err)
	}
	for _, invalid := range []gguf.TensorInfo{
		{Name: "mapping", Type: dtype.F32, Dimensions: 1, Shape: [gguf.MaxDimensions]uint64{3}},
		{Name: "mapping", Type: dtype.I64, Dimensions: 2, Shape: [gguf.MaxDimensions]uint64{3, 1}},
		{Name: "mapping", Type: dtype.I64, Dimensions: 1},
	} {
		if err := ValidateInfo(invalid, requirement); err == nil {
			t.Fatalf("invalid relation accepted: %+v", invalid)
		}
	}
}

func TestValidateAllowedRanks(t *testing.T) {
	requirement := Requirement{Ranks: []uint32{1, 3}, NonEmpty: true}
	for _, rank := range []uint32{1, 3} {
		info := gguf.TensorInfo{Name: "value", Dimensions: rank}
		for axis := range rank {
			info.Shape[axis] = 1
		}
		if err := ValidateInfo(info, requirement); err != nil {
			t.Fatal(err)
		}
	}
	if err := ValidateInfo(
		gguf.TensorInfo{Name: "value", Dimensions: 2, Shape: [gguf.MaxDimensions]uint64{1, 1}}, requirement,
	); err == nil {
		t.Fatal("unlisted rank accepted")
	}
}

func TestShapeAndIndexedCount(t *testing.T) {
	shapes := map[string][]int{
		"layer.0.weight": {2, 3},
		"layer.1.weight": {2, 3},
	}
	shape, err := Shape(shapes, "layer.0.weight", 2)
	if err != nil || shape[0] != 2 || shape[1] != 3 {
		t.Fatalf("shape = %v, %v", shape, err)
	}
	count, err := IndexedCount(shapes, "layer.", ".weight")
	if err != nil || count != 2 {
		t.Fatalf("count = %d, %v", count, err)
	}
	delete(shapes, "layer.0.weight")
	if _, err := IndexedCount(shapes, "layer.", ".weight"); err == nil {
		t.Fatal("nonzero starting index accepted")
	}
}
