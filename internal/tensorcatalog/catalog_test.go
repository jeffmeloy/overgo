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
