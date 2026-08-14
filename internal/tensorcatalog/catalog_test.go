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
