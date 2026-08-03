package tensorcatalog

import (
	"testing"

	"llamacpp2go/internal/gguf"
)

func TestValidateCatalog(t *testing.T) {
	info := gguf.TensorInfo{Name: "blk.weight", Dimensions: 2, Shape: [gguf.MaxDimensions]uint64{2, 3}}
	if err := Validate(
		map[string]gguf.TensorInfo{"blk.weight": info}, "blk.",
		[]Requirement{{Name: "weight", Shape: []uint64{2, 3}}},
	); err != nil {
		t.Fatal(err)
	}
	if err := ValidateInfo(info, Requirement{Name: info.Name, Shape: []uint64{3, 2}}); err == nil {
		t.Fatal("invalid shape accepted")
	}
}
