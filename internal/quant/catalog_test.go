package quant

import (
	"slices"
	"testing"

	"llamacpp2go/internal/tensor/dtype"
)

func TestCodecCatalogOwnsSupportedTypePolicy(t *testing.T) {
	types := Types()
	names := TypeNames()
	if len(types) == 0 || len(types) != len(names) {
		t.Fatalf("catalog types = %d, names = %d", len(types), len(names))
	}
	seen := make(map[dtype.Type]struct{}, len(types))
	for index, dataType := range types {
		if _, duplicate := seen[dataType]; duplicate {
			t.Fatalf("duplicate type %s", dataType)
		}
		seen[dataType] = struct{}{}
		if !CanQuantize(dataType) {
			t.Fatalf("catalog type %s is not quantizable", dataType)
		}
		parsed, ok := ParseType(names[index])
		if !ok || parsed != dataType {
			t.Fatalf("ParseType(%q) = %s, %t; want %s", names[index], parsed, ok, dataType)
		}
	}
	if !RequiresImportance(dtype.IQ2XXS) || RequiresImportance(dtype.Q4_0) {
		t.Fatal("importance policy mismatch")
	}
	if _, ok := ParseType("unknown"); ok || CanQuantize(dtype.I8) {
		t.Fatal("unsupported type accepted")
	}
	cloned := Types()
	cloned[0] = dtype.I8
	if slices.Equal(cloned, Types()) {
		t.Fatal("Types returned shared catalog storage")
	}
}
