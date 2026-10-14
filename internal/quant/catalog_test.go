package quant

import (
	"testing"

	"overgo/internal/tensor/dtype"
)

func TestCodecCatalogOwnsSupportedTypePolicy(t *testing.T) {
	t.Parallel()
	names := TypeNames()
	if len(codecCatalog) == 0 || len(codecCatalog) != len(names) {
		t.Fatalf("catalog types = %d, names = %d", len(codecCatalog), len(names))
	}
	seen := make(map[dtype.Type]struct{}, len(codecCatalog))
	for index, codec := range codecCatalog {
		dataType := codec.dataType
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
}
