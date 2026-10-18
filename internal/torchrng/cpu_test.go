package torchrng

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
	"path/filepath"
	"slices"
	"testing"
)

// TestCPUNormalGolden checks independent Torch outputs across seed and fill boundaries.
func TestCPUNormalGolden(t *testing.T) {
	t.Parallel()
	path := filepath.Join(testutil.RepoRoot(t), "fixtures", "torchrng", "cpu-normal.safetensors")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if hash := fmt.Sprintf("%x", sha256.Sum256(raw)); hash != "3c98167f1812b016c241bfad8db14765689b03fbca9a24691d357cfa611a5cf4" {
		t.Fatalf("reference identity differs: %s", hash)
	}
	source, err := safetensors.OpenFile(path)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	if len(source.Tensors) != 60 || len(source.Metadata) != 1 {
		t.Fatal("reference coverage or provenance differs")
	}
	for _, meta := range source.Metadata {
		if meta["torch"] != "2.12.0+cu126" || meta["producer_sha256"] != "e6c2fd9059b6bff2f8e7d48e066fe771e8716eaf8e70faa699901fe32c730eaf" {
			t.Fatal("reference producer differs")
		}
	}
	seeds := []int64{0, 1, 42, -1, 4294967338, math.MaxInt64}
	counts := []int{16, 17, 31, 32, 33, 623, 624, 625, 1001, 4096}
	for _, seed := range seeds {
		for _, count := range counts {
			name := fmt.Sprintf("seed.%d.count.%d", seed, count)
			t.Run(name, func(t *testing.T) {
				value, ok := source.Tensors[name]
				if !ok || value.DType != "F32" || !slices.Equal(value.Shape, []uint64{uint64(count)}) {
					t.Fatal("missing exact reference geometry")
				}
				want, err := io.ReadAll(value.Reader())
				if err != nil {
					t.Fatal(err)
				}
				got, err := CPUNormal(seed, count)
				if err != nil {
					t.Fatal(err)
				}
				if len(got) != count || len(want) != count*4 {
					t.Fatal("noise length differs")
				}
				for i, x := range got {
					if bits := binary.LittleEndian.Uint32(want[i*4:]); math.Float32bits(x) != bits {
						t.Fatalf("value %d got bits %08x want %08x", i, math.Float32bits(x), bits)
					}
				}
			})
		}
	}
}

func TestCPUNormalRefusesScalarFill(t *testing.T) {
	t.Parallel()
	for _, count := range []int{math.MinInt, -1, 0, 1, 8, 15} {
		if values, err := CPUNormal(42, count); err == nil || values != nil {
			t.Fatalf("unsupported scalar fill %d accepted", count)
		}
	}
}
