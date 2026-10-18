package hfgguf

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/safetensors"
)

func TestMOSSTransformGolden(t *testing.T) {
	root := filepath.Join("..", "..", "fixtures", "moss")
	path := filepath.Join(root, "import-transforms.safetensors")
	encoded, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(encoded)) != "f568ae58eb27427734a9e8d31fb19944b152aeedc41e6e6fe13039291b79b88f" {
		t.Fatal("upstream golden changed identity")
	}
	source, err := safetensors.OpenFile(path)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	if len(source.Metadata) != 1 {
		t.Fatal("missing provenance")
	}
	var metadata map[string]string
	for _, entry := range source.Metadata {
		metadata = entry
	}
	for key, want := range map[string]string{"source_commit": "797fa8e678b4f0a766271bf5103e6cd7a2faa999", "package_torch": "2.12.0+cu126", "package_transformers": "5.14.1", "package_safetensors": "0.8.0", "device": "cpu"} {
		if metadata[key] != want {
			t.Fatalf("provenance %s = %q", key, metadata[key])
		}
	}
	groups, err := strconv.ParseUint(metadata["groups"], 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	heads, err := strconv.ParseUint(metadata["heads"], 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	var names map[string]string
	if err := json.Unmarshal([]byte(metadata["source_tensors"]), &names); err != nil {
		t.Fatal(err)
	}
	contract, err := os.ReadFile(filepath.Join(root, "oracle-contract.json"))
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(contract)) != "0c2767456a75075671de88538d8c1ce0f673ef41c2bae897c4e59449fad0ca86" {
		t.Fatal("frozen acceptance changed")
	}
	var acceptance struct {
		Comparison struct {
			DTypes map[string]struct {
				ATol float64 `json:"atol"`
				RTol float64 `json:"rtol"`
			} `json:"dtype_tolerances"`
		} `json:"comparison"`
	}
	if err := json.Unmarshal(contract, &acceptance); err != nil {
		t.Fatal(err)
	}
	tolerance, ok := acceptance.Comparison.DTypes["F32"]
	if !ok {
		t.Fatal("missing F32 tolerance")
	}
	// Norm and decay are scalar vectors, so each head block contains one scalar.
	transforms := qwen35TensorTransforms(qwen35LinearDims{keyHeads: groups, valueHeads: heads, valueWidth: 1})
	for _, name := range []string{"norm", "decay"} {
		t.Run(name, func(t *testing.T) {
			input, ok := source.Tensors[name+".source"]
			if !ok {
				t.Fatal("missing source")
			}
			expected, ok := source.Tensors[name+".expected"]
			if !ok || expected.DType != "F32" {
				t.Fatal("missing F32 output")
			}
			input.Name = names[name]
			mapped, err := mappedTensorData([]tensorMapping{{name: name, tensor: input, shape: input.Shape, storage: gguf.DTypeF32, transforms: transforms(input.Name)}})
			if err != nil {
				t.Fatal(err)
			}
			actual, err := io.ReadAll(mapped[0].Data)
			if err != nil {
				t.Fatal(err)
			}
			want, err := io.ReadAll(expected.Reader())
			if err != nil {
				t.Fatal(err)
			}
			if len(actual) != len(want) {
				t.Fatalf("payload size %d != %d", len(actual), len(want))
			}
			unequal := 0
			for i := 0; i < len(want); i += 4 {
				a := float64(math.Float32frombits(binary.LittleEndian.Uint32(actual[i:])))
				w := float64(math.Float32frombits(binary.LittleEndian.Uint32(want[i:])))
				if a != w {
					unequal++
				}
				if math.IsNaN(a) || math.IsInf(a, 0) || math.Abs(a-w) > tolerance.ATol+tolerance.RTol*math.Abs(w) {
					t.Fatalf("value %d = %g, upstream %g", i/4, a, w)
				}
			}
			t.Logf("%d upstream values pass frozen F32 tolerance; %d differ bitwise", len(want)/4, unequal)
		})
	}
}
