package latentimage

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/safetensors"
)

const mossReferenceCommit = "797fa8e678b4f0a766271bf5103e6cd7a2faa999"
const mossReferenceSource = "1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31"

func mossFixtureBytes(t *testing.T, name, digest string) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "fixtures", "moss", name))
	if err != nil {
		t.Fatal(err)
	}
	actual := sha256.Sum256(data)
	if hex.EncodeToString(actual[:]) != digest {
		t.Fatalf("fixture %s changed identity", name)
	}
	return data
}

func mossGolden(t *testing.T, name string) *safetensors.Source {
	t.Helper()
	digest := ""
	switch name {
	case "primitives":
		digest = "f55669402c6d12d20d047a461e90fa9e2da3f8bdb7b3d55cc189cc85f84b2160"
	case "tokenizer":
		digest = "78cb6fb6b261f4cb61405bace3a19815f0a535a0ece4a34e4183a905fb8563aa"
	default:
		t.Fatalf("unknown golden %q", name)
	}
	mossFixtureBytes(t, name+".safetensors", digest)
	source, err := safetensors.OpenFile(filepath.Join("..", "..", "fixtures", "moss", name+".safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := source.Close(); err != nil {
			t.Error(err)
		}
	})
	if len(source.Metadata) != 1 {
		t.Fatal("fixture must bind one provenance header")
	}
	for _, metadata := range source.Metadata {
		if metadata["source_commit"] != mossReferenceCommit || metadata["source_sha256"] != mossReferenceSource || metadata["source_repository"] != "https://www.modelscope.cn/openmoss/MOSS-Image-SFT.git" {
			t.Fatal("fixture lacks pinned SFT provenance")
		}
		for field, want := range map[string]string{"package_torch": "2.12.0+cu126", "package_transformers": "5.14.1", "package_safetensors": "0.8.0"} {
			if metadata[field] != want {
				t.Fatalf("oracle %s=%s want %s", field, metadata[field], want)
			}
		}
	}
	return source
}

func mossF32(t *testing.T, source *safetensors.Source, name string) []float32 {
	t.Helper()
	tensor, found := source.Tensors[name]
	if !found || tensor.DType != "F32" {
		t.Fatalf("missing F32 golden %s", name)
	}
	data, err := io.ReadAll(tensor.Reader())
	if err != nil {
		t.Fatal(err)
	}
	values := make([]float32, len(data)/4)
	for index := range values {
		values[index] = math.Float32frombits(binary.LittleEndian.Uint32(data[index*4:]))
	}
	return values
}

// TestMOSSGoldenProvenance holds portable upstream fixtures and their component identities.
// Full model parity is evaluated separately against the stored full-resolution references.
func TestMOSSGoldenProvenance(t *testing.T) {
	var receipt struct {
		Source   string `json:"source"`
		Revision string `json:"revision"`
		Files    []struct {
			Path  string `json:"path"`
			SHA   string `json:"sha256"`
			Bytes int64  `json:"bytes"`
		} `json:"files"`
	}
	data := mossFixtureBytes(t, "checkpoint.json", "0d9b600f6232957565c26e86c9f9b0d19411b87fe97d665bbe7b93bd7770a9db")
	if err := json.Unmarshal(data, &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Source != "https://www.modelscope.cn/openmoss/MOSS-Image-SFT.git" || receipt.Revision != mossReferenceCommit || len(receipt.Files) != 37 {
		t.Fatal("incomplete checkpoint inventory")
	}
	files := map[string]string{}
	for _, file := range receipt.Files {
		if file.Bytes < 0 || files[file.Path] != "" {
			t.Fatal("invalid checkpoint file")
		}
		files[file.Path] = file.SHA
	}
	if files["pipeline_moss_image.py"] != mossReferenceSource {
		t.Fatal("source differs from checkpoint")
	}
	t.Run("primitives", func(t *testing.T) {
		source := mossGolden(t, "primitives")
		if len(source.Tensors) != 96 {
			t.Fatal("incomplete primitive coverage")
		}
		for _, meta := range source.Metadata {
			if meta["producer_sha256"] != "c041bf2d6926f35bd4660ec6801cfae0dd2b71f20fb02cdb7c70df92ea90140d" || meta["device"] != "cpu" || meta["seed"] != "42" || !strings.Contains(meta["scope"], "Synthetic") {
				t.Fatal("primitive provenance differs")
			}
		}
		for name, tensor := range source.Tensors {
			if tensor.DType == "BOOL" {
				continue
			}
			for _, value := range mossF32(t, source, name) {
				if math.IsInf(float64(value), 0) || math.IsNaN(float64(value)) {
					t.Fatalf("nonfinite fixture %s", name)
				}
			}
		}
		for requested, actual := range map[int]int{1: 1, 2: 2, 50: 50, 1001: 616} {
			timesteps := source.Tensors[fmt.Sprintf("schedule.%d.timesteps", requested)]
			if !slices.Equal(timesteps.Shape, []uint64{uint64(actual)}) {
				t.Fatalf("schedule %d shape %v", requested, timesteps.Shape)
			}
			sigmas := mossF32(t, source, fmt.Sprintf("schedule.%d.sigmas", requested))
			if len(sigmas) != actual+1 || sigmas[0] != 1 || sigmas[actual] != 0 {
				t.Fatal("schedule boundaries differ")
			}
			for i := 1; i < len(sigmas); i++ {
				if sigmas[i] >= sigmas[i-1] {
					t.Fatal("snapped schedule is not strictly descending")
				}
			}
		}
	})
	t.Run("tokenizer", func(t *testing.T) {
		source := mossGolden(t, "tokenizer")
		if len(source.Tensors) != 12 {
			t.Fatal("incomplete tokenizer coverage")
		}
		for _, meta := range source.Metadata {
			if meta["producer_sha256"] != "853e49d6b55e722d98510e3e9ac95b005f2a0b59876ce1689e0b60a5e4e8b135" || meta["tokenizer_class"] != "Qwen2Tokenizer" || meta["automatic_class"] != "Qwen2Tokenizer" {
				t.Fatal("tokenizer provenance differs")
			}
			var tokenizerFiles map[string]string
			if err := json.Unmarshal([]byte(meta["tokenizer_files"]), &tokenizerFiles); err != nil {
				t.Fatal(err)
			}
			if len(tokenizerFiles) != 3 {
				t.Fatal("missing tokenizer dependencies")
			}
			for name, digest := range tokenizerFiles {
				if files["tokenizer/"+name] != digest {
					t.Fatalf("tokenizer source %s differs", name)
				}
			}
			var prompts map[string]string
			if err := json.Unmarshal([]byte(meta["prompts"]), &prompts); err != nil {
				t.Fatal(err)
			}
			var layout struct {
				Summary struct {
					Caption string `json:"caption"`
				} `json:"summary"`
				Items  []json.RawMessage `json:"items"`
				Format string            `json:"bbox_format"`
				Space  string            `json:"bbox_coord_space"`
			}
			if err := json.Unmarshal([]byte(prompts["layout"]), &layout); err != nil {
				t.Fatal(err)
			}
			if layout.Summary.Caption == "" || len(layout.Items) != 2 || layout.Format != "yxyx" || layout.Space != "abs_centralized" {
				t.Fatal("layout fixture must use documented format")
			}
		}
		for name, count := range map[string]int{"english": 22, "chinese": 22, "empty": 10, "whitespace": 20, "layout": 151, "truncation": 4096} {
			for _, suffix := range []string{"input_ids", "attention_mask"} {
				tensor, found := source.Tensors[name+"."+suffix]
				if !found || tensor.DType != "I64" || !slices.Equal(tensor.Shape, []uint64{1, uint64(count)}) {
					t.Fatalf("invalid token fixture %s.%s", name, suffix)
				}
				raw, err := io.ReadAll(tensor.Reader())
				if err != nil {
					t.Fatal(err)
				}
				for i := 0; i < len(raw); i += 8 {
					value := int64(binary.LittleEndian.Uint64(raw[i:]))
					if value < 0 || suffix == "attention_mask" && value != 1 {
						t.Fatalf("invalid unpadded token/mask %s", name)
					}
				}
			}
		}
	})
}
