package testutil

import (
	"context"
	"encoding/binary"
	"io/fs"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
)

// Float32LE encodes float32 values as little-endian bytes for a GGUF/safetensors
// tensor body.
func Float32LE(values []float32) []byte {
	out := make([]byte, len(values)*4)
	for index, value := range values {
		binary.LittleEndian.PutUint32(out[index*4:], math.Float32bits(value))
	}
	return out
}

const (
	wavPCMFormat         = 1
	wavMonoChannels      = 1
	wavPCM16Bits         = 16
	bitsPerByte          = 8
	wavHeaderBytes       = 44
	wavRIFFPayloadBytes  = 36
	wavFormatPayloadSize = 16
)

// ArtifactID: checked fixture identity.
func ArtifactID(t testing.TB, kind artifact.Kind, content string) artifact.ID {
	t.Helper()
	return ArtifactBytesID(t, kind, []byte(content))
}

// ArtifactBytesID: checked binary fixture identity.
func ArtifactBytesID(t testing.TB, kind artifact.Kind, content []byte) artifact.ID {
	t.Helper()
	id, err := artifact.IdentifyBytes(kind, content)
	if err != nil {
		t.Fatal(err)
	}
	return id
}

// PublishArtifact records one identity-only fixture fact.
func PublishArtifact(t testing.TB, repository artifact.Repository, id artifact.ID) {
	t.Helper()
	if _, err := repository.Commit(context.Background(), artifact.Batch{
		Key: "test/artifact/" + id.String(), Artifacts: []artifact.Descriptor{{ID: id}},
	}); err != nil {
		t.Fatal(err)
	}
}

// RepoRoot: nearest parent containing go.mod.
func RepoRoot(t testing.TB) string {
	t.Helper()
	directory, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(directory, "go.mod")); err == nil {
			return directory
		}
		parent := filepath.Dir(directory)
		if parent == directory {
			t.Fatal("testutil: repository root not found")
		}
		directory = parent
	}
}

// FixturePath: repository fixture path.
func FixturePath(t testing.TB, elements ...string) string {
	t.Helper()
	return filepath.Join(append([]string{RepoRoot(t), "fixtures"}, elements...)...)
}

// WriteTextFile writes a repository-relative text fixture, creating parents.
func WriteTextFile(t testing.TB, root, name, content string, permissions ...fs.FileMode) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(name))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	mode := fs.FileMode(0o644)
	if len(permissions) > 0 {
		mode = permissions[0]
	}
	if err := os.WriteFile(path, []byte(content), mode); err != nil {
		t.Fatal(err)
	}
}

type DenseCausalSpec struct {
	Vocab, Hidden, Heads, HeadDim int
	KVHeads, Intermediate, Layers int
	Seed                          int64
}

// DenseCausalWeights: seeded tied-embedding Llama fixture catalog.
func DenseCausalWeights(t testing.TB, spec DenseCausalSpec) (map[string][]float32, map[string][]int) {
	t.Helper()
	if spec.Vocab <= 0 || spec.Hidden <= 0 || spec.Heads <= 0 || spec.HeadDim <= 0 ||
		spec.KVHeads <= 0 || spec.Intermediate <= 0 || spec.Layers <= 0 {
		t.Fatal("testutil: invalid dense causal fixture dimensions")
	}
	random := rand.New(rand.NewSource(spec.Seed))
	weights := make(map[string][]float32)
	shapes := make(map[string][]int)
	add := func(name string, dimensions ...int) {
		elements := 1
		for _, dimension := range dimensions {
			elements *= dimension
		}
		values := make([]float32, elements)
		for index := range values {
			values[index] = float32(random.NormFloat64() * 0.02)
		}
		weights[name], shapes[name] = values, dimensions
	}
	norm := func(name string) {
		values := make([]float32, spec.Hidden)
		for index := range values {
			values[index] = float32(1 + random.NormFloat64()*0.01)
		}
		weights[name], shapes[name] = values, []int{spec.Hidden}
	}
	add("model.embed_tokens.weight", spec.Vocab, spec.Hidden)
	for layer := range spec.Layers {
		prefix := "model.layers." + strconv.Itoa(layer) + "."
		norm(prefix + "input_layernorm.weight")
		add(prefix+"self_attn.q_proj.weight", spec.Heads*spec.HeadDim, spec.Hidden)
		add(prefix+"self_attn.k_proj.weight", spec.KVHeads*spec.HeadDim, spec.Hidden)
		add(prefix+"self_attn.v_proj.weight", spec.KVHeads*spec.HeadDim, spec.Hidden)
		add(prefix+"self_attn.o_proj.weight", spec.Hidden, spec.Heads*spec.HeadDim)
		norm(prefix + "post_attention_layernorm.weight")
		add(prefix+"mlp.gate_proj.weight", spec.Intermediate, spec.Hidden)
		add(prefix+"mlp.up_proj.weight", spec.Intermediate, spec.Hidden)
		add(prefix+"mlp.down_proj.weight", spec.Hidden, spec.Intermediate)
	}
	norm("model.norm.weight")
	return weights, shapes
}

func WriteGGUF(
	t testing.TB,
	path string,
	metadata []gguf.Metadata,
	tensors []gguf.TensorData,
) {
	t.Helper()
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := gguf.Write(file, metadata, tensors, gguf.WriteOptions{}); err != nil {
		_ = file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
}

func TempGGUF(
	t testing.TB,
	name string,
	metadata []gguf.Metadata,
	tensors []gguf.TensorData,
) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	WriteGGUF(t, path, metadata, tensors)
	return path
}

func MonoPCM16WAV(sampleRate uint32, samples []int16) []byte {
	bytesPerSample := wavPCM16Bits / bitsPerByte
	body := make([]byte, len(samples)*bytesPerSample)
	for index, sample := range samples {
		binary.LittleEndian.PutUint16(body[index*bytesPerSample:], uint16(sample))
	}
	return monoWAV(sampleRate, wavPCMFormat, wavPCM16Bits, body)
}

func monoWAV(sampleRate uint32, format, bits uint16, body []byte) []byte {
	bodyBytes := len(body)
	data := make([]byte, 0, wavHeaderBytes+bodyBytes)
	data = append(data, "RIFF"...)
	data = binary.LittleEndian.AppendUint32(data, uint32(wavRIFFPayloadBytes+bodyBytes))
	data = append(data, "WAVE"...)
	data = append(data, "fmt "...)
	data = binary.LittleEndian.AppendUint32(data, wavFormatPayloadSize)
	data = binary.LittleEndian.AppendUint16(data, format)
	data = binary.LittleEndian.AppendUint16(data, wavMonoChannels)
	data = binary.LittleEndian.AppendUint32(data, sampleRate)
	bytesPerSample := bits / bitsPerByte
	data = binary.LittleEndian.AppendUint32(data, sampleRate*uint32(bytesPerSample))
	data = binary.LittleEndian.AppendUint16(data, bytesPerSample)
	data = binary.LittleEndian.AppendUint16(data, bits)
	data = append(data, "data"...)
	data = binary.LittleEndian.AppendUint32(data, uint32(bodyBytes))
	data = append(data, body...)
	return data
}
