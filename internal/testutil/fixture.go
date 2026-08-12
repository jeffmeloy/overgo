package testutil

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
)

const (
	wavPCMFormat         = 1
	wavIEEEFloatFormat   = 3
	wavMonoChannels      = 1
	wavPCM16Bits         = 16
	wavFloat32Bits       = 32
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

func MonoFloat32WAV(sampleRate uint32, samples []float32) []byte {
	bytesPerSample := wavFloat32Bits / bitsPerByte
	body := make([]byte, len(samples)*bytesPerSample)
	for index, sample := range samples {
		binary.LittleEndian.PutUint32(body[index*bytesPerSample:], math.Float32bits(sample))
	}
	return monoWAV(sampleRate, wavIEEEFloatFormat, wavFloat32Bits, body)
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
