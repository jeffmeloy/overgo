package testutil

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"llamacpp2go/internal/gguf"
)

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
	body := make([]byte, len(samples)*2)
	for index, sample := range samples {
		binary.LittleEndian.PutUint16(body[index*2:], uint16(sample))
	}
	return monoWAV(sampleRate, 1, 16, body)
}

func MonoFloat32WAV(sampleRate uint32, samples []float32) []byte {
	body := make([]byte, len(samples)*4)
	for index, sample := range samples {
		binary.LittleEndian.PutUint32(body[index*4:], math.Float32bits(sample))
	}
	return monoWAV(sampleRate, 3, 32, body)
}

func monoWAV(sampleRate uint32, format, bits uint16, body []byte) []byte {
	bodyBytes := len(body)
	data := make([]byte, 44+bodyBytes)
	copy(data[0:4], "RIFF")
	binary.LittleEndian.PutUint32(data[4:8], uint32(36+bodyBytes))
	copy(data[8:12], "WAVE")
	copy(data[12:16], "fmt ")
	binary.LittleEndian.PutUint32(data[16:20], 16)
	binary.LittleEndian.PutUint16(data[20:22], format)
	binary.LittleEndian.PutUint16(data[22:24], 1)
	binary.LittleEndian.PutUint32(data[24:28], sampleRate)
	bytesPerSample := bits / 8
	binary.LittleEndian.PutUint32(data[28:32], sampleRate*uint32(bytesPerSample))
	binary.LittleEndian.PutUint16(data[32:34], bytesPerSample)
	binary.LittleEndian.PutUint16(data[34:36], bits)
	copy(data[36:40], "data")
	binary.LittleEndian.PutUint32(data[40:44], uint32(bodyBytes))
	copy(data[44:], body)
	return data
}
