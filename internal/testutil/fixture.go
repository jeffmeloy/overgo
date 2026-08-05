package testutil

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"llamacpp2go/internal/gguf"
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
