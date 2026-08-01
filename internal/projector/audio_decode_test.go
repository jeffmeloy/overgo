package projector

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestDecodeWAV(t *testing.T) {
	tests := []struct {
		name   string
		format uint16
		bits   uint16
		body   []byte
		want   []float32
	}{
		{name: "pcm16", format: wavPCM, bits: 16, body: []byte{0, 128, 0, 0, 255, 127}, want: []float32{-1, 0, 32767.0 / 32768}},
		{name: "float32", format: wavIEEEFloat, bits: 32, body: float32Bytes(-0.5, 0.25), want: []float32{-0.5, 0.25}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			input := tinyWAV(test.format, test.bits, test.body)
			got, rate, err := DecodeWAV(input)
			if err != nil {
				t.Fatal(err)
			}
			if rate != 16000 || len(got) != len(test.want) {
				t.Fatalf("rate=%d samples=%v", rate, got)
			}
			for index, value := range test.want {
				if got[index] != value {
					t.Fatalf("sample[%d] = %g, want %g", index, got[index], value)
				}
			}
		})
	}
}

func TestDecodeFloat32LERejectsPartialSample(t *testing.T) {
	if _, err := DecodeFloat32LE([]byte{1, 2, 3}); err == nil {
		t.Fatal("expected partial-sample error")
	}
}

func tinyWAV(format, bits uint16, body []byte) []byte {
	result := make([]byte, 44+len(body))
	copy(result[0:4], "RIFF")
	binary.LittleEndian.PutUint32(result[4:8], uint32(len(result)-8))
	copy(result[8:12], "WAVE")
	copy(result[12:16], "fmt ")
	binary.LittleEndian.PutUint32(result[16:20], 16)
	binary.LittleEndian.PutUint16(result[20:22], format)
	binary.LittleEndian.PutUint16(result[22:24], 1)
	binary.LittleEndian.PutUint32(result[24:28], 16000)
	bytesPerSample := bits / 8
	binary.LittleEndian.PutUint32(result[28:32], uint32(16000*bytesPerSample))
	binary.LittleEndian.PutUint16(result[32:34], bytesPerSample)
	binary.LittleEndian.PutUint16(result[34:36], bits)
	copy(result[36:40], "data")
	binary.LittleEndian.PutUint32(result[40:44], uint32(len(body)))
	copy(result[44:], body)
	return result
}

func float32Bytes(values ...float32) []byte {
	result := make([]byte, len(values)*4)
	for index, value := range values {
		binary.LittleEndian.PutUint32(result[index*4:], math.Float32bits(value))
	}
	return result
}
