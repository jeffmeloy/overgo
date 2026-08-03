package projector

import (
	"testing"

	"llamacpp2go/internal/testutil"
)

func TestDecodeWAV(t *testing.T) {
	tests := []struct {
		name  string
		input []byte
		want  []float32
	}{
		{name: "pcm16", input: testutil.MonoPCM16WAV(16000, []int16{-32768, 0, 32767}), want: []float32{-1, 0, 32767.0 / 32768}},
		{name: "float32", input: testutil.MonoFloat32WAV(16000, []float32{-0.5, 0.25}), want: []float32{-0.5, 0.25}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, rate, err := DecodeWAV(test.input)
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
