//go:build windows

package routedlm

import (
	"bytes"
	"testing"

	"overgo/internal/cuda/driver"
)

func TestBF16MatrixDeviceBytesPreserveStorage(t *testing.T) {
	raw := []byte{1, 2, 3, 4}
	if got := bf16MatrixBytes(BF16Matrix{Raw: raw, In: 2, Out: 1}); !bytes.Equal(got, raw) {
		t.Fatalf("raw device bytes=%v", got)
	}
	words := []uint16{0x1234, 0xabcd}
	if got := bf16MatrixBytes(BF16Matrix{Data: words, In: 2, Out: 1}); !bytes.Equal(got, driver.Bytes(words)) {
		t.Fatalf("decoded device bytes=%v", got)
	}
}
