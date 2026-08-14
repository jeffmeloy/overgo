//go:build windows && integration

package quant

import (
	"bytes"
	"os"
	"runtime"
	"syscall"
	"testing"
	"unsafe"

	"overgo/internal/tensor/dtype"
)

func TestQuantizeWeightedMatchesPinnedGGML(t *testing.T) {
	path := os.Getenv("OVERGO_GGML_BASE_ORACLE")
	if path == "" {
		t.Skip("set OVERGO_GGML_BASE_ORACLE to pinned ggml-base.dll")
	}
	library := syscall.NewLazyDLL(path)
	initialize := library.NewProc("ggml_quantize_init")
	values := quantizeOracleValues(256)
	weights := make([]float32, len(values))
	for index := range weights {
		weights[index] = 0.25 + float32((index*29)%37)/11
	}
	for _, dataType := range []dtype.Type{dtype.IQ2XXS, dtype.IQ2XS, dtype.IQ1S, dtype.IQ1M} {
		t.Run(dataType.String(), func(t *testing.T) {
			if result, _, callErr := initialize.Call(uintptr(dataType)); result != 0 {
				t.Fatalf("ggml_quantize_init result/error = %d/%v", result, callErr)
			}
			got, err := QuantizeWeighted(dataType, values, weights)
			if err != nil {
				t.Fatal(err)
			}
			want := make([]byte, len(got))
			function := library.NewProc("quantize_" + dataType.String())
			written, _, callErr := function.Call(
				uintptr(unsafe.Pointer(&values[0])),
				uintptr(unsafe.Pointer(&want[0])),
				1,
				256,
				uintptr(unsafe.Pointer(&weights[0])),
			)
			runtime.KeepAlive(values)
			runtime.KeepAlive(weights)
			runtime.KeepAlive(want)
			if int(written) != len(want) {
				t.Fatalf("oracle bytes/error = %d/%v, want %d", written, callErr, len(want))
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("weighted bytes differ\n got: %x\nwant: %x", got, want)
			}
			t.Logf("SHA256 %s", hashHex(got))
		})
	}
}
