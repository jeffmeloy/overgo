//go:build windows

package driver

import (
	"runtime"
	"testing"
	"unsafe"
)

func TestReadCString(t *testing.T) {
	data := []byte("cuda error\x00suffix")
	pointer := uintptr(unsafe.Pointer(&data[0]))
	if got := ReadCString(pointer, 256); got != "cuda error" {
		t.Fatalf("string = %q", got)
	}
	if got := ReadCString(pointer, 4); got != "cuda" {
		t.Fatalf("bounded string = %q", got)
	}
	runtime.KeepAlive(data)
}
