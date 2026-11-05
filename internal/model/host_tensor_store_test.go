package model

import (
	"bytes"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/testcheck"
)

func TestHostTensorStoreRetainsLoadedTensor(t *testing.T) {
	data := hostTensorFixture(t, "weight", 4)
	file, err := gguf.Parse(bytes.NewReader(data), uint64(len(data)), gguf.DefaultOptions())
	testcheck.NoError(t, err)
	store := NewHostTensorStore()
	first, err := store.Load(t.Context(), file, file.Tensors[0])
	testcheck.NoError(t, err)
	second, err := store.Load(t.Context(), file, file.Tensors[0])
	testcheck.NoError(t, err)
	if len(first.Data) == 0 || &first.Data[0] != &second.Data[0] {
		t.Fatal("host tensor store did not reuse retained storage")
	}
	store.Release()
	third, err := store.Load(t.Context(), file, file.Tensors[0])
	testcheck.NoError(t, err)
	if &first.Data[0] == &third.Data[0] {
		t.Fatal("released host tensor storage was reused")
	}
}
