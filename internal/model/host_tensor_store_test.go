package model

import (
	"bytes"
	"context"
	"testing"

	"overgo/internal/gguf"
)

func TestHostTensorStoreRetainsLoadedTensor(t *testing.T) {
	data := hostTensorFixture(t)
	file, err := gguf.Parse(bytes.NewReader(data), uint64(len(data)), gguf.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	store := NewHostTensorStore()
	first, err := store.Load(context.Background(), file, file.Tensors[0])
	if err != nil {
		t.Fatal(err)
	}
	second, err := store.Load(context.Background(), file, file.Tensors[0])
	if err != nil {
		t.Fatal(err)
	}
	if len(first.Data) == 0 || &first.Data[0] != &second.Data[0] {
		t.Fatal("host tensor store did not reuse retained storage")
	}
	store.Release()
	third, err := store.Load(context.Background(), file, file.Tensors[0])
	if err != nil {
		t.Fatal(err)
	}
	if &first.Data[0] == &third.Data[0] {
		t.Fatal("released host tensor storage was reused")
	}
}
