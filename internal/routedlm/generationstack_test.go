//go:build windows

package routedlm

import (
	"bytes"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func TestGenerationRetainedPrefixDirectBinding(t *testing.T) {
	b := tensor.NewBuilder()
	row := b.Input("row", dtype.F32, tensor.MustShape(1))
	prefixKey := b.Input("prefix-key", dtype.F32, tensor.MustShape(1))
	prefixValue := b.Input("prefix-value", dtype.F32, tensor.MustShape(1))
	output := b.Add(b.Add(row, prefixKey), prefixValue)
	indexed, err := executor.CompileIndexed(output)
	if err != nil {
		t.Fatal(err)
	}
	rowSlot, _ := indexed.Graph.InputSlot(row)
	keySlot, _ := indexed.Graph.InputSlot(prefixKey)
	valueSlot, _ := indexed.Graph.InputSlot(prefixValue)
	graph := &DeviceGenerationLayerGraph{
		Row: row, PrefixKey: prefixKey, PrefixValue: prefixValue,
		Vision: DevicePrefillBranch{},
	}
	branch := &generationStackBranch{
		IndexedGraph: indexed, graph: graph,
		prefixKey: keySlot, prefixValue: valueSlot,
		prefixKeys: []driver.DevicePtr{11, 12}, prefixValues: []driver.DevicePtr{21, 22},
		row: 31,
	}
	branch.Inputs.Pointers[rowSlot] = branch.row
	branch.bindPrefix(1)
	if branch.Inputs.Pointers[keySlot] != 12 || branch.Inputs.Pointers[valueSlot] != 22 || branch.Inputs.Pointers[rowSlot] != 31 {
		t.Fatalf("generation inputs=%v", branch.Inputs.Pointers)
	}
}

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
