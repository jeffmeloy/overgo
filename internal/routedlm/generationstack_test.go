//go:build windows

package routedlm

import (
	"bytes"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func TestSenseNovaRetainedPrefixDirectBinding(t *testing.T) {
	b := tensor.NewBuilder()
	graph := &DeviceGenerationLayerGraph{
		Row:         b.Input("row", dtype.F32, tensor.MustShape(1)),
		PrefixKey:   b.Input("prefix-key", dtype.F32, tensor.MustShape(1)),
		PrefixValue: b.Input("prefix-value", dtype.F32, tensor.MustShape(1)),
		Vision:      DevicePrefillBranch{},
	}
	branch := &generationStackBranch{
		graph:      graph,
		prefixKeys: []driver.DevicePtr{11, 12}, prefixValues: []driver.DevicePtr{21, 22},
		row: 31,
	}
	feeds := bindGenerationBranch(branch, branchDeviceWeights{}, 1)
	if feeds[graph.PrefixKey] != 12 || feeds[graph.PrefixValue] != 22 || feeds[graph.Row] != 31 {
		t.Fatalf("generation feeds key=%d value=%d row=%d", feeds[graph.PrefixKey], feeds[graph.PrefixValue], feeds[graph.Row])
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
