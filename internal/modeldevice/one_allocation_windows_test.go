//go:build windows

package modeldevice

import (
	"errors"
	"math/rand/v2"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
)

// TestAStoreHoldsItsTensorsInOneAllocation loads a fixture onto the card: the driver's ledger counts one allocation for
// each store the load fills, not one a tensor, every tensor lies at an aligned offset in its store's allocation and
// reads back as the file holds it, and closing the stores leaves the ledger empty.
func TestAStoreHoldsItsTensorsInOneAllocation(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	source := mixedFixture(t, rand.New(rand.NewPCG(41, 42)))
	loaded := newDeviceStores(t)
	before := testcheck.Must(loaded.worker.MemoryStats(t.Context()))(t)
	file := source.parse(t)
	loader := NewLoader(file)
	source.route(loader, loaded.stores, file)
	testcheck.NoError(t, loader.Load(t.Context(), nil))
	after := testcheck.Must(loaded.worker.MemoryStats(t.Context()))(t)
	if routed := source.check(t, file, loaded.stores, loaded.download); routed != loaded.loaded() {
		t.Fatalf("stores hold %d tensors, want the %d routed", loaded.loaded(), routed)
	}
	filled := 0
	for _, store := range []*deviceTensorStore{loaded.raw.deviceTensorStore, loaded.f32.deviceTensorStore, loaded.bf16.deviceTensorStore} {
		if len(store.tensors) == 0 {
			continue
		}
		filled++
		if len(store.blocks) != 1 {
			t.Fatalf("%s store holds its %d tensors in %d allocations, want one", store.kind, len(store.tensors), len(store.blocks))
		}
		for name, value := range store.tensors {
			if offset := value.Pointer - store.blocks[0]; value.Pointer < store.blocks[0] || offset%residentAlignment != 0 {
				t.Fatalf("%s tensor %q lies at %#x, not at an aligned offset in its store's allocation at %#x", store.kind, name, uint64(value.Pointer), uint64(store.blocks[0]))
			}
		}
	}
	if made := after.Allocations - before.Allocations; made != uint64(filled) {
		t.Fatalf("the load made %d allocations for %d filled stores, want one each", made, filled)
	}
	testcheck.NoError(t, errors.Join(loaded.raw.Close(), loaded.f32.Close(), loaded.bf16.Close()))
	if got := loaded.currentBytes(t); got != 0 {
		t.Fatalf("after closing the stores the ledger holds %d bytes", got)
	}
}
