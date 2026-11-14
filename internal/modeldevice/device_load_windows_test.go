//go:build windows

package modeldevice

import (
	"errors"
	"math/rand/v2"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
)

type deviceStores struct {
	worker *device.Worker
	stores
}

func newDeviceStores(t *testing.T) deviceStores {
	t.Helper()
	worker := testcheck.Must(device.New(modelTestDeviceOrdinal))(t)
	t.Cleanup(func() { _ = worker.Close() })
	raw := testcheck.Must(NewDeviceWeights(worker))(t)
	f32 := testcheck.Must(NewDeviceConvertedWeights(worker, dtype.F32))(t)
	bf16 := testcheck.Must(NewDeviceConvertedWeights(worker, dtype.BF16))(t)
	return deviceStores{worker: worker, stores: stores{raw: raw, f32: f32, bf16: bf16}}
}

// download reads a resident tensor back; a raw tensor is Info.Size bytes, a
// converted one Size.
func (d deviceStores) download(t *testing.T, value DeviceTensor) []byte {
	t.Helper()
	size := value.Size
	if size == 0 {
		size = value.Info.Size
	}
	data := make([]byte, size)
	err := d.worker.Do(t.Context(), func(state *device.State) error {
		return state.Driver.MemcpyDtoH(data, value.Pointer)
	})
	testcheck.NoError(t, err)
	return data
}

func (d deviceStores) currentBytes(t *testing.T) uint64 {
	t.Helper()
	stats := testcheck.Must(d.worker.MemoryStats(t.Context()))(t)
	return stats.CurrentBytes
}

// TestLoaderUploadsVerifiedBytesToTheDevice loads a fixture through the real
// driver with its claim, and finds on the device the bytes the file holds; the
// same file changed after its claim is refused, with the driver's own ledger
// back at what it was before the pass.
func TestLoaderUploadsVerifiedBytesToTheDevice(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	random := rand.New(rand.NewPCG(31, 32))
	source := mixedFixture(t, random)
	claim := []artifact.ID{identity(t, source.data)}

	refused := newDeviceStores(t)
	baseline := refused.currentBytes(t)
	altered := fixture{data: slices.Clone(source.data), tensors: source.tensors, payloads: source.payloads}
	parsed := altered.parse(t)
	altered.data[len(altered.data)/2] ^= 0x20
	loader := NewLoader(parsed)
	altered.route(loader, refused.stores, parsed)
	if err := loader.Load(t.Context(), claim); !errors.Is(err, artifact.ErrContentMismatch) {
		t.Fatalf("load of changed bytes = %v, want a content mismatch", err)
	}
	if got := refused.currentBytes(t); got != baseline || refused.loaded() != 0 {
		t.Fatalf("after the refusal the ledger holds %d bytes (was %d) and %d tensors are stored", got, baseline, refused.loaded())
	}
	testcheck.NoError(t, refused.worker.Close())

	loaded := newDeviceStores(t)
	file := source.parse(t)
	loader = NewLoader(file)
	source.route(loader, loaded.stores, file)
	testcheck.NoError(t, loader.Load(t.Context(), claim))
	if routed := source.check(t, file, loaded.stores, loaded.download); routed != loaded.loaded() {
		t.Fatalf("stores hold %d tensors, want the %d routed", loaded.loaded(), routed)
	}
	testcheck.NoError(t, errors.Join(loaded.raw.Close(), loaded.f32.Close(), loaded.bf16.Close()))
	if got := loaded.currentBytes(t); got != 0 {
		t.Fatalf("after closing the stores the ledger holds %d bytes", got)
	}
}
