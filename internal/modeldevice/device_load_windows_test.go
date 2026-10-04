//go:build windows

package modeldevice

import (
	"bytes"
	"context"
	"errors"
	"math/rand/v2"
	"os"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testskip"
)

// loadGGUFEnvironment names a real, small GGUF the device tests load.
const loadGGUFEnvironment = "OVERGO_SINGLE_PASS_GGUF"

type deviceStores struct {
	worker *device.Worker
	stores
}

func newDeviceStores(t *testing.T) deviceStores {
	t.Helper()
	worker, err := device.New(modelTestDeviceOrdinal)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = worker.Close() })
	raw, err := NewDeviceWeights(worker)
	if err != nil {
		t.Fatal(err)
	}
	f32, err := NewDeviceConvertedWeights(worker, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	bf16, err := NewDeviceConvertedWeights(worker, dtype.BF16)
	if err != nil {
		t.Fatal(err)
	}
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
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func (d deviceStores) currentBytes(t *testing.T) uint64 {
	t.Helper()
	stats, err := d.worker.MemoryStats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
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
	if err := refused.worker.Close(); err != nil {
		t.Fatal(err)
	}

	loaded := newDeviceStores(t)
	file := source.parse(t)
	loader = NewLoader(file)
	source.route(loader, loaded.stores, file)
	if err := loader.Load(t.Context(), claim); err != nil {
		t.Fatal(err)
	}
	if routed := source.check(t, file, loaded.stores, loaded.download); routed != loaded.loaded() {
		t.Fatalf("stores hold %d tensors, want the %d routed", loaded.loaded(), routed)
	}
	if err := errors.Join(loaded.raw.Close(), loaded.f32.Close(), loaded.bf16.Close()); err != nil {
		t.Fatal(err)
	}
	if got := loaded.currentBytes(t); got != 0 {
		t.Fatalf("after closing the stores the ledger holds %d bytes", got)
	}
}

// TestLoaderRealModelResidentBytesMatchTheFile loads a real GGUF through the
// pass with the claim a whole-file hash gives, and holds every resident tensor
// to what the per-store loads this pass replaced put there: raw tensors the
// file's own bytes, converted ones the stores' encoding of the dequantized
// file bytes. A claim for other bytes is refused with nothing left resident.
// Choose a model under 2 GB; the pass reads it once.
func TestLoaderRealModelResidentBytesMatchTheFile(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	path := os.Getenv(loadGGUFEnvironment)
	if path == "" {
		testskip.NotApplicable(t, "set "+loadGGUFEnvironment+" to a small GGUF path")
	}
	file, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	var claim []artifact.ID
	for _, source := range file.SourcePaths() {
		held, err := os.Open(source)
		if err != nil {
			t.Fatal(err)
		}
		id, _, identifyErr := artifact.Identify(artifact.KindTensorSet, held)
		if err := errors.Join(identifyErr, held.Close()); err != nil {
			t.Fatal(err)
		}
		claim = append(claim, id)
	}

	// Route as a hybrid-native open does for a model without adapters: two-
	// dimensional half-precision and quantized tensors raw, the rest converted.
	var raw, converted []gguf.TensorInfo
	for _, info := range file.Tensors {
		if info.Dimensions == 2 && (info.Type == dtype.F16 || info.Type == dtype.BF16 || info.Type.IsQuantized()) {
			raw = append(raw, info)
		} else {
			converted = append(converted, info)
		}
	}

	wrong, err := artifact.IdentifyBytes(artifact.KindTensorSet, []byte("not this model"))
	if err != nil {
		t.Fatal(err)
	}
	refused := newDeviceStores(t)
	baseline := refused.currentBytes(t)
	loader := NewLoader(file)
	loader.Raw(refused.raw, raw)
	loader.Converted(refused.f32, converted)
	wrongClaim := slices.Repeat([]artifact.ID{wrong}, len(claim))
	if err := loader.Load(t.Context(), wrongClaim); !errors.Is(err, artifact.ErrContentMismatch) {
		t.Fatalf("load under a claim for other bytes = %v", err)
	}
	if got := refused.currentBytes(t); got != baseline || refused.loaded() != 0 {
		t.Fatalf("after the refusal the ledger holds %d bytes (was %d) and %d tensors are stored", got, baseline, refused.loaded())
	}
	if err := refused.worker.Close(); err != nil {
		t.Fatal(err)
	}

	loaded := newDeviceStores(t)
	loader = NewLoader(file)
	loader.Raw(loaded.raw, raw)
	loader.Converted(loaded.f32, converted)
	if err := loader.Load(t.Context(), claim); err != nil {
		t.Fatal(err)
	}
	for _, info := range raw {
		value, found := loaded.raw.Lookup(info.Name)
		stored := make([]byte, info.Size)
		if err := file.ReadTensorData(info, stored); err != nil {
			t.Fatal(err)
		}
		if !found || !bytes.Equal(loaded.download(t, value), stored) {
			t.Fatalf("raw tensor %q on the device differs from the file (found %t)", info.Name, found)
		}
	}
	for _, info := range converted {
		value, found := loaded.f32.Lookup(info.Name)
		host, err := model.LoadHostTensor(context.Background(), file, info)
		if err != nil {
			t.Fatal(err)
		}
		want := loaded.f32.encode(host.Data, len(host.Data)*4)
		if !found || !bytes.Equal(loaded.download(t, value), want) {
			t.Fatalf("converted tensor %q on the device differs from the file's values (found %t)", info.Name, found)
		}
	}
}
