//go:build modeltest && windows

package modeldevice

import (
	"bytes"
	"context"
	"errors"
	"os"
	"slices"
	"testing"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modeltest"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
)

// TestLoaderRealModelResidentBytesMatchTheFile loads a real GGUF through the
// pass with the claim a whole-file hash gives, and holds every resident tensor
// to what the per-store loads this pass replaced put there: raw tensors the
// file's own bytes, converted ones the stores' encoding of the dequantized
// file bytes. A claim for other bytes is refused with nothing left resident.
// Choose a model under 2 GB; the pass reads it once.
func TestLoaderRealModelResidentBytesMatchTheFile(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	// The smallest registered Qwen2 GGUF.
	path := modeltest.GGUF(t, "qwen2")
	file := testcheck.Must(gguf.Open(path))(t)
	defer file.Close()
	var claim []artifact.ID
	for _, source := range file.SourcePaths() {
		held := testcheck.Must(os.Open(source))(t)
		id, _, identifyErr := artifact.Identify(artifact.KindTensorSet, held)
		testcheck.NoError(t, errors.Join(identifyErr, held.Close()))
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

	wrong := testcheck.Must(artifact.IdentifyBytes(artifact.KindTensorSet, []byte("not this model")))(t)
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
	testcheck.NoError(t, refused.worker.Close())

	loaded := newDeviceStores(t)
	loader = NewLoader(file)
	loader.Raw(loaded.raw, raw)
	loader.Converted(loaded.f32, converted)
	testcheck.NoError(t, loader.Load(t.Context(), claim))
	for _, info := range raw {
		value, found := loaded.raw.Lookup(info.Name)
		stored := make([]byte, info.Size)
		testcheck.NoError(t, file.ReadTensorData(info, stored))
		if !found || !bytes.Equal(loaded.download(t, value), stored) {
			t.Fatalf("raw tensor %q on the device differs from the file (found %t)", info.Name, found)
		}
	}
	for _, info := range converted {
		value, found := loaded.f32.Lookup(info.Name)
		host := testcheck.Must(model.LoadHostTensor(context.Background(), file, info))(t)
		want := loaded.f32.encode(host.Data, len(host.Data)*4)
		if !found || !bytes.Equal(loaded.download(t, value), want) {
			t.Fatalf("converted tensor %q on the device differs from the file's values (found %t)", info.Name, found)
		}
	}
}
