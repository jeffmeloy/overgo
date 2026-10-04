//go:build windows

package inference

import (
	"crypto/sha256"
	"errors"
	"io"
	"maps"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/modelartifact"
	"overgo/internal/modeldevice"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/servingtest"
	"overgo/internal/testskip"
)

// singlePassEnvironment names a real, small GGUF this test opens; it is copied
// first, so the original is never written.
const singlePassEnvironment = "OVERGO_SINGLE_PASS_GGUF"

// residentDigests opens the program and digests every tensor it holds on the
// device, by store; the runner is closed before it returns.
func residentDigests(t *testing.T, loaded modelrecipe.LoadedProgram) map[string][sha256.Size]byte {
	t.Helper()
	runner, err := OpenWithProgram(t.Context(), &loaded, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	digests := map[string][sha256.Size]byte{}
	err = runner.rawWeights.Do(t.Context(), func(state *device.State, tensors map[string]modeldevice.DeviceTensor) error {
		for name, value := range tensors {
			data := make([]byte, value.Info.Size)
			if err := state.Driver.MemcpyDtoH(data, value.Pointer); err != nil {
				return err
			}
			digests["raw "+name] = sha256.Sum256(data)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, info := range runner.file.Tensors {
		value, found := runner.deviceWeights.Lookup(info.Name)
		if !found {
			continue
		}
		data := make([]byte, value.Size)
		err := runner.worker.Do(t.Context(), func(state *device.State) error {
			return state.Driver.MemcpyDtoH(data, value.Pointer)
		})
		if err != nil {
			t.Fatal(err)
		}
		digests["f32 "+info.Name] = sha256.Sum256(data)
	}
	if len(digests) == 0 {
		t.Fatal("the runner holds no tensors on the device")
	}
	return digests
}

// TestOpenWithProgramClaimedLoadMatchesTheVerifiedLoad opens one real model
// twice, once resolved by hashing its file and once from its claimed identity
// proved by the load itself, and finds the same bytes resident. The same model
// with a byte changed after its claim is refused.
func TestOpenWithProgramClaimedLoadMatchesTheVerifiedLoad(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	original := os.Getenv(singlePassEnvironment)
	if original == "" {
		testskip.NotApplicable(t, "set "+singlePassEnvironment+" to a small GGUF path")
	}
	path := filepath.Join(t.TempDir(), "model.gguf")
	source, err := os.Open(original)
	if err != nil {
		t.Fatal(err)
	}
	copied, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	_, copyErr := io.Copy(copied, source)
	if err := errors.Join(copyErr, source.Close(), copied.Close()); err != nil {
		t.Fatal(err)
	}

	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	err = servingtest.PublishActiveGGUFWithPolicy(
		t.Context(), store, path, recipe.PlacementHybrid, modelrecipe.DecodeSessionCapacity, recipe.ResidencyHybridNative,
	)
	if err != nil {
		t.Fatal(err)
	}
	inventory, err := modelartifact.FromGGUFPath(path)
	if err != nil {
		t.Fatal(err)
	}
	remembered := modelartifact.Claimer(func(candidate string) (artifact.ID, uint64, bool) {
		for index, component := range inventory.Components {
			if inventory.Locations[index].Value == candidate {
				return component.ID, component.Size, true
			}
		}
		return artifact.ID{}, 0, false
	})

	verified, err := modelrecipe.ResolveActiveGGUF(t.Context(), store, path)
	if err != nil {
		t.Fatal(err)
	}
	want := residentDigests(t, verified)
	claimed, err := modelrecipe.ResolveClaimedGGUF(t.Context(), store, path, remembered)
	if err != nil {
		t.Fatal(err)
	}
	if got := residentDigests(t, claimed); !maps.Equal(got, want) {
		t.Fatalf("the claimed load holds %d tensors that differ from the verified load's %d", len(got), len(want))
	}

	claimed, err = modelrecipe.ResolveClaimedGGUF(t.Context(), store, path, remembered)
	if err != nil {
		t.Fatal(err)
	}
	file, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	info, err := file.Stat()
	if err != nil {
		t.Fatal(err)
	}
	last := make([]byte, 1)
	if _, err := file.ReadAt(last, info.Size()-1); err != nil {
		t.Fatal(err)
	}
	last[0] ^= 0x01
	if _, err := file.WriteAt(last, info.Size()-1); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	runner, err := OpenWithProgram(t.Context(), &claimed, OpenOptions{})
	if runner != nil {
		_ = runner.Close()
	}
	if !errors.Is(err, artifact.ErrContentMismatch) || runner != nil {
		t.Fatalf("open of a model changed after its claim = %v, want a content mismatch and no runner", err)
	}
}
