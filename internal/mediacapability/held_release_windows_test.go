//go:build windows

package mediacapability

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/heldfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
)

// TestARequestWithNoSessionLetsItsHeldFilesGo holds a picture request that ends with no session to letting go the files
// it named (held-release, SQA-454): served as the image capability serves a latent picture request, its files named
// through the process's held files (capabilityruntime.Reading over preparedFirst, capabilityruntime.Contents over
// heldfile.Session and pictureFiles), a request refused before its session (its depth control's picture a stored file
// the store does not hold) and a request answered with no session leased each leave the adapter and the depth
// estimator to a writer, who replaces each after either. adapter-hold had held them in the process's set until a
// session under the model closed, which neither request opens, so a writer could not replace the adapter while the
// process ran; and a test reading a model's adapter with no session left it held, so its directory could not be
// removed (TestAMOSSBlockIsHeldToComfyUIs in the models lane, 2026-11-21).
func TestARequestWithNoSessionLetsItsHeldFilesGo(t *testing.T) {
	model := t.TempDir()
	adapter, estimator := filepath.Join(model, "adapters", "moss-control.safetensors"), filepath.Join(model, "depth", "depth_anything_v2_vits.pth")
	for _, file := range []string{adapter, estimator} {
		testcheck.NoError(t, os.MkdirAll(filepath.Dir(file), 0o755))
		testcheck.NoError(t, os.WriteFile(file, []byte(filepath.Base(file)+"'s weights"), 0o600))
	}
	store := overgodbtest.Open(t, t.TempDir())
	absent := testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, []byte("a picture the store does not hold")))(t)
	answered := func(context.Context, artifact.Repository, string, modelrecipe.CapabilityEvidenceSelection, string) (any, error) {
		return "an answer the store kept", nil
	}
	serve := capabilityruntime.Reading(preparedFirst(answered), capabilityruntime.Contents(heldfile.Session, pictureFiles))
	request := func(picture string) string {
		return `{"prompt":"a fox","width":1024,"height":576,"seed":7,"adapter":"moss-control","control":{"picture":"` + picture + `"}}`
	}
	replaceable := func(after string) {
		t.Helper()
		for _, file := range []string{adapter, estimator} {
			other := file + ".written"
			testcheck.NoError(t, os.WriteFile(other, []byte("weights written after "+after), 0o600))
			if err := os.Rename(other, file); err != nil {
				t.Errorf("after %s, a writer could not replace %s: %v; want it let go", after, filepath.Base(file), err)
			}
		}
	}
	if _, err := serve(t.Context(), store, model, modelrecipe.CapabilityEvidenceSelection{}, request(absent.String())); err == nil {
		t.Fatal("a request whose depth control's picture the store does not hold was served; want it refused before its session")
	}
	replaceable("a request refused before its session")
	testcheck.Must(serve(t.Context(), store, model, modelrecipe.CapabilityEvidenceSelection{}, request("a picture")))(t)
	replaceable("a request answered with no session")
}
