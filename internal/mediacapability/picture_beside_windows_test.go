//go:build windows

package mediacapability

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/capabilityruntime"
	"overgo/internal/latentimage"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
)

// TestAReplacedAdapterIsReadAgain holds a picture request naming an adapter to the adapter's bytes (SQA-406): the
// picture director keys its run by the file's content as well as the request, so a file replaced under the adapter's
// name is a new run, computed afresh, and the earlier bytes again key the earlier run, which the store replays. A run was
// keyed by the request alone, and a replaced adapter replayed the picture the earlier file made. So is a depth control
// drawn from a picture keyed by the depth estimator it is read by, and one given its map by none. A request naming no
// adapter is keyed as before, and a name that is no file name is refused.
func TestAReplacedAdapterIsReadAgain(t *testing.T) {
	t.Parallel()
	model := t.TempDir()
	adapter, estimator := filepath.Join(model, "adapters", "turbo.safetensors"), filepath.Join(model, "depth", "depth_anything_v2_vits.pth")
	written := time.Now()
	write := func(file, bytes string) {
		t.Helper()
		// Each write is a later file, as a replaced one is.
		written = written.Add(time.Second)
		testcheck.NoError(t, os.MkdirAll(filepath.Dir(file), 0o755))
		testcheck.NoError(t, os.WriteFile(file, []byte(bytes), 0o600))
		testcheck.NoError(t, os.Chtimes(file, written, written))
	}
	director := pictureService(t)
	var definition recipe.Definition
	key := func(request latentimage.Request) string {
		t.Helper()
		content := testcheck.Must(capabilityruntime.InputContent("overgo.image-gen.request.v1", request))(t)
		return testcheck.Must(director.RunKey(t.Context(), model, definition, request, content))(t)
	}
	named := latentimage.Request{Prompt: "a lighthouse at dusk", Width: 64, Height: 64, Seed: 7, Adapters: &[]latentimage.AppliedAdapter{{Name: "turbo"}}}

	write(adapter, "the first adapter's weights")
	first := key(named)
	write(adapter, "the second adapter weights!")
	if second := key(named); second == first {
		t.Fatalf("a request naming an adapter replaced under its name is keyed %s, as the earlier file's run; want a run of its own", second)
	}
	write(adapter, "the first adapter's weights")
	if again := key(named); again != first {
		t.Fatalf("a request naming the adapter's first bytes again is keyed %s; want the first run's key %s, which the store replays", again, first)
	}

	// A preset's adapters are read as the request's own: the depth section's control adapter here.
	testcheck.NoError(t, os.WriteFile(filepath.Join(model, "model_index.json"), []byte(`{"_class_name": "MOSSImagePipeline"}`), 0o600))
	control := filepath.Join(model, "adapters", "moss-control.safetensors")
	preset := latentimage.Request{Prompt: "a fox", Width: 64, Height: 64, Preset: "ultra-depth-control"}
	write(control, "the first control adapter")
	presetFirst := key(preset)
	write(control, "the other control adapter")
	if key(preset) == presetFirst {
		t.Fatal("a request whose preset names an adapter replaced under its name is keyed as before; want a run of its own")
	}

	drawn, given := named, named
	drawn.Control, given.Control = &latentimage.DepthControl{Picture: "a picture"}, &latentimage.DepthControl{Map: "a depth map"}
	write(estimator, "the first estimator's weights")
	drawnFirst, givenFirst := key(drawn), key(given)
	write(estimator, "the other estimator weights!!")
	if key(drawn) == drawnFirst || key(given) != givenFirst {
		t.Fatal("a depth control is keyed apart from its estimator's bytes; want one drawn from a picture keyed by them, and one given its map not")
	}

	bare := named
	bare.Adapters = nil
	content := testcheck.Must(capabilityruntime.InputContent("overgo.image-gen.request.v1", bare))(t)
	if got, want := key(bare), capabilityruntime.RunKeyOf[*latentimage.Generator](definition, content); got != want {
		t.Fatalf("a request naming no adapter is keyed %s; want %s, as before", got, want)
	}
	outside := named
	outside.Adapter, outside.Adapters = `..\turbo`, nil
	content = testcheck.Must(capabilityruntime.InputContent("overgo.image-gen.request.v1", outside))(t)
	if _, err := director.RunKey(t.Context(), model, definition, outside, content); err == nil {
		t.Fatal("a request naming an adapter outside the model's adapters was keyed; want it refused")
	}
}
