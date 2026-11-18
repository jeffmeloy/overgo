//go:build windows

package mediacapability

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/capabilityruntime"
	"overgo/internal/testcheck"
)

// TestAReplacedAdapterIsReadAgain holds a picture request naming an adapter to the adapter's bytes (SQA-406): its
// answer is kept under the bytes of the files it reads beside its model (pictureFiles, capabilityruntime.Contents), so
// other weights installed under the adapter's name are a new run, computed afresh, even under the size and the time
// the earlier file had, and the earlier bytes again name the earlier run, which the store replays. Each file named is
// held as named until the request is served, so no weights are switched under it between its naming and its reading.
// A request's preset's adapter is named as its own, a depth control drawn from a picture names the depth estimator it
// is read by and one given its map does not, a request reading no such file is named nothing and kept as before, and a
// name that is no file name is not named, so the request is run afresh and refused by its session.
func TestAReplacedAdapterIsReadAgain(t *testing.T) {
	t.Parallel()
	model := t.TempDir()
	adapter, estimator := filepath.Join(model, "adapters", "turbo.safetensors"), filepath.Join(model, "depth", "depth_anything_v2_vits.pth")
	installed := time.Now().Add(-time.Hour)
	install := func(file, bytes string) {
		t.Helper()
		// Each file is installed under the one time, as a copy keeping its source's time installs it.
		testcheck.NoError(t, os.MkdirAll(filepath.Dir(file), 0o755))
		testcheck.NoError(t, os.WriteFile(file, []byte(bytes), 0o600))
		testcheck.NoError(t, os.Chtimes(file, installed, installed))
	}
	naming := capabilityruntime.Contents(pictureFiles)
	name := func(request string) string {
		t.Helper()
		named, release, err := naming(t.Context(), nil, model, request)
		testcheck.NoError(t, err)
		testcheck.NoError(t, release())
		return named
	}
	const named = `{"prompt":"a lighthouse at dusk","width":64,"height":64,"seed":7,"adapters":[{"name":"turbo"}]}`

	install(adapter, "the first adapter's weights")
	first := name(named)
	install(adapter, "the other adapter's weights")
	if second := name(named); second == first || second == "" {
		t.Fatalf("other weights under the adapter's name, size and time are named %q, as the earlier file's %q; want a run of their own", second, first)
	}
	install(adapter, "the first adapter's weights")
	if again := name(named); again != first {
		t.Fatalf("the adapter's first bytes again are named %q; want the first run's %q, which the store replays", again, first)
	}

	_, release, err := naming(t.Context(), nil, model, named)
	testcheck.NoError(t, err)
	if os.WriteFile(adapter, []byte("weights switched while read"), 0o600) == nil {
		t.Error("the adapter was written between its naming and the request's end; want it held as named")
	}
	testcheck.NoError(t, release())
	testcheck.NoError(t, os.WriteFile(adapter, []byte("the first adapter's weights"), 0o600))

	// A preset's adapters are read as the request's own: the depth section's control adapter here.
	testcheck.NoError(t, os.WriteFile(filepath.Join(model, "model_index.json"), []byte(`{"_class_name": "MOSSImagePipeline"}`), 0o600))
	control := filepath.Join(model, "adapters", "moss-control.safetensors")
	const preset = `{"prompt":"a fox","width":64,"height":64,"seed":7,"preset":"ultra-depth-control","control":{"map":"a depth map"}}`
	install(control, "the first control adapter")
	presetFirst := name(preset)
	install(control, "the other control adapter")
	if name(preset) == presetFirst {
		t.Fatal("a request whose preset names an adapter replaced under its name is named as before; want a run of its own")
	}

	const drawn = `{"prompt":"a fox","width":64,"height":64,"seed":7,"adapter":"turbo","control":{"picture":"a picture"}}`
	const given = `{"prompt":"a fox","width":64,"height":64,"seed":7,"adapter":"turbo","control":{"map":"a depth map"}}`
	install(estimator, "the first estimator's weights")
	drawnFirst, givenFirst := name(drawn), name(given)
	install(estimator, "the other estimator weights!!")
	if name(drawn) == drawnFirst || name(given) != givenFirst {
		t.Fatal("a depth control is named apart from its estimator's bytes; want one drawn from a picture named by them, and one given its map not")
	}

	if bare := name(`{"prompt":"a lighthouse at dusk","width":64,"height":64,"seed":7}`); bare != "" {
		t.Fatalf("a request naming no adapter is named %q; want nothing, so it is kept as before", bare)
	}
	if _, _, err := naming(t.Context(), nil, model, `{"prompt":"a lighthouse","width":64,"height":64,"adapter":"..\\turbo"}`); err == nil {
		t.Fatal("a request naming an adapter outside the model's adapters was named; want it run afresh, named nothing")
	}
}
