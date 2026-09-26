package mediacapability

import (
	"cmp"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// TestActivityDetectionHonoursRequestOverrides runs the catalog's activity
// detector end to end on a recording the model's own repository ships. The
// default policy is the one the model's tools ship and finds the speech; a
// request that names a field replaces that field alone, and a stricter
// threshold keeps less of the recording; a field the policy does not have is
// refused rather than ignored; and weights no reviewed declaration answers
// for are refused before any tensor is bound.
func TestActivityDetectionHonoursRequestOverrides(t *testing.T) {
	repository := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(repository)
	if err != nil {
		t.Fatal(err)
	}
	// The registration records where the reviewed weights live; the test
	// asks the store for them by the digest the reviewed policy names.
	reference := cmp.Or(os.Getenv("OVERGO_AUDIO_REFERENCE_STORE"), roots.AudioReference)
	if _, err := os.Stat(reference); errors.Is(err, fs.ErrNotExist) {
		testskip.NotApplicable(t, "no audio reference store holds the activity model's registration here")
	}
	registrations, err := overgodb.OpenReadOnly(reference)
	if err != nil {
		t.Fatal(err)
	}
	defer registrations.Close()
	var policy struct {
		Source struct {
			Weights string `json:"offline_weights_sha256"`
		} `json:"source"`
	}
	if err := jsonfile.Decode(filepath.Join(repository, "internal", "audioparity", "recipes", "firered_vad_policy.json"), &policy); err != nil {
		t.Fatal(err)
	}
	weights, err := artifact.ParseID("tensor-set:sha256:" + policy.Source.Weights)
	if err != nil {
		t.Fatal(err)
	}
	offline, err := artifact.AvailablePath(t.Context(), registrations, weights, artifact.LocationDirectory)
	if err != nil {
		t.Fatal(err)
	}
	// The recording and the streaming checkpoint ship in the same model
	// repository as the registered offline checkpoint.
	root := filepath.Dir(offline)
	recording, err := os.ReadFile(filepath.Join(root, "FireRedVAD", "assets", "hello_en.wav"))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	capability := Catalog[recipe.TaskActivityDetection]
	detect := func(t *testing.T, directory, policy string) ([]recipecontract.ActivitySegment, error) {
		t.Helper()
		source, err := capability.Resolve(directory)
		if err != nil {
			t.Fatal(err)
		}
		definition, err := modelrecipe.CapabilityDefinition(recipe.TaskActivityDetection, source.Inventory.Manifest.ID)
		if err != nil {
			t.Fatal(err)
		}
		program, err := modelrecipe.CompileCapability(definition)
		if err != nil {
			t.Fatal(err)
		}
		request, err := json.Marshal(map[string]any{
			"audio": recording, "maximum_samples": len(recording), "memory_bytes": 64 << 20, "policy": json.RawMessage(policy),
		})
		if err != nil {
			t.Fatal(err)
		}
		output, err := capability.Execute(t.Context(), store, directory, modelrecipe.CapabilityEvidenceSelection{Program: program}, string(request))
		if err != nil {
			return nil, err
		}
		segments, ok := capabilityruntime.Unwrap(output).([]recipecontract.ActivitySegment)
		if !ok {
			t.Fatalf("activity output is %T", capabilityruntime.Unwrap(output))
		}
		return segments, nil
	}
	speech := func(segments []recipecontract.ActivitySegment) uint64 {
		var total uint64
		for _, segment := range segments {
			total += segment.Span.End - segment.Span.Start
		}
		return total
	}
	defaults, err := detect(t, offline, `{}`)
	if err != nil {
		t.Fatal(err)
	}
	if len(defaults) == 0 {
		t.Fatal("the shipped policy found no speech in a spoken greeting")
	}
	strict, err := detect(t, offline, `{"threshold":0.95}`)
	if err != nil {
		t.Fatal(err)
	}
	if speech(strict) >= speech(defaults) {
		t.Fatalf("a stricter threshold kept %d samples of speech, the default %d", speech(strict), speech(defaults))
	}
	if _, err := detect(t, offline, `{"thresold":0.95}`); err == nil || !strings.Contains(err.Error(), "policy override") {
		t.Fatalf("a misspelled policy field was not refused: %v", err)
	}
	if _, err := detect(t, filepath.Join(root, "Stream-VAD"), `{}`); err == nil || !strings.Contains(err.Error(), "no reviewed declaration") {
		t.Fatalf("weights without a reviewed declaration were bound: %v", err)
	}
}
