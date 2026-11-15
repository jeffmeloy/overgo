package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"overgo/internal/model"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAModelHoldsNoFeatureList holds the compatibility manifest's models to what the architecture registry does not
// say (single-source/models-features): each of the registry's architectures has its entry, and no entry lists the
// architecture's features. The manifest held 137 hand-written lists, 50 KB of names for the policies each
// architecture's profile holds (internal/model), checked by nothing and printed only by the -models report; the
// manifest's model entry has no field for one, and the manifest lists none.
func TestAModelHoldsNoFeatureList(t *testing.T) {
	t.Parallel()
	if _, held := reflect.TypeFor[modelClaim]().FieldByName("Features"); held {
		t.Error("the manifest's model entry holds a feature list, which its registry profile holds")
	}
	var document struct {
		Models map[string]map[string]json.RawMessage `json:"models"`
	}
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(filepath.Join(testutil.RepoRoot(t), manifestPath)))(t), &document))
	for _, name := range model.SupportedArchitectures() {
		entry, found := document.Models[name]
		if !found {
			t.Errorf("the registry's architecture %s has no entry in the manifest", name)
		}
		if _, listed := entry["features"]; listed {
			t.Errorf("the manifest lists %s's features, which its profile in the registry holds", name)
		}
	}
}
