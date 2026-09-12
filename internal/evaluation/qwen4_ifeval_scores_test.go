package evaluation

import (
	"fmt"
	"maps"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

func TestQwenFourIFEvalScoresAcceptance(t *testing.T) {
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	parse := func(value string) artifact.ID {
		t.Helper()
		id, err := artifact.ParseID(value)
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	profile := parse("evidence:sha256:cb3c98ead452ad292b511e8d58dfb40f16f70b0c42e0cb096a0e8af960654257")
	resolve := func(suffix string) artifact.ID {
		t.Helper()
		id, found, err := store.ResolveAlias(t.Context(), "validation/qwen4-ifeval/"+profile.DigestHex()+"/"+suffix)
		if err != nil || !found {
			t.Fatalf("Qwen4 IFEval %s is absent: found=%t err=%v", suffix, found, err)
		}
		return id
	}
	selectionID := resolve("acquired")
	nativeID := resolve("native")
	var scores ifevalNativeScores
	readRetainedEvidence(t, store, nativeID, &scores)
	if scores.Profile != profile || scores.Selection != selectionID {
		t.Fatal("native scores changed the acquisition profile or response selection")
	}
	// Keep the native implementation and its operand dependencies independent
	// of this model's responses and the Go checker under evaluation.
	type nativeReference struct {
		Version      string            `json:"reference_version"`
		Sources      map[string]string `json:"sources"`
		Packages     map[string]string `json:"runtime_packages"`
		Dependencies map[string]string `json:"runtime_dependencies"`
		InputSHA256  string            `json:"native_input_sha256"`
		LanguageSeed int               `json:"langdetect_seed"`
		RandomSeed   int               `json:"random_seed"`
	}
	var previous struct{ Scores artifact.ID }
	readRetainedEvidence(t, store, parse("evidence:sha256:78a1241f82ce179d229248564ccffb66f291d922eba864d1a261fc94d386c839"), &previous)
	var reference, actual nativeReference
	readRetainedEvidence(t, store, previous.Scores, &reference)
	readRetainedEvidence(t, store, nativeID, &actual)
	if actual.Version != "0.4.9.1" || actual.Version != reference.Version ||
		actual.InputSHA256 != reference.InputSHA256 || actual.LanguageSeed != 0 || actual.RandomSeed != 0 ||
		len(actual.Sources) == 0 || len(actual.Dependencies) == 0 ||
		!maps.Equal(actual.Sources, reference.Sources) || !maps.Equal(actual.Packages, reference.Packages) ||
		!maps.Equal(actual.Dependencies, reference.Dependencies) {
		t.Fatal("pinned native judge or task inputs changed")
	}
	var selection struct {
		Profile artifact.ID
		Cells   []struct {
			Name   string
			Output artifact.ID
		}
	}
	readRetainedEvidence(t, store, selectionID, &selection)
	if selection.Profile != profile || len(selection.Cells) != 541 {
		t.Fatal("raw acquisition denominator differs")
	}
	responses := make(map[string]string, len(selection.Cells))
	for _, cell := range selection.Cells {
		if _, duplicate := responses[cell.Name]; duplicate {
			t.Fatal("duplicate raw case")
		}
		var row struct {
			Profile artifact.ID
			Result  ExactResult
		}
		readRetainedEvidence(t, store, cell.Output, &row)
		if row.Profile != profile || row.Result.Name != cell.Name || row.Result.GeneratedTokens > 1280 {
			t.Fatal("raw response changed its profile, case or native token cap")
		}
		responses[cell.Name] = row.Result.Text
	}
	for index := range 541 {
		if _, found := responses[fmt.Sprintf("ifeval/default/train/%d", index)]; !found {
			t.Fatal("native prompt is missing")
		}
	}
	checkRetainedIFEvalScores(t, store, scores, responses)
}
