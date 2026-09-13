package evaluation

import (
	"fmt"
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
	checkIFEvalNativeReference(t, store, nativeID)
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
