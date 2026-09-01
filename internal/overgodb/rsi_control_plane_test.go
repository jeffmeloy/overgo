package overgodb

import (
	"encoding/json"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
)

// TestDeterministicRSIControlPlane proves the storage slice of the control
// plane: the causal evidence a decision cites reproduces byte-for-byte after
// a cold rebuild into a fresh segmented layout, from a read-only open, with
// the same bounded causality answers — the reproducible-explanation
// guarantee every other control stage stands on.
func TestDeterministicRSIControlPlane(t *testing.T) {
	base := t.TempDir()
	sourceRoot := filepath.Join(base, "source")
	ctx := t.Context()
	store, err := Open(sourceRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	rootEvidence, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("plane motivation"))
	if err != nil {
		t.Fatal(err)
	}
	finding, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("plane finding"))
	if err != nil {
		t.Fatal(err)
	}
	explanation := []byte(`{"decision":"rollback","cause":"sustained live regression"}`)
	explanationID, err := artifact.IdentifyBytes(artifact.KindEvidence, explanation)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "plane/causal-evidence",
		Artifacts: []artifact.Descriptor{
			{ID: rootEvidence}, {ID: finding},
			{ID: explanationID, Size: uint64(len(explanation)), MediaType: "application/json"},
		},
		Contents: []artifact.Content{{
			Descriptor: artifact.Descriptor{
				ID: explanationID, Size: uint64(len(explanation)), MediaType: "application/json",
			},
			Data: explanation,
		}},
		Causality: []artifact.CausalLink{{
			Execution: finding, Root: rootEvidence, Trigger: "controller_proposal",
		}},
		Lineage: []artifact.Lineage{{
			Child: explanationID, Parent: finding, Relation: artifact.RelationDerivedFrom,
		}},
	}); err != nil {
		t.Fatal(err)
	}

	answer := func(from *Store) (artifact.CausalLink, []byte) {
		t.Helper()
		result, queryErr := from.QueryCausality(ctx, CausalityQuery{
			Execution: &finding, MaxResults: 4,
		})
		if queryErr != nil || result.Matched != 1 || len(result.Links) != 1 {
			t.Fatalf("causality answer = (%+v, %v)", result, queryErr)
		}
		content, found, readErr := artifact.ReadContent(ctx, from, explanationID)
		if readErr != nil || !found {
			t.Fatalf("explanation content = (%v, %v)", found, readErr)
		}
		return result.Links[0], content.Data
	}
	sourceLink, sourceExplanation := answer(store)
	if sourceLink.Root != rootEvidence {
		t.Fatalf("stored link lost the root: %+v", sourceLink)
	}

	// Cold rebuild into a fresh layout, then answer read-only.
	rebuiltRoot := filepath.Join(base, "rebuilt")
	if _, err := Rebuild(ctx, store, rebuiltRoot, nil, nil); err != nil {
		t.Fatal(err)
	}
	rebuilt, err := OpenReadOnly(rebuiltRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer rebuilt.Close()
	rebuiltLink, rebuiltExplanation := answer(rebuilt)
	sourceEncoded, err := json.Marshal(sourceLink)
	if err != nil {
		t.Fatal(err)
	}
	rebuiltEncoded, err := json.Marshal(rebuiltLink)
	if err != nil {
		t.Fatal(err)
	}
	if string(sourceEncoded) != string(rebuiltEncoded) {
		t.Fatalf("causal link drifted across the rebuild: %s vs %s", sourceEncoded, rebuiltEncoded)
	}
	if string(sourceExplanation) != string(rebuiltExplanation) {
		t.Fatal("the reproduced explanation differs byte-for-byte")
	}
}
