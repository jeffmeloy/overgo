package closurescan

import (
	"encoding/json"
	"testing"

	"overgo/internal/closureledger"
	"overgo/internal/repoanalysis"
)

func TestValidateBindingsDetectsSourceDrift(t *testing.T) {
	snapshot := scanTestSnapshot(t, map[string]string{
		"internal/policy.go": "package policy\nconst Limit = 8\n",
	})
	candidate := onlyCandidate(t, snapshot)
	document := candidateDocument(t, candidate)
	changed, err := snapshot.Overlay(map[string][]byte{
		"internal/policy.go": []byte("package policy\nconst Limit = 9\n"),
	})
	if err != nil {
		t.Fatal(err)
	}
	issues, err := ValidateBindings(changed, []closureledger.Document{document})
	if err != nil || len(issues) != 1 || issues[0].Kind != DriftSource {
		t.Fatalf("source issues = (%+v, %v)", issues, err)
	}
}

func TestValidateBindingsDetectsOrphansAndCallsiteDrift(t *testing.T) {
	snapshot := scanTestSnapshot(t, map[string]string{
		"internal/policy/policy.go": "package policy\nconst Limit = 8\n",
		"internal/use/use.go":       "package use\nimport \"example/internal/policy\"\nvar configured = policy.Limit\n",
	})
	candidate := onlyCandidate(t, snapshot)
	document := candidateDocument(t, candidate)

	t.Run("orphan", func(t *testing.T) {
		changed, err := snapshot.Overlay(map[string][]byte{"internal/policy/policy.go": nil})
		if err != nil {
			t.Fatal(err)
		}
		issues, err := ValidateBindings(changed, []closureledger.Document{document})
		if err != nil || len(issues) != 1 || issues[0].Kind != DriftOrphan {
			t.Fatalf("orphan issues = (%+v, %v)", issues, err)
		}
	})
	t.Run("callsite", func(t *testing.T) {
		changed, err := snapshot.Overlay(map[string][]byte{
			"internal/use/use.go": []byte("package use\nimport \"example/internal/policy\"\nvar configured = 8\n"),
		})
		if err != nil {
			t.Fatal(err)
		}
		issues, err := ValidateBindings(changed, []closureledger.Document{document})
		if err != nil || len(issues) != 1 || issues[0].Kind != DriftCallsite {
			t.Fatalf("callsite issues = (%+v, %v)", issues, err)
		}
	})
}

func onlyCandidate(t *testing.T, snapshot repoanalysis.SourceSnapshot) Candidate {
	t.Helper()
	candidates, err := ScanSnapshot(snapshot, nil, CandidateConstants)
	if err != nil || len(candidates) != 1 {
		t.Fatalf("candidates = (%+v, %v)", candidates, err)
	}
	return candidates[0]
}

func candidateDocument(t *testing.T, candidate Candidate) closureledger.Document {
	t.Helper()
	binding, err := candidate.Binding()
	if err != nil {
		t.Fatal(err)
	}
	document, err := closureledger.New(
		candidate.Name, json.RawMessage(candidate.ValueJSON()), closureledger.TierImplementation,
		closureledger.StatusClosed, "Fixed test policy.", []closureledger.SourceBinding{binding},
		"Replace when policy ownership changes.", "Policy ownership change.", binding.Owner,
	)
	if err != nil {
		t.Fatal(err)
	}
	return document
}
