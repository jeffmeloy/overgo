package gate

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/repoanalysis"
)

// TestPreflightPassesDriftedClosureBindings holds the preflight to judging a
// catalogued literal that only moved offset as the gate's rebind will: as
// covered. Before, the preflight wrote no store and so could not rebind, a
// moved literal read as new policy, and the landing stopped for a hand-run
// rebind the gate would have done itself. A literal no decision covers is
// still refused.
func TestPreflightPassesDriftedClosureBindings(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	write := func(source string) repoanalysis.SourceSnapshot {
		t.Helper()
		path := filepath.Join(root, "internal", "policy", "policy.go")
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(source), 0o644); err != nil {
			t.Fatal(err)
		}
		snapshot, err := repoanalysis.DiscoverGo(root, "internal")
		if err != nil {
			t.Fatal(err)
		}
		return snapshot
	}
	original := write("package policy\n\nfunc admitted(n int) bool { return n > 4096 }\n")
	candidates, err := closurescan.ScanSnapshot(original, nil, closurescan.CandidateAll)
	if err != nil || len(candidates) != 1 {
		t.Fatalf("candidates = %d: %v", len(candidates), err)
	}
	binding, err := candidates[0].Binding()
	if err != nil {
		t.Fatal(err)
	}
	fixture, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("drift fixture"))
	if err != nil {
		t.Fatal(err)
	}
	document, err := closureledger.New(
		candidates[0].Name, json.RawMessage(candidates[0].ValueJSON()), closureledger.TierImplementation, closureledger.StatusClosed,
		"Admission ceiling.", []closureledger.SourceBinding{binding},
		"Retain while the ceiling holds.", "A different ceiling.", fixture,
	)
	if err != nil {
		t.Fatal(err)
	}
	alias, err := closureledger.ActiveAlias(binding)
	if err != nil {
		t.Fatal(err)
	}
	documents, aliases := []closureledger.Document{document}, map[string]artifact.ID{alias: document.ID}
	if _, err := closurescan.ValidatePermanentActiveAuthority(original, documents, aliases); err != nil {
		t.Fatalf("the catalogued literal was refused at its own offset: %v", err)
	}

	moved := write("package policy\n\n// A comment above moves every offset below it.\nfunc admitted(n int) bool { return n > 4096 }\n")
	if _, err := closurescan.ValidatePermanentActiveAuthority(moved, documents, aliases); err == nil || !staleClosureAuthorityFailure(err) {
		t.Fatalf("a moved literal passed against its old offset: %v", err)
	}
	rebound, reboundAliases, count, err := projectedMagicBindings(moved, documents, aliases)
	if err != nil || count != 1 {
		t.Fatalf("projection moved %d: %v", count, err)
	}
	if _, err := closurescan.ValidatePermanentActiveAuthority(moved, rebound, reboundAliases); err != nil {
		t.Fatalf("a literal that only moved was refused after the projected rebind: %v", err)
	}

	added := write("package policy\n\n// A comment above moves every offset below it.\nfunc admitted(n int) bool { return n > 4096 }\n\nfunc capped(n int) bool { return n < 8191 }\n")
	rebound, reboundAliases, _, err = projectedMagicBindings(added, documents, aliases)
	if err != nil {
		t.Fatal(err)
	}
	_, err = closurescan.ValidatePermanentActiveAuthority(added, rebound, reboundAliases)
	if _, uncatalogued := errors.AsType[*closurescan.UncataloguedPolicyError](err); !uncatalogued {
		t.Fatalf("a new literal passed the projected rebind: %v", err)
	}
}
