package gate

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/overgodb"
	"overgo/internal/repoanalysis"
)

// closureFixture is one policy file under a temporary root and the single
// closure decision catalogued against its first version.
type closureFixture struct {
	root      string
	documents []closureledger.Document
	aliases   map[string]artifact.ID
}

const catalogued = "package policy\n\nfunc admitted(n int) bool { return n > 4096 }\n"

func newClosureFixture(t *testing.T) *closureFixture {
	t.Helper()
	fixture := &closureFixture{root: t.TempDir()}
	candidates, err := closurescan.ScanSnapshot(fixture.write(t, catalogued), nil, closurescan.CandidateAll)
	if err != nil || len(candidates) != 1 {
		t.Fatalf("candidates = %d: %v", len(candidates), err)
	}
	binding, err := candidates[0].Binding()
	if err != nil {
		t.Fatal(err)
	}
	evidence, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("closure fixture"))
	if err != nil {
		t.Fatal(err)
	}
	document, err := closureledger.New(
		candidates[0].Name, json.RawMessage(candidates[0].ValueJSON()), closureledger.TierImplementation, closureledger.StatusClosed,
		"Admission ceiling.", []closureledger.SourceBinding{binding},
		"Retain while the ceiling holds.", "A different ceiling.", evidence,
	)
	if err != nil {
		t.Fatal(err)
	}
	alias, err := closureledger.ActiveAlias(binding)
	if err != nil {
		t.Fatal(err)
	}
	fixture.documents, fixture.aliases = []closureledger.Document{document}, map[string]artifact.ID{alias: document.ID}
	return fixture
}

// write replaces the policy file and returns the snapshot of the root.
func (f *closureFixture) write(t *testing.T, source string) repoanalysis.SourceSnapshot {
	t.Helper()
	path := filepath.Join(f.root, "internal", "policy", "policy.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	snapshot, err := repoanalysis.DiscoverGo(f.root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

// TestPreflightPassesDriftedClosureBindings holds the preflight to judging a
// catalogued literal that only moved offset as the gate's rebind will: as
// covered. Before, the preflight wrote no store and so could not rebind, a
// moved literal read as new policy, and the landing stopped for a hand-run
// rebind the gate would have done itself. A literal no decision covers is
// still refused.
func TestPreflightPassesDriftedClosureBindings(t *testing.T) {
	t.Parallel()
	fixture := newClosureFixture(t)
	if _, err := closurescan.ValidatePermanentActiveAuthority(fixture.write(t, catalogued), fixture.documents, fixture.aliases); err != nil {
		t.Fatalf("the catalogued literal was refused at its own offset: %v", err)
	}
	moved := fixture.write(t, "package policy\n\n// A comment above moves every offset below it.\nfunc admitted(n int) bool { return n > 4096 }\n")
	if _, err := closurescan.ValidatePermanentActiveAuthority(moved, fixture.documents, fixture.aliases); err == nil || !staleClosureAuthorityFailure(err) {
		t.Fatalf("a moved literal passed against its old offset: %v", err)
	}
	projection, err := projectedMagicBindings(moved, fixture.documents, fixture.aliases)
	if err != nil || projection.moved != 1 {
		t.Fatalf("projection moved %d: %v", projection.moved, err)
	}
	if _, err := closurescan.ValidatePermanentActiveAuthority(moved, projection.documents, projection.aliases); err != nil {
		t.Fatalf("a literal that only moved was refused after the projected rebind: %v", err)
	}
	added := fixture.write(t, "package policy\n\n// A comment above moves every offset below it.\nfunc admitted(n int) bool { return n > 4096 }\n\nfunc capped(n int) bool { return n < 8191 }\n")
	if projection, err = projectedMagicBindings(added, fixture.documents, fixture.aliases); err != nil {
		t.Fatal(err)
	}
	_, err = closurescan.ValidatePermanentActiveAuthority(added, projection.documents, projection.aliases)
	if _, uncatalogued := errors.AsType[*closurescan.UncataloguedPolicyError](err); !uncatalogued {
		t.Fatalf("a new literal passed the projected rebind: %v", err)
	}
}

// TestRemediationRetiresRemovedClosureCode holds the gate to retiring, after
// its rebind, a decision whose code is gone -- the one closure step that was
// still run by hand -- and to retiring nothing while an unmatched decision's
// value survives in its file, which a person must judge.
func TestRemediationRetiresRemovedClosureCode(t *testing.T) {
	t.Parallel()
	fixture := newClosureFixture(t)
	store, err := overgodb.Open(filepath.Join(fixture.root, gateStorePath))
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	var recorded []string
	gate := &gateContext{repo: fixture.root, storePath: gateStorePath, runCommand: func(_, name string, args ...string) (string, error) {
		recorded = append(recorded, name+" "+strings.Join(args, " "))
		return "imported 0 closure document(s), unmatched=1", nil
	}}

	removed := fixture.write(t, "package policy\n\nfunc admitted(n int) bool { return n > 0 }\n")
	stale := errors.New("permanent authority: 1 stale active binding(s)")
	if _, err := closurescan.ValidatePermanentActiveAuthority(removed, fixture.documents, fixture.aliases); err == nil {
		t.Fatal("a decision whose code is gone passed")
	}
	if _, err := gate.retireRemovedClosures(removed, fixture.documents, fixture.aliases, closurescan.AuthorityReport{}, stale); err != nil {
		t.Fatalf("removed code was not retired: %v", err)
	}
	if len(recorded) != 1 || !strings.HasSuffix(recorded[0], "-retire-unmatched") {
		t.Fatalf("retirement commands = %q", recorded)
	}

	recorded = nil
	survives := fixture.write(t, "package policy\n\nfunc allowed(m int) bool {\n\treturn m >= 4096\n}\n")
	_, err = gate.retireRemovedClosures(survives, fixture.documents, fixture.aliases, closurescan.AuthorityReport{}, stale)
	if err == nil || !strings.Contains(err.Error(), "judge them by hand") || len(recorded) != 0 {
		t.Fatalf("a decision whose value survives was retired: %v (commands %q)", err, recorded)
	}
}
