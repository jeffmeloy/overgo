package gate

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// TestPreflightProposesEveryClosureSite pins the one-pass triage: a refusal
// for uncatalogued production policy names every site; the scanner's
// policy inventory separates the new sites from the ones that drifted from
// catalogued offsets; the propose mode runs once for all the new sites
// against the store and its rows land in the proposals file; the refusal
// names the file, the triage command for the new sites and the rebind for
// the drifted ones; a catalogued tree proposes nothing.
func TestPreflightProposesEveryClosureSite(t *testing.T) {
	t.Parallel()
	root, existing := magicGateFixture(t, true)
	writeMagicSource(t, existing, "package p\n")
	writeMagicSource(t, filepath.Join(root, "internal", "p", "new.go"), "package p\n\nconst FirstLimit = 3\n\nconst SecondLimit = 5\n")
	var recorded []string
	var mutex sync.Mutex
	// The inventory projects SecondLimit onto catalogued history: it is
	// drift for the rebind, not a new site.
	const inventory = `{"counts":{"unclassified":1},"candidates":[{"current":{"name":"FirstLimit"}}],"triage_proposal":{"rows":null}}` + "\n"
	const proposal = `{"rows":[{"kind":"constant","name":"FirstLimit"}]}` + "\n"
	fake := func(dir, name string, args ...string) (string, error) {
		mutex.Lock()
		defer mutex.Unlock()
		invocation := name + " " + strings.Join(args, " ")
		recorded = append(recorded, invocation)
		if strings.Contains(invocation, "-inventory-unclassified-policy") {
			return inventory, nil
		}
		return proposal, nil
	}
	g := &gateContext{repo: root, paths: []string{"internal/p/new.go"}, storePath: "store", preflight: true, runCommand: fake}
	_, err := g.stepMagics()
	if err == nil {
		t.Fatal("uncatalogued sites passed the magics check")
	}
	message := err.Error()
	for _, want := range []string{
		"stale active binding", "ExistingLimit",
		"uncatalogued production policy FirstLimit at", "uncatalogued production policy SecondLimit at",
		"1 new site(s) have proposal rows in " + gateClosureProposalsFile, "-triage " + gateClosureProposalsFile,
		"1 site(s) drifted from catalogued offsets [SecondLimit]", "-import-store " + gateStorePath,
	} {
		if !strings.Contains(message, want) {
			t.Errorf("refusal lacks %q:\n%s", want, message)
		}
	}
	if len(recorded) != 2 || !strings.Contains(recorded[0], "./cmd/closure-scan -store store -inventory-unclassified-policy -format json") || !strings.HasSuffix(recorded[1], "./cmd/closure-scan -store store -propose FirstLimit") {
		t.Fatalf("commands = %v, want one inventory then one propose of the new site against the tree's store", recorded)
	}
	written, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(gateClosureProposalsFile)))
	if err != nil || string(written) != proposal {
		t.Fatalf("proposals file = %q, %v; want the scanner's rows", written, err)
	}

	// A tree whose sites are all catalogued proposes nothing.
	recorded = nil
	writeMagicSource(t, existing, "package p\nconst ExistingLimit = 8\n")
	if err := os.Remove(filepath.Join(root, "internal", "p", "new.go")); err != nil {
		t.Fatal(err)
	}
	catalogued := &gateContext{repo: root, paths: []string{"internal/p/p.go"}, storePath: "store", preflight: true, runCommand: fake}
	if _, err := catalogued.stepMagics(); err != nil || len(recorded) != 0 {
		t.Fatalf("catalogued tree = %v with commands %v, want a pass and no proposal", err, recorded)
	}
}
