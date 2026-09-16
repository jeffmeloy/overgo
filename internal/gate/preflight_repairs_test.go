package gate

import (
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"overgo/internal/closurescan"
	"overgo/internal/repoanalysis"
)

// TestPreflightAppliesMechanicalRepairs pins the preflight's repairs: a Go
// change has the working tree formatted and the census and API manifest
// republished, printed with the files each rewrote, while the closure
// rebind that writes the store is left to the gate; a change to a file the
// compatibility manifest pins as evidence refreshes the identities and the
// matrix and nothing else; a change reaching no repair stages none.
func TestPreflightAppliesMechanicalRepairs(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	write := func(path, content string) {
		full := filepath.Join(repo, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	read := func(path string) string {
		data, err := os.ReadFile(filepath.Join(repo, filepath.FromSlash(path)))
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}
	write("internal/plan/a.go", "package x\n\nfunc  A( ) {}\n")
	write("docs/evidence/capture.txt", "capture\n")
	write(apiManifestFile, "old manifest\n")
	write(compatibilityManifestFile, `{"claims":[{"id":"c","evidence":[{"path":"docs/evidence/capture.txt","identity":"old"}]}]}`+"\n")
	write(compatibilityMatrixFile, "old matrix\n")
	write("docs/modern_go_census.json", "old census\n")
	write("docs/modern_go_baseline.json", "old baseline\n")
	write(harnessSurfaceBaselineFile, string(mustIndent(t, liveHarnessSurface(t, repo)))+"\n")

	var recorded []string
	var mutex sync.Mutex
	fake := func(root, name string, args ...string) (string, error) {
		invocation := name + " " + strings.Join(args, " ")
		mutex.Lock()
		recorded = append(recorded, invocation)
		mutex.Unlock()
		switch {
		case strings.Contains(invocation, "-publish-census"):
			write("docs/modern_go_census.json", "new census\n")
			write("docs/modern_go_baseline.json", "new baseline\n")
		case strings.Contains(invocation, "api-manifest -update"):
			write(apiManifestFile, "new manifest\n")
		case strings.Contains(invocation, "compatibility -refresh-identities"):
			write(compatibilityManifestFile, `{"claims":[{"id":"c","evidence":[{"path":"docs/evidence/capture.txt","identity":"new"}]}]}`+"\n")
			write(compatibilityMatrixFile, "new matrix\n")
		default:
			t.Errorf("unexpected command %q", invocation)
		}
		return "", nil
	}
	var output bytes.Buffer
	g := &gateContext{repo: repo, storePath: gateStorePath, paths: []string{"internal/plan/a.go"}, runCommand: fake, preflight: true}
	if err := g.preflightRepairs(&output); err != nil {
		t.Fatal(err)
	}
	if read("internal/plan/a.go") != "package x\n\nfunc A() {}\n" || read(apiManifestFile) != "new manifest\n" || read("docs/modern_go_census.json") != "new census\n" {
		t.Fatal("the preflight did not apply the derived-file repairs to the working tree")
	}
	if read(compatibilityMatrixFile) != "old matrix\n" {
		t.Fatal("a Go change without pinned evidence refreshed the compatibility identities")
	}
	for _, invocation := range recorded {
		if strings.Contains(invocation, "closure-scan") {
			t.Fatalf("the preflight wrote the store: %v", recorded)
		}
	}
	for _, want := range []string{
		"preflight: staged repair: gofmt for the fmt phase", "rewrote=[internal/plan/a.go]",
		"preflight: staged repair: modern-Go census", "preflight: staged repair: API manifest for the test phase", "rewrote=[" + apiManifestFile + "]",
	} {
		if !strings.Contains(output.String(), want) {
			t.Errorf("preflight output lacks %q:\n%s", want, output.String())
		}
	}
	if strings.Contains(output.String(), "closure rebind") || strings.Contains(output.String(), "compatibility identities") {
		t.Fatalf("preflight output names a repair it did not apply:\n%s", output.String())
	}
	wantPaths := []string{"internal/plan/a.go", "docs/modern_go_census.json", "docs/modern_go_baseline.json", apiManifestFile}
	if !slices.Equal(g.paths, wantPaths) {
		t.Fatalf("planned paths = %v, want %v", g.paths, wantPaths)
	}

	recorded = nil
	output.Reset()
	evidence := &gateContext{repo: repo, storePath: gateStorePath, paths: []string{"docs/evidence/capture.txt"}, runCommand: fake, preflight: true}
	if err := evidence.preflightRepairs(&output); err != nil {
		t.Fatal(err)
	}
	if len(recorded) != 1 || !strings.Contains(recorded[0], "compatibility -refresh-identities") {
		t.Fatalf("evidence change ran %v, want the identity refresh alone", recorded)
	}
	if read(compatibilityMatrixFile) != "new matrix\n" || !strings.Contains(read(compatibilityManifestFile), `"identity":"new"`) {
		t.Fatal("the identity refresh did not rewrite the manifest and the matrix")
	}
	if !strings.Contains(output.String(), "preflight: staged repair: compatibility identities for the claims phase") || !strings.Contains(output.String(), "rewrote=["+compatibilityManifestFile+","+compatibilityMatrixFile+"]") {
		t.Fatalf("evidence repair output:\n%s", output.String())
	}
	if !slices.Equal(evidence.paths, []string{"docs/evidence/capture.txt", compatibilityManifestFile, compatibilityMatrixFile}) {
		t.Fatalf("planned paths = %v", evidence.paths)
	}
	// The gate applies the same registry with the store repair.
	gate := &gateContext{repo: repo, storePath: gateStorePath, paths: []string{compatibilityManifestFile}, runCommand: fake}
	if !gate.compatibilityEvidenceChanged() {
		t.Fatal("a change to the compatibility manifest itself does not refresh its identities")
	}
	if slices.ContainsFunc(gate.mechanicalRepairs(), func(repair mechanicalRepair) bool { return repair.name == "closure rebind" && !repair.store }) {
		t.Fatal("the closure rebind is not marked as a store repair")
	}

	recorded = nil
	output.Reset()
	docs := &gateContext{repo: repo, storePath: gateStorePath, paths: []string{"docs/plan.json"}, runCommand: fake, preflight: true}
	if err := docs.preflightRepairs(&output); err != nil || len(recorded) != 0 || output.Len() != 0 {
		t.Fatalf("plan-only change staged %v with %v and printed %q, want nothing", recorded, err, output.String())
	}
}

// liveHarnessSurface measures the fixture's harness surface so its baseline
// holds and the repair leaves it alone.
func liveHarnessSurface(t *testing.T, repo string) closurescan.HarnessSurface {
	t.Helper()
	snapshot, err := repoanalysis.DiscoverGo(repo, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	surface, err := closurescan.BuildAgentHarnessSurface(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	return surface
}
