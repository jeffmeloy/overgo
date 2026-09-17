package gate

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"overgo/internal/closurescan"
	"overgo/internal/repoanalysis"
)

// TestGateStagesMechanicalRepairs pins the repair registry: before the
// candidate freezes a Go change has its planned files formatted, the
// modern-Go census and baseline and the API
// manifest republished through their owners and bound into the
// planned paths, and the harness surface baseline left alone when nothing
// tightened; each repair is audited with the files it rewrote. A repair
// that would raise a reviewed ceiling refuses with the delta, a census
// refusal stops the gate, and a commit without Go input stages nothing.
func TestGateStagesMechanicalRepairs(t *testing.T) {
	t.Parallel()
	repo := modernMemoFixture(t)
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
	modernMemoPublish(t, repo)
	priorBaseline := read("docs/modern_go_baseline.json")
	write("docs/modern_go_census.json", "old census\n")
	snapshot, err := repoanalysis.DiscoverGo(repo, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	surface, err := closurescan.BuildAgentHarnessSurface(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.MarshalIndent(surface, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	write(harnessSurfaceBaselineFile, string(encoded)+"\n")

	var recorded []string
	var recordedMutex sync.Mutex
	fake := func(root, name string, args ...string) (string, error) {
		invocation := name + " " + strings.Join(args, " ")
		recordedMutex.Lock()
		recorded = append(recorded, invocation)
		recordedMutex.Unlock()
		switch {
		case strings.Contains(invocation, "api-manifest -update"):
			write(apiManifestFile, "new manifest\n")
			return "wrote", nil
		}
		t.Fatalf("unexpected command %q", invocation)
		return "", nil
	}
	write(apiManifestFile, "old manifest\n")
	g := &gateContext{repo: repo, storePath: gateStorePath, paths: []string{"internal/plan/a.go"}, runCommand: fake}
	if err := g.stageMechanicalRepairs(); err != nil {
		t.Fatal(err)
	}
	if got := read("internal/plan/a.go"); got != "package x\n\nfunc A() {}\n" {
		t.Fatalf("formatted source = %q", got)
	}
	if read("docs/modern_go_census.json") == "old census\n" || read("docs/modern_go_baseline.json") == priorBaseline {
		t.Fatal("the census repair did not publish the formatted source")
	}
	if read(apiManifestFile) != "new manifest\n" {
		t.Fatal("the API manifest repair did not rewrite through its command")
	}
	wantPaths := []string{"internal/plan/a.go", repoanalysis.ModernGoPublishedCensusFile, repoanalysis.ModernGoBaselineFile, apiManifestFile}
	if !slices.Equal(g.paths, wantPaths) {
		t.Fatalf("planned paths = %v, want %v", g.paths, wantPaths)
	}
	if read(harnessSurfaceBaselineFile) != string(encoded)+"\n" {
		t.Fatal("an unchanged harness surface rewrote its baseline")
	}
	staged := 0
	for _, line := range g.audit {
		if strings.HasPrefix(line, "staged repair: ") {
			staged++
		}
		if strings.HasPrefix(line, "staged repair: modern-Go census") && !strings.Contains(line, "rewrote=["+repoanalysis.ModernGoPublishedCensusFile+","+repoanalysis.ModernGoBaselineFile+"]") {
			t.Fatalf("census repair audit = %q", line)
		}
		if strings.HasPrefix(line, "staged repair: gofmt") && !strings.Contains(line, "rewrote=[internal/plan/a.go]") {
			t.Fatalf("gofmt repair audit = %q", line)
		}
	}
	if staged != 4 {
		t.Fatalf("staged repairs audited = %d, want four derived-file repairs: %q", staged, g.audit)
	}
	if len(recorded) != 1 {
		t.Fatalf("commands = %v, want only the manifest update", recorded)
	}
	// A retry starts with the caller's original paths and already repaired files.
	g.paths = g.paths[:1]
	if err := g.stageMechanicalRepairs(); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(g.paths, wantPaths) {
		t.Fatalf("retry omitted byte-identical repaired outputs: got %v, want %v", g.paths, wantPaths)
	}

	raised := surface
	raised.ProductionFiles = 0
	encoded, err = json.MarshalIndent(raised, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	write(harnessSurfaceBaselineFile, string(encoded)+"\n")
	err = g.stageMechanicalRepairs()
	if err == nil || !strings.Contains(err.Error(), "reviewed threshold") || !strings.Contains(err.Error(), "production_files 0 -> 1") {
		t.Fatalf("raised harness surface = %v, want the refusal with its delta", err)
	}
	if read(harnessSurfaceBaselineFile) != string(encoded)+"\n" {
		t.Fatal("a refused harness repair rewrote the baseline")
	}
	write(harnessSurfaceBaselineFile, string(mustIndent(t, surface))+"\n")

	write("internal/plan/a.go", "package x\nfunc A(value, other string) string { if value == \"\" { value = other }; return value }\n")
	err = g.stageMechanicalRepairs()
	if err == nil || !strings.Contains(err.Error(), "staged repair modern-Go census refused") || !strings.Contains(err.Error(), "debt increased") {
		t.Fatalf("census refusal = %v, want the gate stopped with the reason", err)
	}

	docs := &gateContext{repo: repo, storePath: gateStorePath, paths: []string{"docs/plan.json"}, runCommand: fake}
	recorded = nil
	if err := docs.stageMechanicalRepairs(); err != nil || len(recorded) != 0 || len(docs.audit) != 0 {
		t.Fatalf("plan-only change staged %v with %v, want nothing", recorded, err)
	}
}

func mustIndent(t *testing.T, surface closurescan.HarnessSurface) []byte {
	t.Helper()
	encoded, err := json.MarshalIndent(surface, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}
