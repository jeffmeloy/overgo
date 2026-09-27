package gate

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/closurescan"
	"overgo/internal/gosyntax"
)

// TestUnchangedDerivedFilesStayOutOfTheShipSet holds a repaired derived file
// to the question the ship set asks: whether it differs from the last commit.
// Once the census baseline, the harness baseline and the API manifest are
// committed, a Go change whose repairs reproduce them byte for byte ships
// none of them, so they select no check the change cannot affect.
func TestUnchangedDerivedFilesStayOutOfTheShipSet(t *testing.T) {
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
	write("internal/plan/a.go", "package x\n\nfunc A() {}\n")
	modernMemoPublish(t, repo)
	snapshot, err := gosyntax.DiscoverGo(repo, "internal", "cmd")
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
	write(apiManifestFile, "manifest\n")
	runGitFixture(t, repo, "add", ".")
	runGitFixture(t, repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-q", "-m", "derived files committed")

	// The source changes; its repairs reproduce every derived file exactly.
	write("internal/plan/a.go", "package x\n\nfunc B() {}\n")
	unchanged := func(root, name string, args ...string) (string, error) {
		if invocation := name + " " + strings.Join(args, " "); !strings.Contains(invocation, "api-manifest -update") {
			t.Fatalf("unexpected command %q", invocation)
		}
		write(apiManifestFile, "manifest\n")
		return "wrote", nil
	}
	g := &gateContext{repo: repo, storePath: gateStorePath, paths: []string{"internal/plan/a.go"}, runCommand: unchanged}
	if err := g.stageMechanicalRepairs(); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(g.paths, []string{"internal/plan/a.go"}) {
		t.Fatalf("unchanged derived files joined the ship set: %v", g.paths)
	}
}
