package gate

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/plan"
)

func TestProjectedMergeCodeProvenance(t *testing.T) {
	repo := t.TempDir()
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "gate@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Gate Test")
	if err := os.Mkdir(filepath.Join(repo, "internal"), 0o755); err != nil {
		t.Fatal(err)
	}
	a := filepath.Join(repo, "internal", "a.go")
	b := filepath.Join(repo, "internal", "b.go")
	if err := os.WriteFile(a, []byte("package internal\nconst A = 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(b, []byte("package internal\nconst B = 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "internal")
	runGitFixture(t, repo, "commit", "-q", "-m", "local")
	local := recoveryGit(t, repo, "rev-parse", "HEAD")
	localTree := recoveryGit(t, repo, "rev-parse", "HEAD^{tree}")
	if err := os.WriteFile(a, []byte("package internal\nconst A = 2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(b); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "-A", "internal")
	runGitFixture(t, repo, "commit", "-q", "-m", "incoming")
	incoming := recoveryGit(t, repo, "rev-parse", "HEAD")
	g := &gateContext{
		repo: repo, planHead: local, mergeBefore: &gateMergeIntent{Head: []byte(incoming + "\n")},
		mergeSourceStore: "source", planProjection: plan.MergeProjectionFirstParentTarget,
		paths:     []string{"internal/a.go", "internal/b.go"},
		fixedTree: recoveryGit(t, repo, "rev-parse", "HEAD^{tree}"),
	}
	if covered, err := g.sourceProvenMergeGo(); err != nil || !covered {
		t.Fatalf("incoming edit and deletion were not parent-proven: covered=%t err=%v", covered, err)
	}
	g.fixedTree = localTree
	if covered, err := g.sourceProvenMergeGo(); err != nil || !covered {
		t.Fatalf("unchanged local blobs were not parent-proven: covered=%t err=%v", covered, err)
	}
	if err := os.WriteFile(a, []byte("package internal\nconst A = 3\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "internal/a.go")
	g.fixedTree = recoveryGit(t, repo, "write-tree")
	if covered, err := g.sourceProvenMergeGo(); err != nil || covered {
		t.Fatalf("new merge resolution was parent-proven: covered=%t err=%v", covered, err)
	}
	g.planProjection = plan.MergeProjectionSemanticUnion
	if covered, err := g.sourceProvenMergeGo(); err != nil || covered {
		t.Fatalf("semantic-union merge bypassed base proof: covered=%t err=%v", covered, err)
	}
}
