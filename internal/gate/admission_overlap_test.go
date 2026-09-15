package gate

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"overgo/internal/plan"
)

// TestAdmissionStepsOverlap pins the repeated work the admission chain no
// longer does: the planned tree is built once per candidate state and
// rebuilt only when a planned path or HEAD changes, the plan is parsed once
// for the gate's readers, and the staged repairs run in one wave after the
// formatter, so the chain's wall is the longest repair, not their sum.
func TestAdmissionStepsOverlap(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "gate@test")
	runGitFixture(t, repo, "config", "user.name", "gate")
	write := func(path, content string) {
		full := filepath.Join(repo, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("internal/plan/a.go", "package plan\n\nfunc A() {}\n")
	write("other.txt", "unplanned\n")
	document := plan.Plan{Lane: "hatchet", Items: []plan.Item{{ID: "row", Title: "Row", Status: plan.StatusOpen, Steps: []plan.Step{{ID: "do", Title: "Do", Status: plan.StatusOpen, Verify: "true"}}}}}
	if err := os.MkdirAll(filepath.Join(repo, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := plan.Save(filepath.Join(repo, filepath.FromSlash(plan.Path)), document); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "-A")
	runGitFixture(t, repo, "commit", "-q", "-m", "fixture")

	g := &gateContext{repo: repo, storePath: gateStorePath, paths: []string{"internal/plan/a.go"}}
	first, err := g.plannedTree()
	if err != nil {
		t.Fatal(err)
	}
	if again, err := g.plannedTree(); err != nil || again != first || g.plannedTreeBuilds != 1 {
		t.Fatalf("unchanged candidate rebuilt its tree: builds=%d tree=%s err=%v", g.plannedTreeBuilds, again, err)
	}
	write("other.txt", "still unplanned\n")
	if again, err := g.plannedTree(); err != nil || again != first || g.plannedTreeBuilds != 1 {
		t.Fatalf("an unplanned edit rebuilt the tree: builds=%d err=%v", g.plannedTreeBuilds, err)
	}
	write("internal/plan/a.go", "package plan\n\nfunc A() { _ = 1 }\n")
	changed, err := g.plannedTree()
	if err != nil || changed == first || g.plannedTreeBuilds != 2 {
		t.Fatalf("a planned edit did not rebuild the tree: builds=%d same=%t err=%v", g.plannedTreeBuilds, changed == first, err)
	}
	if err := g.requireCandidateTree(candidateTreeKey(changed)); err != nil {
		t.Fatal(err)
	}
	if err := g.requireCandidateTree(candidateTreeKey(first)); err == nil {
		t.Fatal("the drift check accepted the earlier candidate")
	}

	for range 3 {
		if _, err := g.loadPlan(); err != nil {
			t.Fatal(err)
		}
	}
	if g.planLoads != 1 || g.planDocument == nil || g.planDocument.Lane != "hatchet" {
		t.Fatalf("plan parsed %d times", g.planLoads)
	}

	write("docs/modern_go_census.json", "old\n")
	write("docs/modern_go_baseline.json", "old\n")
	// The closure rebind and the census run in one wave: they prove it by
	// meeting inside the command runner.
	rebindKey, publishKey := filepath.Join(repo, gateStorePath), "-lower-baseline"
	wave := newRendezvous(rebindKey, publishKey)
	var mutex sync.Mutex
	ran := map[string]bool{}
	g.runCommand = func(root, name string, args ...string) (string, error) {
		key := args[len(args)-1]
		wave.meet(key)
		mutex.Lock()
		ran[key] = true
		mutex.Unlock()
		return "done", nil
	}
	if err := g.stageMechanicalRepairs(); err != nil {
		t.Fatal(err)
	}
	if !ran[rebindKey] || !ran[publishKey] || len(ran) != 2 {
		t.Fatalf("repairs ran %v", ran)
	}
	staged := slices.IndexFunc(g.audit, func(line string) bool { return strings.HasPrefix(line, "staged repair: ") })
	if staged < 0 || !strings.HasPrefix(g.audit[staged], "staged repair: gofmt") {
		t.Fatalf("the formatter did not run first: %q", g.audit)
	}
}
