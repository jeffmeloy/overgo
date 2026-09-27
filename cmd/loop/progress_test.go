package main

import (
	"os"
	"os/exec"
	"strings"
	"testing"

	"overgo/internal/gitauthority"
	"overgo/internal/loop"
	"overgo/internal/plan"
)

// TestCampaignAcceptedProgress exercises execWorld against real Git
// transitions: a commit that names no row earns the open row nothing, and the
// landing of a prerequisite it waits on is progress, read from the landing's
// trailer.
func TestCampaignAcceptedProgress(t *testing.T) {
	root := t.TempDir()
	t.Chdir(root)
	git := func(args ...string) string {
		t.Helper()
		command := exec.CommandContext(t.Context(), "git", args...)
		command.Env = gitauthority.RepositoryEnvironment()
		output, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v: %s", args, err, output)
		}
		return strings.TrimSpace(string(output))
	}
	git("init", "-q")
	git("config", "user.name", "progress-test")
	git("config", "user.email", "progress@test.invalid")
	if err := os.MkdirAll("docs", 0o700); err != nil {
		t.Fatal(err)
	}
	const verify = "go test ./..."
	document := plan.Plan{Campaign: "progress fixture", Doctrine: "test", Items: []plan.Item{{ID: "grammar", Owner: "colibri", Status: plan.StatusOpen, Steps: []plan.Step{
		{ID: "do", Status: plan.StatusOpen, Verify: verify, DependsOn: []string{"grammar/binding"}},
		{ID: "binding", Status: plan.StatusOpen, Verify: verify},
	}}}}
	if err := plan.Save(plan.Path, document); err != nil {
		t.Fatal(err)
	}
	git("add", "--", plan.Path)
	git("commit", "-qm", "baseline")
	world := &execWorld{}
	step := loop.Step{Item: "grammar", ID: "do"}
	before := git("rev-parse", "HEAD")
	git("commit", "-q", "--allow-empty", "-m", "unrelated work")
	if accepted, err := world.AcceptedProgress(step, before); err != nil || accepted {
		t.Fatalf("a commit naming no row earned progress: %v, %v", accepted, err)
	}
	git("commit", "-q", "--allow-empty", "-m", "binding\n\n"+plan.LandedRowTrailer+": grammar/binding")
	if accepted, err := world.AcceptedProgress(step, before); err != nil || !accepted {
		t.Fatalf("a landed prerequisite earned no progress: %v, %v", accepted, err)
	}
}
