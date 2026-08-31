package gate

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/plan"
)

func TestGateCommitAdvancesPlanAtomically(t *testing.T) {
	repo := t.TempDir()
	if err := os.Mkdir(filepath.Join(repo, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	document := plan.Plan{Campaign: "test", Doctrine: "test", Items: []plan.Item{{
		ID: "automation", Title: "automation", Status: "open", Steps: []plan.Step{
			{ID: "first", Title: "first", Status: "open", Verify: "go test ./..."},
			{ID: "second", Title: "second", Status: "open", Verify: "go test ./..."},
		},
	}}}
	path := filepath.Join(repo, filepath.FromSlash(plan.Path))
	if err := plan.Save(path, document); err != nil {
		t.Fatal(err)
	}
	original, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	after, err := advancedPlanBytes(original, "automation/first")
	if err != nil {
		t.Fatal(err)
	}
	preparation, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte(t.Name()))
	if err != nil {
		t.Fatal(err)
	}
	rollback, err := publishPlanTransition(repo, preparation, original, after, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	advanced, err := plan.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(advanced.Items) != 1 || advanced.Items[0].ID != "automation" ||
		len(advanced.Items[0].Steps) != 1 || advanced.Items[0].Steps[0].ID != "second" {
		t.Fatalf("advanced plan = %+v", advanced.Items)
	}
	if err := rollback(); err != nil {
		t.Fatal(err)
	}
	restored, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(restored) != string(original) {
		t.Fatal("failed commit did not restore the original plan bytes")
	}
	rollback, err = publishPlanTransition(repo, preparation, original, after, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	document.Doctrine = "changed after publication"
	if err := plan.Save(path, document); err != nil {
		t.Fatal(err)
	}
	concurrentRollback, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := rollback(); err == nil {
		t.Fatal("plan rollback overwrote concurrent bytes")
	}
	retainedRollback, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(retainedRollback) != string(concurrentRollback) {
		t.Fatal("refused rollback changed concurrent plan bytes")
	}
	document.Doctrine = "test"
	if err := plan.Save(path, document); err != nil {
		t.Fatal(err)
	}
	document.Doctrine = "changed concurrently"
	if err := plan.Save(path, document); err != nil {
		t.Fatal(err)
	}
	concurrent, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := publishPlanTransition(repo, preparation, original, after, 0o644); err == nil {
		t.Fatal("plan transition overwrote concurrent bytes")
	}
	retained, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(retained) != string(concurrent) {
		t.Fatal("refused transition changed concurrent plan bytes")
	}
}
