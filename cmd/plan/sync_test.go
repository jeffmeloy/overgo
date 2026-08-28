package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
)

func TestPrepareMergeSnapshotsSourceAndRegeneratesDerivedDocs(t *testing.T) {
	item := func(id string) plan.Item {
		return plan.Item{ID: id, Title: id, Status: "open", Steps: []plan.Step{{
			ID: "do", Title: id, Status: "open", Verify: "go test ./...",
		}}}
	}
	base := plan.Plan{Campaign: "campaign", Doctrine: "doctrine", Items: []plan.Item{item("base-local"), item("base-upstream")}}
	local := plan.Plan{Campaign: base.Campaign, Doctrine: base.Doctrine, Items: []plan.Item{item("base-local"), item("base-upstream"), item("local-new")}}
	upstream := plan.Plan{Campaign: base.Campaign, Doctrine: base.Doctrine, Items: []plan.Item{item("base-local"), item("base-upstream"), item("upstream-new")}}
	local.Items[0].Status = plan.StatusDone
	local.Items[0].Steps[0].Status = plan.StatusDone
	upstream.Items[1].Status = plan.StatusDone
	upstream.Items[1].Steps[0].Status = plan.StatusDone
	merged, err := plan.MergeDocuments(base, local, upstream)
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]string, len(merged.Items))
	for index, mergedItem := range merged.Items {
		ids[index] = mergedItem.ID
	}
	if !slices.Equal(ids, []string{"base-local", "base-upstream", "local-new", "upstream-new"}) ||
		merged.Items[0].Status != plan.StatusDone || merged.Items[1].Status != plan.StatusDone {
		t.Fatalf("merged retained document = %v", merged.Items)
	}
	merged, err = insertItem(merged, "merge-0123456789ab", "Merge topic at 0123456789ab", "", "go run ./cmd/compatibility -check")
	if err != nil || merged.Items[0].ID != "merge-0123456789ab" || merged.Items[0].Steps[0].Verify != "go run ./cmd/compatibility -check" {
		t.Fatalf("prepared merge plan = %+v, %v", merged.Items[0], err)
	}
	local = plan.Plan{Campaign: base.Campaign, Doctrine: base.Doctrine, Items: slices.Clone(base.Items)}
	upstream = plan.Plan{Campaign: base.Campaign, Doctrine: base.Doctrine, Items: slices.Clone(base.Items)}
	local.Items[0].Title = "local edit"
	upstream.Items[0].Title = "upstream edit"
	if _, err := plan.MergeDocuments(base, local, upstream); err == nil {
		t.Fatal("concurrent edits to one live plan row were guessed")
	}

	local = plan.Plan{Campaign: base.Campaign, Doctrine: base.Doctrine, Items: slices.Clone(base.Items)}
	upstream = plan.Plan{Campaign: base.Campaign, Doctrine: base.Doctrine, Items: slices.Clone(base.Items)}
	local.Items = local.Items[1:]
	if _, err := plan.MergeDocuments(base, local, upstream); err == nil {
		t.Fatal("legacy row deletion was accepted as completion")
	}

	base.Items[0].Steps[0].Rationale = "why"
	base.Items[0].Steps[0].DependsOn = []string{"base-upstream/do"}
	base.Items[0].Steps[0].Capabilities = []string{"runtime"}
	base.Items[0].Steps[0].Outcome = json.RawMessage(`{"verdict":"pass"}`)
	local = clonePlan(t, base)
	upstream = clonePlan(t, base)
	local.Items[0].Steps[0].Status = plan.StatusDone
	local.Items[0].Status = plan.StatusDone
	merged, err = plan.MergeDocuments(base, local, upstream)
	if err != nil {
		t.Fatal(err)
	}
	step := merged.Items[0].Steps[0]
	if step.Rationale != "why" || !slices.Equal(step.DependsOn, []string{"base-upstream/do"}) ||
		!slices.Equal(step.Capabilities, []string{"runtime"}) || string(step.Outcome) != `{"verdict":"pass"}` {
		t.Fatalf("merged step lost fields: %+v", step)
	}

	base.Items[0].Owner = ""
	local = clonePlan(t, base)
	upstream = clonePlan(t, base)
	local.Items[0].Owner = "research"
	merged, err = plan.MergeDocuments(base, local, upstream)
	if err != nil || merged.Items[0].Owner != "research" {
		t.Fatalf("merged owner = %q, %v", merged.Items[0].Owner, err)
	}
	upstream.Items[0].Owner = "developer"
	if _, err := plan.MergeDocuments(base, local, upstream); err == nil {
		t.Fatal("concurrent owner edits were guessed")
	}
}

func clonePlan(t *testing.T, document plan.Plan) plan.Plan {
	t.Helper()
	data, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	cloned, err := plan.Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	return cloned
}

func TestPrepareMergeSnapshotEvidence(t *testing.T) {
	root, runGit := mergeTestRepository(t)
	snapshot := runGit("rev-parse", "HEAD")
	source, err := overgodb.Open(filepath.Join(root, "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	first, err := source.Commit(context.Background(), mergeEvidenceBatch(t, "first"))
	closeErr := source.Close()
	if err != nil || closeErr != nil {
		t.Fatalf("seed source = %s, %v, close=%v", first, err, closeErr)
	}

	frozen, err := captureClosureEvidence(root, snapshot)
	if err != nil {
		t.Fatal(err)
	}
	defer frozen.cleanup()
	if frozen.head != first || frozen.sequence != 1 || frozen.store == "" {
		t.Fatalf("snapshot = %+v", frozen)
	}
	source, err = overgodb.Open(filepath.Join(root, "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	second, err := source.Commit(context.Background(), mergeEvidenceBatch(t, "second"))
	closeErr = source.Close()
	if err != nil || closeErr != nil || second == first {
		t.Fatalf("advance source = %s, %v, close=%v", second, err, closeErr)
	}
	copyStore, err := overgodb.OpenReadOnly(frozen.store)
	if err != nil {
		t.Fatal(err)
	}
	head, sequence := copyStore.Head()
	if err := copyStore.Close(); err != nil || head != first || sequence != 1 {
		t.Fatalf("frozen snapshot advanced = %s@%d, close=%v", head, sequence, err)
	}
}

func mergeEvidenceBatch(t *testing.T, key string) artifact.Batch {
	t.Helper()
	payload := []byte(key)
	id, err := artifact.IdentifyBytes(artifact.KindEvidence, payload)
	if err != nil {
		t.Fatal(err)
	}
	return artifact.Batch{Key: key, Artifacts: []artifact.Descriptor{{
		ID: id, Size: uint64(len(payload)), MediaType: "application/octet-stream",
	}}}
}

func TestPrepareMergeSourceAdvance(t *testing.T) {
	root, runGit := mergeTestRepository(t)
	first := runGit("rev-parse", "HEAD")
	if err := os.WriteFile(filepath.Join(root, "fixture"), []byte("two"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGit("commit", "-am", "second")
	latest, err := gitOutput(root, "rev-parse", "--verify", "HEAD^{commit}")
	if err != nil {
		t.Fatal(err)
	}
	if err := verifySourceSnapshot("HEAD", first, latest, nil); err == nil {
		t.Fatal("advanced source snapshot was accepted")
	}
}

func mergeTestRepository(t *testing.T) (string, func(...string) string) {
	t.Helper()
	root := t.TempDir()
	runGit := func(arguments ...string) string {
		t.Helper()
		output, err := commandOutput(root, "git", arguments...)
		if err != nil {
			t.Fatal(err)
		}
		return strings.TrimSpace(string(output))
	}
	runGit("init")
	runGit("config", "user.email", "overgo@example.invalid")
	runGit("config", "user.name", "Overgo Test")
	if err := os.WriteFile(filepath.Join(root, "fixture"), []byte("one"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGit("add", "fixture")
	runGit("commit", "-m", "first")
	return root, runGit
}
