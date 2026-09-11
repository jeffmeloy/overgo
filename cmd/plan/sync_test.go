package main

import (
	"encoding/json"
	"io"
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

func TestPrepareMergeFirstParentTargetRetainsOnlyLocalPlan(t *testing.T) {
	item := func(id string) plan.Item {
		return plan.Item{ID: id, Status: plan.StatusOpen, Steps: []plan.Step{{
			ID: "do", Status: plan.StatusOpen, Verify: "go test ./...",
		}}}
	}
	base := plan.Plan{Campaign: "campaign", Doctrine: "doctrine", Items: []plan.Item{item("shared")}}
	local := clonePlan(t, base)
	local.Items = append(local.Items, item("local-only"))
	incoming := clonePlan(t, base)
	incoming.Items = append(incoming.Items, item("incoming-only"))

	projected, err := projectMergePlan(
		base, local, incoming, plan.CompletionAuthority{}, plan.CompletionAuthority{},
		plan.MergeProjectionFirstParentTarget,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.EqualFunc(projected.Items, local.Items, func(left, right plan.Item) bool {
		return left.ID == right.ID
	}) || slices.ContainsFunc(projected.Items, func(item plan.Item) bool { return item.ID == "incoming-only" }) {
		t.Fatalf("target projection = %+v, want local identities only", projected.Items)
	}
	projected, err = insertItem(
		projected, "merge-0123456789ab", "Merge source", "", "go run ./cmd/compatibility -check",
	)
	if err != nil || projected.Items[0].ID != "merge-0123456789ab" ||
		projected.Items[1].ID != "shared" || projected.Items[2].ID != "local-only" {
		t.Fatalf("target merge row projection = %+v, %v", projected.Items, err)
	}
}

func TestMergeFinalizeProjectionArguments(t *testing.T) {
	if got, err := mergeFinalizeProjectionArguments(plan.MergeProjectionSemanticUnion, `C:\source lane\overgodb-store`); err != nil || got != "" {
		t.Fatalf("semantic-union arguments = %q, %v", got, err)
	}

	source := filepath.Join(t.TempDir(), "source lane", "overgodb-store")
	want := ` -plan-projection first-parent-target -merge-source-store "` + source + `"`
	got, err := mergeFinalizeProjectionArguments(plan.MergeProjectionFirstParentTarget, source)
	if err != nil || got != want {
		t.Fatalf("first-parent-target arguments = %q, %v, want %q", got, err, want)
	}
	if _, err := mergeFinalizeProjectionArguments(plan.MergeProjectionFirstParentTarget, ""); err == nil ||
		!strings.Contains(err.Error(), "requires its live source store") {
		t.Fatalf("missing first-parent-target source store error = %v", err)
	}
}

func TestFirstParentTargetSourceStoreRequiresRegisteredWorktree(t *testing.T) {
	repository, runGit := mergeTestRepository(t)
	revision := runGit("rev-parse", "HEAD")
	if _, err := firstParentTargetSourceStore(t.Context(), repository, revision, ""); err == nil ||
		!strings.Contains(err.Error(), "registered source worktree") {
		t.Fatalf("missing source store error = %v", err)
	}

	unregistered := filepath.Join(t.TempDir(), "overgodb-store")
	if err := os.Mkdir(unregistered, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := firstParentTargetSourceStore(t.Context(), repository, revision, unregistered); err == nil {
		t.Fatalf("unregistered source store error = %v", err)
	}

	sourceRoot := filepath.Join(t.TempDir(), "source lane")
	runGit("worktree", "add", "--detach", sourceRoot, revision)
	sourceStore := filepath.Join(sourceRoot, "overgodb-store")
	if err := os.Mkdir(sourceStore, 0o755); err != nil {
		t.Fatal(err)
	}
	got, err := firstParentTargetSourceStore(t.Context(), repository, revision, sourceStore)
	if err != nil {
		t.Fatal(err)
	}
	gotInfo, err := os.Stat(got)
	if err != nil {
		t.Fatal(err)
	}
	wantInfo, err := os.Stat(sourceStore)
	if err != nil {
		t.Fatal(err)
	}
	if !os.SameFile(gotInfo, wantInfo) {
		t.Fatalf("registered source store = %q, want %q", got, sourceStore)
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
	first, err := source.Commit(t.Context(), mergeEvidenceBatch(t, "first"))
	closeErr := source.Close()
	if err != nil || closeErr != nil {
		t.Fatalf("seed source = %s, %v, close=%v", first, err, closeErr)
	}

	frozen, err := captureClosureEvidence(t.Context(), root, "HEAD", snapshot)
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
	second, err := source.Commit(t.Context(), mergeEvidenceBatch(t, "second"))
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
	frozen.cleanup()
	advanced, err := captureClosureEvidence(t.Context(), root, "HEAD", snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if advanced.store != frozen.store || advanced.head != second {
		t.Fatalf("workspace/extent not reused: %+v", advanced)
	}
	advanced.cleanup()
	reused, err := captureClosureEvidence(t.Context(), root, "HEAD", snapshot)
	if err != nil {
		t.Fatal(err)
	}
	defer reused.cleanup()
	if reused.backup.BytesCopied != 0 || reused.backup.FilesReused == 0 {
		t.Fatalf("retry recopied evidence: %+v", reused.backup)
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

func TestPrepareMergeRejectsAmbiguousMergeBaseBeforeProjection(t *testing.T) {
	repository, runGit := mergeTestRepository(t)
	root := runGit("rev-parse", "HEAD")
	tree := runGit("rev-parse", "HEAD^{tree}")
	commitTree := func(message string, parents ...string) string {
		t.Helper()
		arguments := []string{"commit-tree", tree}
		for _, parent := range parents {
			arguments = append(arguments, "-p", parent)
		}
		arguments = append(arguments, "-m", message)
		return runGit(arguments...)
	}
	left := commitTree("left", root)
	right := commitTree("right", root)
	leftMerge := commitTree("left merge", left, right)
	rightMerge := commitTree("right merge", right, left)

	// A criss-cross history (each side merged the other) has two bases;
	// prepare-merge takes the one on the local side's first-parent chain.
	if base, err := uniqueMergeBase(repository, leftMerge, rightMerge); err != nil || base != left {
		t.Fatalf("criss-cross prepare-merge base = %q, %v; want %q", base, err, left)
	}
	// A local side whose first-parent chain holds none of the bases refuses.
	fork := commitTree("fork", root)
	forkMerge := commitTree("fork merge", fork, leftMerge)
	if _, err := uniqueMergeBase(repository, forkMerge, rightMerge); err == nil ||
		!strings.Contains(err.Error(), "first-parent chain, found 0 of 2") {
		t.Fatalf("ambiguous prepare-merge base error = %v", err)
	}
}

func TestPlanGitAuthorityIgnoresAmbientRepositoryOverrides(t *testing.T) {
	repository, runGit := mergeTestRepository(t)
	foreign, _ := mergeTestRepository(t)
	t.Setenv("GIT_DIR", filepath.Join(foreign, ".git"))
	t.Setenv("GIT_WORK_TREE", foreign)

	root, err := gitOutput(repository, "rev-parse", "--show-toplevel")
	if err != nil {
		t.Fatal(err)
	}
	want, err := os.Stat(repository)
	if err != nil {
		t.Fatal(err)
	}
	got, err := os.Stat(strings.TrimSpace(string(root)))
	if err != nil {
		t.Fatal(err)
	}
	if !os.SameFile(got, want) {
		t.Fatalf("Git authority root = %q, want %q", strings.TrimSpace(string(root)), repository)
	}
	if head := runGit("rev-parse", "HEAD"); head == "" {
		t.Fatal("repository head is absent")
	}
}

func TestPrepareMergeRequiresExactRepositoryRoot(t *testing.T) {
	repository, _ := mergeTestRepository(t)
	subdirectory := filepath.Join(repository, "nested")
	if err := os.Mkdir(subdirectory, 0o755); err != nil {
		t.Fatal(err)
	}
	err := prepareMerge(subdirectory, "HEAD", io.Discard)
	if err == nil || !strings.Contains(err.Error(), "exact repository root") {
		t.Fatalf("subdirectory prepare merge error = %v", err)
	}
}

func TestMergeOwnedDocumentLimitsAutomaticConflictResolution(t *testing.T) {
	for _, path := range []string{
		plan.Path,
		compatibilityDocumentPath,
		trainingCompatibilityDocumentPath,
		apiManifestJSONPath,
		modernGoBaselinePath,
		modernGoCensusPath,
	} {
		if !mergeOwnedDocument(path) {
			t.Fatalf("owned generated document %q was rejected", path)
		}
	}
	for _, path := range []string{"go.mod", "internal/server/routes.go", "README.md"} {
		if mergeOwnedDocument(path) {
			t.Fatalf("handwritten source %q was accepted", path)
		}
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
