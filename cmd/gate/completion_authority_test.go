package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/codemanifest"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestCompletedPlanIdentityCannotBeReused pins the gate-side trailer writer:
// an operator cannot smuggle a second completion identity into the message
// that the shared authority reader will later consume.
func TestCompletedPlanIdentityCannotBeReused(t *testing.T) {
	repository := t.TempDir()
	if err := os.Mkdir(filepath.Join(repository, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	document := completionAuthorityPlan()
	if err := plan.Save(filepath.Join(repository, filepath.FromSlash(plan.Path)), document); err != nil {
		t.Fatal(err)
	}
	messagePath := filepath.Join(repository, "message.txt")
	if err := os.WriteFile(messagePath, []byte("subject\n\nOvergo-Plan-Item: reused\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	manifest := testutil.ArtifactID(t, artifact.KindRecipe, "completion-plan")
	codeManifest := testutil.ArtifactID(t, artifact.KindProfile, "completion-code")
	preparation := testutil.ArtifactID(t, artifact.KindEvidence, "completion-preparation")
	preparationCommit := artifact.CommitID{1}
	gate := gateContext{
		repo: repository, planRef: "item/step", messageFile: messagePath,
		manifestPlan: &automationcheck.ManifestPlan{
			ID: manifest,
		},
		candidateManifest: &codemanifest.Manifest{
			ID: codeManifest,
		},
		preparation: runrecord.GateLifecycle{
			ID: preparation,
		},
		preparationCommit: preparationCommit,
	}
	if _, err := gate.completionMessageFile(document); err == nil {
		t.Fatal("operator-supplied completion identity was accepted")
	} else if !strings.Contains(err.Error(), "reserved trailer Overgo-Plan-Item") {
		t.Fatalf("completion identity refused for the wrong reason: %v", err)
	}
	if err := os.WriteFile(messagePath, []byte("subject\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	mutated := document
	mutated.Items = append([]plan.Item(nil), document.Items...)
	mutated.Items[0].Steps = append([]plan.Step(nil), document.Items[0].Steps...)
	mutated.Items[0].Steps[0].Verify = "go test ./changed"
	if err := plan.Save(filepath.Join(repository, filepath.FromSlash(plan.Path)), mutated); err != nil {
		t.Fatal(err)
	}
	completionPath, err := gate.completionMessageFile(document)
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(completionPath)
	completion, err := os.ReadFile(completionPath)
	if err != nil {
		t.Fatal(err)
	}
	expected, err := plan.CompletionCommitMessage(
		[]byte("subject\n"), document, "item", "step", manifest, codeManifest, preparation, preparationCommit,
	)
	if err != nil {
		t.Fatal(err)
	}
	if string(completion) != string(expected) {
		t.Fatal("gate completion message differs from the canonical pre-advance message")
	}
	mutatedExpected, err := plan.CompletionCommitMessage(
		[]byte("subject\n"), mutated, "item", "step", manifest, codeManifest, preparation, preparationCommit,
	)
	if err != nil {
		t.Fatal(err)
	}
	if string(completion) == string(mutatedExpected) {
		t.Fatal("mutable on-disk plan replaced the pre-advance completion authority")
	}
}

func TestAcceptanceEvidenceRejectsPlanMutationBeforeCommit(t *testing.T) {
	repository := t.TempDir()
	if err := os.Mkdir(filepath.Join(repository, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	document := completionAuthorityPlan()
	if err := plan.Save(filepath.Join(repository, filepath.FromSlash(plan.Path)), document); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repository, "init", "-q")
	runGitFixture(t, repository, "config", "user.email", "authority@example.invalid")
	runGitFixture(t, repository, "config", "user.name", "Authority Test")
	runGitFixture(t, repository, "add", "--", plan.Path)
	runGitFixture(t, repository, "commit", "-q", "-m", "baseline")
	store, err := overgodb.Open(filepath.Join(repository, gateStorePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	authority, err := plan.ResolveCompletionAuthority(context.Background(), repository, "HEAD", document, store)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := completionAcceptanceEvidenceForPlan(document, "item/step", authority); err != nil {
		t.Fatal(err)
	}
	mutated := document
	mutated.Items = append([]plan.Item(nil), document.Items...)
	mutated.Items[0].Steps = append([]plan.Step(nil), document.Items[0].Steps...)
	mutated.Items[0].Steps[0].Verify = "go test ./changed"
	if _, err := completionAcceptanceEvidenceForPlan(mutated, "item/step", authority); err == nil {
		t.Fatal("plan mutation retained acceptance authority")
	}
}

func TestCompletionAuthorityRejectsAlternateGateStore(t *testing.T) {
	if err := requireCanonicalGateStore("alternate-store", true); err == nil {
		t.Fatal("alternate completion-authority store was accepted")
	}
	if err := requireCanonicalGateStore(gateStorePath, true); err != nil {
		t.Fatalf("canonical store rejected: %v", err)
	}
	if err := requireCanonicalGateStore("alternate-store", false); err != nil {
		t.Fatalf("read-only mode rejected injected store: %v", err)
	}
}

func TestGatePreparationReceiptBindsStoreCommit(t *testing.T) {
	store, err := overgodb.Open(filepath.Join(t.TempDir(), gateStorePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "completion-environment")
	preparation, err := runrecord.NewGatePreparation(strings.Repeat("0", 64), environment, time.Unix(1, 0))
	if err != nil {
		t.Fatal(err)
	}
	content, err := preparation.Content()
	if err != nil {
		t.Fatal(err)
	}
	commit, err := store.Commit(context.Background(), artifact.Batch{
		Key:       "completion-authority/preparation",
		Artifacts: []artifact.Descriptor{{ID: environment}},
		Contents:  []artifact.Content{content},
		Lineage:   preparation.Lineage(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := requireGatePreparationReceipt(context.Background(), store, preparation, commit); err != nil {
		t.Fatal(err)
	}
	wrong := artifact.CommitID{1}
	if wrong == commit {
		wrong = artifact.CommitID{2}
	}
	if err := requireGatePreparationReceipt(context.Background(), store, preparation, wrong); err == nil {
		t.Fatal("preparation receipt accepted the wrong store commit")
	}
}

func completionAuthorityPlan() plan.Plan {
	return plan.Plan{Campaign: "gate", Doctrine: "fixture", Items: []plan.Item{{
		ID: "item", Status: plan.StatusOpen, Steps: []plan.Step{{
			ID: "step", Status: plan.StatusOpen, Verify: "go test ./...",
		}},
	}}}
}
