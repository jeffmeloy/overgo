package modelrecipe

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
)

const lifecycleDecisionCommit = "0123456789abcdef0123456789abcdef01234567"

func TestLifecyclePromotionAndSupersession(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID, _ := artifact.IdentifyBytes(artifact.KindModel, []byte("lifecycle-model"))
	evidenceID, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("validation"))
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/lifecycle/facts",
		Artifacts: []artifact.Descriptor{
			{ID: modelID, Size: uint64(len("lifecycle-model"))},
			{ID: evidenceID, Size: uint64(len("validation"))},
		},
	}); err != nil {
		t.Fatal(err)
	}
	first, err := Inference(modelID, recipe.PlacementHost)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishCandidate(ctx, store, "fixture/lifecycle/first/candidate", first); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Transition(ctx, store, "fixture/lifecycle/first/validated", first, recipe.StatusValidated, nil, nil); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Transition(ctx, store, "fixture/lifecycle/first/active", first, recipe.StatusActive, []artifact.ID{evidenceID}, nil); err != nil {
		t.Fatal(err)
	}
	active, ok, err := Active(ctx, store, modelID, recipe.TaskInference)
	if err != nil || !ok || active.ID != first.ID {
		t.Fatalf("first active = (%s, %v, %v)", active.ID, ok, err)
	}

	second, err := Inference(modelID, recipe.PlacementDevice)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishCandidate(ctx, store, "fixture/lifecycle/second/candidate", second); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Transition(ctx, store, "fixture/lifecycle/second/validated", second, recipe.StatusValidated, nil, nil); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Transition(
		ctx, store, "fixture/lifecycle/second/active", second, recipe.StatusActive,
		[]artifact.ID{evidenceID}, &first.ID,
	); err != nil {
		t.Fatal(err)
	}
	active, ok, err = Active(ctx, store, modelID, recipe.TaskInference)
	if err != nil || !ok || active.ID != second.ID {
		t.Fatalf("second active = (%s, %v, %v)", active.ID, ok, err)
	}
	oldStatus, err := currentEvent(ctx, store, first.ID)
	if err != nil || oldStatus.To != recipe.StatusSuperseded {
		t.Fatalf("old status = (%+v, %v)", oldStatus, err)
	}
	compiled, ok, err := CompileActive(
		ctx, store, modelID,
		model.Spec{CommonSpec: model.CommonSpec{Architecture: "llama"}}, model.Weights{},
	)
	if err != nil || !ok || compiled.Recipe.ID != second.ID {
		t.Fatalf("compiled active = (%s, %v, %v)", compiled.Recipe.ID, ok, err)
	}
}

func TestActiveRecordSurfacesTierAndRejectsRefusedAlias(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID, _ := artifact.IdentifyBytes(artifact.KindModel, []byte("decision-model"))
	derivationID, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("decision-derivation"))
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/decision/facts",
		Artifacts: []artifact.Descriptor{
			{ID: modelID, Size: uint64(len("decision-model"))},
			{ID: derivationID, Size: uint64(len("decision-derivation"))},
		},
	}); err != nil {
		t.Fatal(err)
	}
	active, err := Inference(modelID, recipe.PlacementHost)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishCandidate(ctx, store, "fixture/decision/active/candidate", active); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Transition(ctx, store, "fixture/decision/active/validated", active, recipe.StatusValidated, nil, nil); err != nil {
		t.Fatal(err)
	}
	accepted, err := recipe.NewDecision(
		active.ID, recipe.DecisionAccepted, recipe.EvidenceParity, "",
		recipe.Decider{CodeCommit: lifecycleDecisionCommit, Derivation: derivationID}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	publishDecision(t, store, "fixture/decision/accepted", accepted)
	if _, _, err := Transition(
		ctx, store, "fixture/decision/active", active, recipe.StatusActive,
		[]artifact.ID{accepted.ID}, nil,
	); err != nil {
		t.Fatal(err)
	}
	record, ok, err := ActiveRecord(ctx, store, modelID, recipe.TaskInference)
	if err != nil || !ok || record.Tier != recipe.EvidenceParity || len(record.Decisions) != 1 {
		t.Fatalf("active record = (%+v, %v, %v)", record, ok, err)
	}

	refused, err := Inference(modelID, recipe.PlacementDevice)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishCandidate(ctx, store, "fixture/decision/refused/candidate", refused); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Transition(ctx, store, "fixture/decision/refused/missing", refused, recipe.StatusRefused, nil, nil); err == nil {
		t.Fatal("refusal without typed decision accepted")
	}
	refusal, err := recipe.NewDecision(
		refused.ID, recipe.DecisionRefused, recipe.EvidenceExperimental, "kernel parity failed",
		recipe.Decider{CodeCommit: lifecycleDecisionCommit, Derivation: derivationID}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	publishDecision(t, store, "fixture/decision/refusal", refusal)
	if _, _, err := Transition(
		ctx, store, "fixture/decision/refused", refused, recipe.StatusRefused,
		[]artifact.ID{refusal.ID}, nil,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/decision/stale-active-alias",
		Aliases: []artifact.AliasBinding{{
			Name: activeAlias(modelID, recipe.TaskInference), Target: refused.ID, Previous: &active.ID,
		}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, ok, err := ActiveRecord(ctx, store, modelID, recipe.TaskInference); err == nil || ok ||
		!strings.Contains(err.Error(), refusal.Reason) {
		t.Fatalf("refused active alias = (%v, %v)", ok, err)
	}
}

func publishDecision(t *testing.T, store artifact.Repository, key string, decision recipe.Decision) {
	t.Helper()
	content, err := decision.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key: key, Contents: []artifact.Content{content},
		Lineage: []artifact.Lineage{{
			Child: decision.ID, Parent: decision.Subject, Relation: artifact.RelationDependsOn,
		}},
	}); err != nil {
		t.Fatal(err)
	}
}
