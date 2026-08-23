package modelrecipe

import (
	"context"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
)

const lifecycleDecisionCommit = "0123456789abcdef0123456789abcdef01234567"

func TestAtomicEvidenceGatedActivation(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "atomic-activation-model")
	testutil.PublishArtifact(t, store, modelID)
	definition, err := inferenceFixture(modelID, recipe.PlacementHost, DecodeSessionRequest)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishCandidate(ctx, store, "fixture/atomic/candidate", definition); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Transition(
		ctx, store, "fixture/atomic/validated", definition, recipe.StatusValidated, nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	verification := publishVerification(t, store, definition.ID, "fixture/atomic/verification")
	_, before := store.Head()
	if _, _, err := ActivateVerified(
		ctx, store, "fixture/atomic/active", definition, verification,
		recipe.EvidenceParity, "exact verifier evidence accepted", nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	_, after := store.Head()
	expectedCommits := uint64(len([]string{"activation"}))
	if after != before+expectedCommits {
		t.Fatalf("activation commits = %d, want %d", after-before, expectedCommits)
	}

	event, err := currentEvent(ctx, store, definition.ID)
	if err != nil {
		t.Fatal(err)
	}
	decisions, err := loadDecisions(ctx, store, event.Evidence)
	if err != nil {
		t.Fatal(err)
	}
	if len(decisions) != int(expectedCommits) || !activationDecisionMatches(decisions[0], definition.ID, verification) {
		t.Fatalf("activation decision = %+v", decisions)
	}
	if !slices.Contains(event.Evidence, decisions[0].ID) ||
		!slices.Contains(event.Evidence, verification.Gate) ||
		!slices.Contains(event.Evidence, verification.Run) {
		t.Fatalf("activation evidence = %v", event.Evidence)
	}
	eventParents, err := store.Parents(ctx, event.ID)
	if err != nil {
		t.Fatal(err)
	}
	for _, edge := range event.Lineage() {
		if !slices.Contains(eventParents, edge) {
			t.Fatalf("activation lineage lacks %+v: %v", edge, eventParents)
		}
	}
	parents, err := store.Parents(ctx, decisions[0].ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(parents) != len(decisions[0].Lineage()) {
		t.Fatalf("decision lineage = %v", parents)
	}
	resolved, ok, err := store.ResolveAlias(ctx, activeAlias(modelID, definition.Task))
	if err != nil || !ok || resolved != definition.ID {
		t.Fatalf("active alias = (%s, %v, %v)", resolved, ok, err)
	}
}

func TestLifecyclePromotionAndSupersession(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "lifecycle-model")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "fixture/lifecycle/facts",
		Artifacts: []artifact.Descriptor{{ID: modelID, Size: uint64(len("lifecycle-model"))}},
	}); err != nil {
		t.Fatal(err)
	}
	first, err := inferenceFixture(modelID, recipe.PlacementHost, DecodeSessionRequest)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishCandidate(ctx, store, "fixture/lifecycle/first/candidate", first); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Transition(ctx, store, "fixture/lifecycle/first/validated", first, recipe.StatusValidated, nil, nil); err != nil {
		t.Fatal(err)
	}
	firstVerification := publishVerification(t, store, first.ID, "fixture/lifecycle/first/verification")
	if _, _, err := Transition(
		ctx, store, "fixture/lifecycle/first/unverified", first, recipe.StatusActive,
		[]artifact.ID{firstVerification.Gate, firstVerification.Run}, nil,
	); err == nil {
		t.Fatal("ordinary lifecycle transition bypassed verified activation")
	}
	if _, _, err := ActivateVerified(
		ctx, store, "fixture/lifecycle/first/active", first, firstVerification,
		recipe.EvidenceExperimental, "first fixture activation", nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	activation, ok, err := ActiveRecord(ctx, store, modelID, recipe.TaskInference)
	active := activation.Definition
	if err != nil || !ok || active.ID != first.ID {
		t.Fatalf("first active = (%s, %v, %v)", active.ID, ok, err)
	}

	second, err := inferenceFixture(modelID, recipe.PlacementDevice, DecodeSessionRequest)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishCandidate(ctx, store, "fixture/lifecycle/second/candidate", second); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Transition(ctx, store, "fixture/lifecycle/second/validated", second, recipe.StatusValidated, nil, nil); err != nil {
		t.Fatal(err)
	}
	if _, _, err := ActivateVerified(
		ctx, store, "fixture/lifecycle/second/mismatched", second, firstVerification,
		recipe.EvidenceExperimental, "mismatched fixture activation", nil, &first.ID,
	); err == nil {
		t.Fatal("activation accepted verifier output for another recipe")
	}
	secondVerification := publishVerification(t, store, second.ID, "fixture/lifecycle/second/verification")
	if _, _, err := ActivateVerified(
		ctx, store, "fixture/lifecycle/second/active", second, secondVerification,
		recipe.EvidenceExperimental, "second fixture activation", nil, &first.ID,
	); err != nil {
		t.Fatal(err)
	}
	activation, ok, err = ActiveRecord(ctx, store, modelID, recipe.TaskInference)
	active = activation.Definition
	if err != nil || !ok || active.ID != second.ID {
		t.Fatalf("second active = (%s, %v, %v)", active.ID, ok, err)
	}
	oldStatus, err := currentEvent(ctx, store, first.ID)
	if err != nil || oldStatus.To != recipe.StatusSuperseded {
		t.Fatalf("old status = (%+v, %v)", oldStatus, err)
	}
	compiled, ok, err := compileActiveFixture(
		ctx, store, modelID,
		model.Spec{CommonSpec: model.CommonSpec{Architecture: "llama"}}, model.Weights{},
	)
	if err != nil || !ok || compiled.Recipe.ID != second.ID {
		t.Fatalf("compiled active = (%s, %v, %v)", compiled.Recipe.ID, ok, err)
	}
}

func TestActivateCapabilityRequiresBoundVerification(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "activation-model")
	testutil.PublishArtifact(t, store, modelID)
	definition, err := inferenceFixture(modelID, recipe.PlacementHost, DecodeSessionRequest)
	if err != nil {
		t.Fatal(err)
	}
	if err := ActivateCapability(
		ctx, store, definition, Verification{}, recipe.EvidenceExperimental, "verified fixture",
	); err == nil {
		t.Fatal("activation accepted missing verification")
	}
	if _, active, err := ActiveRecord(ctx, store, modelID, definition.Task); err != nil || active {
		t.Fatalf("unverified active = (%v, %v)", active, err)
	}
	verification := publishVerification(t, store, definition.ID, "fixture/activation/verification")
	if err := ActivateCapability(
		ctx, store, definition, verification, recipe.EvidenceExperimental, "verified fixture",
	); err != nil {
		t.Fatal(err)
	}
	activation, active, err := ActiveRecord(ctx, store, modelID, definition.Task)
	if err != nil || !active || activation.Definition.ID != definition.ID {
		t.Fatalf("active = (%s, %v, %v)", activation.Definition.ID, active, err)
	}
	refreshed := publishVerification(t, store, definition.ID, "fixture/activation/refreshed")
	if err := ActivateCapability(
		ctx, store, definition, refreshed, recipe.EvidenceParity, "refreshed verifier",
	); err != nil {
		t.Fatal(err)
	}
	activation, active, err = ActiveRecord(ctx, store, modelID, definition.Task)
	if err != nil || !active || activation.Tier != recipe.EvidenceParity ||
		!slices.Contains(activation.Event.Evidence, refreshed.Gate) ||
		!slices.Contains(activation.Event.Evidence, refreshed.Run) {
		t.Fatalf("refreshed active = (%+v, %v, %v)", activation, active, err)
	}
	secondRefresh := publishVerification(t, store, definition.ID, "fixture/activation/refreshed-again")
	if err := ActivateCapability(
		ctx, store, definition, secondRefresh, recipe.EvidenceProduction, "refreshed verifier again",
	); err != nil {
		t.Fatal(err)
	}
	activation, active, err = ActiveRecord(ctx, store, modelID, definition.Task)
	if err != nil || !active || activation.Tier != recipe.EvidenceProduction ||
		!slices.Contains(activation.Event.Evidence, secondRefresh.Gate) {
		t.Fatalf("second refreshed active = (%+v, %v, %v)", activation, active, err)
	}
}

func TestRetireActiveCapabilityRequiresFailedEvidence(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "retirement-model")
	testutil.PublishArtifact(t, store, modelID)
	active, err := inferenceFixture(modelID, recipe.PlacementHost, DecodeSessionRequest)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishCandidate(ctx, store, "fixture/retirement/active-candidate", active); err != nil {
		t.Fatal(err)
	}
	activeVerification := publishVerification(t, store, active.ID, "fixture/retirement/active-verification")
	if err := ActivateCapability(
		ctx, store, active, activeVerification, recipe.EvidenceExperimental, "legacy activation",
	); err != nil {
		t.Fatal(err)
	}
	candidate, err := inferenceFixture(modelID, recipe.PlacementDevice, DecodeSessionRequest)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishCandidate(ctx, store, "fixture/retirement/candidate", candidate); err != nil {
		t.Fatal(err)
	}
	if err := RetireActiveCapability(
		ctx, store, candidate, activeVerification, "exact output mismatch",
	); err == nil {
		t.Fatal("successful verifier evidence retired an active recipe")
	}
	failed := publishFailedVerification(t, store, candidate.ID, "fixture/retirement/failed-verification")
	if err := RetireActiveCapability(
		ctx, store, candidate, failed, "exact output mismatch",
	); err != nil {
		t.Fatal(err)
	}
	if declared, err := HasActiveRecipe(ctx, store, modelID, recipe.TaskInference); err != nil || declared {
		t.Fatalf("retired active declaration = (%v, %v)", declared, err)
	}
	status, published, err := Status(ctx, store, active.ID)
	if err != nil || !published || status != recipe.StatusSuperseded {
		t.Fatalf("retired status = (%s, %v, %v)", status, published, err)
	}
}

func TestActiveRecordSurfacesTierAndRejectsRefusedAlias(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "decision-model")
	derivationID := testutil.ArtifactID(t, artifact.KindEvidence, "decision-derivation")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/decision/facts",
		Artifacts: []artifact.Descriptor{
			{ID: modelID, Size: uint64(len("decision-model"))},
			{ID: derivationID, Size: uint64(len("decision-derivation"))},
		},
	}); err != nil {
		t.Fatal(err)
	}
	active, err := inferenceFixture(modelID, recipe.PlacementHost, DecodeSessionRequest)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishCandidate(ctx, store, "fixture/decision/active/candidate", active); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Transition(ctx, store, "fixture/decision/active/validated", active, recipe.StatusValidated, nil, nil); err != nil {
		t.Fatal(err)
	}
	verification := publishVerification(t, store, active.ID, "fixture/decision/active/verification")
	if _, _, err := ActivateVerified(
		ctx, store, "fixture/decision/active", active, verification,
		recipe.EvidenceParity, "accepted fixture activation", nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	record, ok, err := ActiveRecord(ctx, store, modelID, recipe.TaskInference)
	if err != nil || !ok || record.Tier != recipe.EvidenceParity || len(record.Decisions) != 1 {
		t.Fatalf("active record = (%+v, %v, %v)", record, ok, err)
	}

	refused, err := inferenceFixture(modelID, recipe.PlacementDevice, DecodeSessionRequest)
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
		recipe.Decider{CodeCommit: lifecycleDecisionCommit, Derivation: derivationID},
		[]artifact.ID{testutil.ArtifactID(t, artifact.KindRun, "kernel-parity-failed-run")},
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

func TestActiveRecordRejectsLegacyIntentEvidence(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "legacy-intent-model")
	intentID := testutil.ArtifactID(t, artifact.KindEvidence, "activation-intent")
	testutil.PublishArtifact(t, store, modelID)
	testutil.PublishArtifact(t, store, intentID)
	definition, err := inferenceFixture(modelID, recipe.PlacementHost, DecodeSessionRequest)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishCandidate(ctx, store, "fixture/legacy/candidate", definition); err != nil {
		t.Fatal(err)
	}
	_, validated, err := Transition(
		ctx, store, "fixture/legacy/validated", definition, recipe.StatusValidated, nil, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	active, err := recipe.NewLifecycleEvent(
		definition, recipe.StatusValidated, recipe.StatusActive, &validated.ID, nil, []artifact.ID{intentID},
	)
	if err != nil {
		t.Fatal(err)
	}
	content, err := active.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch(
		"fixture/legacy/active", []artifact.Content{content}, nil,
		[]artifact.AliasBinding{
			{Name: statusAlias(definition.ID), Target: active.ID, Previous: &validated.ID},
			{Name: activeAlias(modelID, definition.Task), Target: definition.ID},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	if _, ok, err := ActiveRecord(ctx, store, modelID, definition.Task); err == nil || ok ||
		!strings.Contains(err.Error(), "lacks verified evidence") {
		t.Fatalf("legacy intent activation = (%v, %v)", ok, err)
	}
	replacement, err := inferenceFixture(modelID, recipe.PlacementDevice, DecodeSessionRequest)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishCandidate(ctx, store, "fixture/legacy/replacement/candidate", replacement); err != nil {
		t.Fatal(err)
	}
	verification := publishVerification(t, store, replacement.ID, "fixture/legacy/replacement/verification")
	if err := ActivateCapability(
		ctx, store, replacement, verification, recipe.EvidenceParity, "replace legacy evidence",
	); err != nil {
		t.Fatal(err)
	}
	activation, ok, err := ActiveRecord(ctx, store, modelID, definition.Task)
	if err != nil || !ok || activation.Definition.ID != replacement.ID {
		t.Fatalf("replacement activation = (%s, %v, %v)", activation.Definition.ID, ok, err)
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
