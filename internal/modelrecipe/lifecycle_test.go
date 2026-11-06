package modelrecipe

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

const lifecycleDecisionCommit = "0123456789abcdef0123456789abcdef01234567"

func TestAtomicEvidenceGatedActivation(t *testing.T) {
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	modelID := testutil.ArtifactID(t, artifact.KindModel, "atomic-activation-model")
	testutil.PublishArtifact(t, store, modelID)
	definition := testcheck.Must(inferenceFixture(modelID, recipe.PlacementHost, DecodeSessionRequest))(t)
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

	event := testcheck.Must(currentEvent(ctx, store, definition.ID))(t)
	decisions := testcheck.Must(loadDecisions(ctx, store, event.Evidence))(t)
	if len(decisions) != int(expectedCommits) || !activationDecisionMatches(decisions[0], definition.ID, verification) {
		t.Fatalf("activation decision = %+v", decisions)
	}
	if !slices.Contains(event.Evidence, decisions[0].ID) ||
		!slices.Contains(event.Evidence, verification.Gate) ||
		!slices.Contains(event.Evidence, verification.Run) {
		t.Fatalf("activation evidence = %v", event.Evidence)
	}
	eventParents := testcheck.Must(store.Parents(ctx, event.ID))(t)
	for _, edge := range event.Lineage() {
		if !slices.Contains(eventParents, edge) {
			t.Fatalf("activation lineage lacks %+v: %v", edge, eventParents)
		}
	}
	parents := testcheck.Must(store.Parents(ctx, decisions[0].ID))(t)
	if len(parents) != len(decisions[0].Lineage()) {
		t.Fatalf("decision lineage = %v", parents)
	}
	resolved, ok, err := store.ResolveAlias(ctx, activeAlias(modelID, definition.Task))
	if err != nil || !ok || resolved != definition.ID {
		t.Fatalf("active alias = (%s, %v, %v)", resolved, ok, err)
	}
}

func TestActivateCapabilityRequiresBoundVerification(t *testing.T) {
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	modelID := testutil.ArtifactID(t, artifact.KindModel, "activation-model")
	testutil.PublishArtifact(t, store, modelID)
	definition := testcheck.Must(inferenceFixture(modelID, recipe.PlacementHost, DecodeSessionRequest))(t)
	if err := ActivateCapability(
		ctx, store, definition, Verification{}, recipe.EvidenceVerified, "verified fixture",
	); err == nil {
		t.Fatal("activation accepted missing verification")
	}
	if _, active, err := ActiveRecord(ctx, store, modelID, definition.Task); err != nil || active {
		t.Fatalf("unverified active = (%v, %v)", active, err)
	}
	verification := publishVerification(t, store, definition.ID, "fixture/activation/verification")
	if err := ActivateCapability(
		ctx, store, definition, verification, recipe.EvidenceVerified, "verified fixture",
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

func TestActiveRecordSurfacesTierAndRejectsRefusedAlias(t *testing.T) {
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
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
	active := testcheck.Must(inferenceFixture(modelID, recipe.PlacementHost, DecodeSessionRequest))(t)
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

	refused := testcheck.Must(inferenceFixture(modelID, recipe.PlacementDevice, DecodeSessionRequest))(t)
	if _, _, err := PublishCandidate(ctx, store, "fixture/decision/refused/candidate", refused); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Transition(ctx, store, "fixture/decision/refused/missing", refused, recipe.StatusRefused, nil, nil); err == nil {
		t.Fatal("refusal without typed decision accepted")
	}
	refusal := testcheck.Must(recipe.NewDecision(
		refused.ID, recipe.DecisionRefused, recipe.EvidenceVerified, "kernel parity failed",
		recipe.Decider{CodeCommit: lifecycleDecisionCommit, Derivation: derivationID},
		[]artifact.ID{testutil.ArtifactID(t, artifact.KindRun, "kernel-parity-failed-run")},
	))(t)
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
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	modelID := testutil.ArtifactID(t, artifact.KindModel, "legacy-intent-model")
	intentID := testutil.ArtifactID(t, artifact.KindEvidence, "activation-intent")
	testutil.PublishArtifact(t, store, modelID)
	testutil.PublishArtifact(t, store, intentID)
	definition := testcheck.Must(inferenceFixture(modelID, recipe.PlacementHost, DecodeSessionRequest))(t)
	if _, _, err := PublishCandidate(ctx, store, "fixture/legacy/candidate", definition); err != nil {
		t.Fatal(err)
	}
	_, validated, err := Transition(
		ctx, store, "fixture/legacy/validated", definition, recipe.StatusValidated, nil, nil,
	)
	testcheck.NoError(t, err)
	active := testcheck.Must(recipe.NewLifecycleEvent(
		definition, recipe.StatusValidated, recipe.StatusActive, &validated.ID, nil, []artifact.ID{intentID},
	))(t)
	content := testcheck.Must(active.Content())(t)
	batch := testcheck.Must(artifact.NewDocumentBatch(
		"fixture/legacy/active", []artifact.Content{content}, nil,
		[]artifact.AliasBinding{
			{Name: statusAlias(definition.ID), Target: active.ID, Previous: &validated.ID},
			{Name: activeAlias(modelID, definition.Task), Target: definition.ID},
		},
	))(t)
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	if _, ok, err := ActiveRecord(ctx, store, modelID, definition.Task); err == nil || ok ||
		!strings.Contains(err.Error(), "lacks verified evidence") {
		t.Fatalf("legacy intent activation = (%v, %v)", ok, err)
	}
	replacement := testcheck.Must(inferenceFixture(modelID, recipe.PlacementDevice, DecodeSessionRequest))(t)
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
	content := testcheck.Must(decision.Content())(t)
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: key, Contents: []artifact.Content{content},
		Lineage: []artifact.Lineage{{
			Child: decision.ID, Parent: decision.Subject, Relation: artifact.RelationDependsOn,
		}},
	}); err != nil {
		t.Fatal(err)
	}
}
