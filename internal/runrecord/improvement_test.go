package runrecord

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

func TestRecursiveImprovementLineageAndExternalPromotion(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	parent := id(artifact.KindModel, "parent")
	dataset := id(artifact.KindDataset, "dataset")
	development := id(artifact.KindDatasetShard, "development")
	promotion := id(artifact.KindDatasetShard, "promotion")
	recipeID := id(artifact.KindRecipe, "recipe")
	code := id(artifact.KindEvidence, "code")
	proposer := id(artifact.KindEvidence, "proposer")
	proposal, err := trainingprogram.CompileImprovementProposal(trainingprogram.ImprovementSpec{
		Kind: trainingprogram.ImprovementComponentComposition, ParentModel: parent,
		Incumbent: id(artifact.KindModelDefinition, "current-composition"),
		Candidate: id(artifact.KindModelDefinition, "composition"), Dataset: dataset,
		DevelopmentSplit: development, Recipe: recipeID, Code: code, Proposer: proposer,
		Components: []artifact.ID{id(artifact.KindModel, "component"), id(artifact.KindAdapter, "bridge")},
	})
	if err != nil {
		t.Fatal(err)
	}
	evaluator := id(artifact.KindEvidence, "evaluator")
	authority := id(artifact.KindEvidence, "admission-authority")
	admission, err := AdmitImprovement(proposal, authority, evaluator, promotion)
	if err != nil {
		t.Fatal(err)
	}
	child := id(artifact.KindModel, "child")
	environment := id(artifact.KindEvidence, "environment")
	output := id(artifact.KindOutput, "prediction")
	run, err := NewBoundRun(
		recipeID, OutcomeSucceeded,
		[]artifact.ID{child, dataset, promotion, code, admission.ID}, []artifact.ID{output},
		"", fixtureCodeCommit, environment, 100, []PhaseMetric{{Phase: PhaseValidate, DurationNS: 100}},
	)
	if err != nil {
		t.Fatal(err)
	}
	evaluation, err := NewEvaluation(recipeID, run.ID, dataset, []Metric{{
		Name: "quality", Value: 1, Direction: DirectionMaximize,
	}})
	if err != nil {
		t.Fatal(err)
	}
	decider := id(artifact.KindEvidence, "decider")
	decision, err := DecideImprovement(admission, child, run, evaluation, decider, ImprovementPromote)
	if err != nil || decision.Rollback != parent {
		t.Fatalf("decision = (%s, %s, %v)", decision.State, decision.Rollback, err)
	}
	parents := map[artifact.ID]bool{}
	for _, edge := range decision.Lineage() {
		if edge.Child == child {
			parents[edge.Parent] = true
		}
	}
	for _, required := range []artifact.ID{
		parent, proposal.Incumbent(), proposal.Candidate(), dataset, development, promotion, recipeID, code, evaluator,
	} {
		if !parents[required] {
			t.Fatalf("child lineage omits %s", required)
		}
	}
	content, err := decision.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := improvementDecisionCodec.Parse(content.Data)
	if err != nil || parsed.ID != decision.ID || parsed.Rollback != parent {
		t.Fatalf("parsed decision = (%s, %s, %v)", parsed.ID, parsed.Rollback, err)
	}
	if _, err := decision.Batch("fixture/improvement/decision"); err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	static := []artifact.ID{
		proposal.ID(), proposal.Incumbent(), proposal.Candidate(), parent, child, dataset, development, promotion,
		recipeID, code, proposer, evaluator, authority, decider, environment, output,
	}
	descriptors := make([]artifact.Descriptor, len(static))
	for index, value := range static {
		descriptors[index] = artifact.Descriptor{ID: value}
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "fixture/improvement/static", Artifacts: descriptors}); err != nil {
		t.Fatal(err)
	}
	for _, batch := range []struct {
		key   string
		build func(string) (artifact.Batch, error)
	}{
		{"fixture/improvement/admission", admission.Batch},
		{"fixture/improvement/run", run.Batch},
		{"fixture/improvement/evaluation", evaluation.Batch},
		{"fixture/improvement/decision", decision.Batch},
	} {
		value, err := batch.build(batch.key)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, value); err != nil {
			t.Fatal(err)
		}
	}
	storedParents, err := store.Parents(ctx, child)
	if err != nil || len(storedParents) != len(parents) {
		t.Fatalf("stored child parents = (%d, %v), want %d", len(storedParents), err, len(parents))
	}
	decisionParents, err := store.Parents(ctx, decision.ID)
	if err != nil || len(decisionParents) != 16 {
		t.Fatalf("stored decision parents = (%d, %v), want 16", len(decisionParents), err)
	}
	if _, err := AdmitImprovement(proposal, proposer, evaluator, promotion); err == nil {
		t.Fatal("self-admission accepted")
	}
	if _, err := AdmitImprovement(proposal, authority, evaluator, development); err == nil {
		t.Fatal("development split accepted as promotion split")
	}
	if _, err := DecideImprovement(admission, child, run, evaluation, evaluator, ImprovementPromote); err == nil {
		t.Fatal("evaluator accepted as promotion decider")
	}
}
