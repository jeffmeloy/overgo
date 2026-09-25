package trainingworkflow

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/trainingprogram"
)

// TestPromoteObjectiveAdaptive pins the evidence-gated ladder climb: a
// declared objective promotes to adaptive-evidence only against a
// committed SUCCEEDED training session observation whose recipe
// grounds this exact objective; a fabricated observation is refused,
// and a second promotion is refused because the objective no longer
// sits at declared. Approval refuses a training observation, a
// fabricated identity, a failed report, and a passed benchmark report
// alone: it needs a held-out verdict.
func TestPromoteObjectiveAdaptive(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	derived := derivedIdentity(t, "overgo/test-objective/")
	objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
		Name: "test-pair", Kind: trainingprogram.ObjectiveFlowMatching,
		Signature: recipecontract.ModalitySignature{
			Inputs:  []recipecontract.Modality{recipecontract.ModalityText},
			Outputs: []recipecontract.Modality{recipecontract.ModalityVideo},
		},
		Dataset: derived(artifact.KindDataset, "dataset"), Split: derived(artifact.KindDatasetShard, "split"),
		Processors: []artifact.ID{derived(artifact.KindProfile, "processor")},
		Loss:       derived(artifact.KindProfile, "loss"), Evaluation: derived(artifact.KindProfile, "evaluation"),
		Metric: trainingprogram.MetricVideoPSNRUnit, Evidence: []artifact.ID{derived(artifact.KindEvidence, "note")},
		Authority: trainingprogram.ObjectiveDeclared,
	})
	if err != nil {
		t.Fatal(err)
	}
	const alias = "objective.registered.test-pair"
	observation := observedTraining(t, ctx, store, root, objective, alias)

	fabricated := derived(artifact.KindEvidence, "fabricated-observation")
	if _, err := PromoteObjectiveAdaptive(ctx, store, alias, []artifact.ID{fabricated}); err == nil {
		t.Fatal("a fabricated observation was accepted as promotion evidence")
	}
	promoted, err := PromoteObjectiveAdaptive(ctx, store, alias, []artifact.ID{observation})
	if err != nil {
		t.Fatal(err)
	}
	if promoted.Authority != trainingprogram.ObjectiveAdaptive {
		t.Fatalf("promoted authority = %q", promoted.Authority)
	}
	bound, found, err := store.ResolveAlias(ctx, alias)
	if err != nil || !found || bound != promoted.ID {
		t.Fatalf("alias after promotion = (%s, %t, %v), want the promoted document", bound, found, err)
	}
	reloaded, err := trainingprogram.LoadObjective(ctx, store, promoted.ID)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(reloaded.Evidence, observation) {
		t.Fatal("promoted objective does not carry the grounding observation in its evidence")
	}
	if _, err := PromoteObjectiveAdaptive(ctx, store, alias, []artifact.ID{observation}); err == nil {
		t.Fatal("an adaptive objective was promoted a second time from declared")
	}

	if _, err := PromoteObjectiveApproved(ctx, store, alias, []artifact.ID{observation}); err == nil {
		t.Fatal("a training observation was accepted as approval evidence")
	}
	if _, err := PromoteObjectiveApproved(ctx, store, alias, []artifact.ID{fabricated}); err == nil {
		t.Fatal("a fabricated report was accepted as approval evidence")
	}
	if _, err := PromoteObjectiveApproved(ctx, store, alias, []artifact.ID{commitBenchmarkReport(t, ctx, store, "failed", false)}); err == nil {
		t.Fatal("a FAILED evaluation report was accepted as approval evidence")
	}
	if _, err := PromoteObjectiveApproved(ctx, store, alias, []artifact.ID{commitBenchmarkReport(t, ctx, store, "passed", true)}); err == nil {
		t.Fatal("a benchmark report alone approved an objective with no held-out verdict")
	}
}

// TestObjectiveApprovalAcceptsHeldOutTargetReports climbs a forecast
// objective bound to the training half of a real group split to approved
// on a held-out verdict: the trained model's forecasts beat the base
// model's on records the training split never held. A verdict that finds
// no gain, and one judged for another objective's contract, are refused,
// and a passed benchmark report may accompany the verdict.
func TestObjectiveApprovalAcceptsHeldOutTargetReports(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	derived := derivedIdentity(t, "overgo/heldout-approval-test/")
	source := derived(artifact.KindDataset, "dataset")
	var records []dataset.Record
	for index := range 12 {
		records = append(records, dataset.Record{ID: fmt.Sprintf("series/%02d", index), Group: fmt.Sprintf("series-%02d", index)})
	}
	split, err := dataset.BuildGroupSplit(source, records, 17, []dataset.SplitPartition{{Name: "heldout", Weight: 1}, {Name: "train", Weight: 1}})
	if err != nil {
		t.Fatal(err)
	}
	memberships := map[string]dataset.Membership{}
	for _, membership := range split.Memberships {
		memberships[membership.Partition] = membership
	}
	training, heldout := memberships["train"], memberships["heldout"]
	if len(training.Records) == 0 || len(heldout.Records) == 0 {
		t.Fatalf("split left a partition empty: train=%d heldout=%d", len(training.Records), len(heldout.Records))
	}
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{Key: "test-dataset", Artifacts: []artifact.Descriptor{{ID: source}}}); err != nil {
		t.Fatal(err)
	}
	splitBatch, err := split.PublicationBatch("test-split", nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, splitBatch); err != nil {
		t.Fatal(err)
	}
	spec := trainingprogram.ObjectiveSpec{
		Name: "forecast-heldout", Kind: trainingprogram.ObjectiveForecast,
		Signature: recipecontract.ModalitySignature{
			Inputs:  []recipecontract.Modality{recipecontract.ModalityTimeSeries},
			Outputs: []recipecontract.Modality{recipecontract.ModalityTimeSeries},
		},
		Dataset: source, Split: training.ID, Processors: []artifact.ID{derived(artifact.KindProfile, "processor")},
		Loss: derived(artifact.KindProfile, "loss"), Evaluation: derived(artifact.KindProfile, "evaluation"),
		Metric: trainingprogram.MetricForecastMAE, Evidence: []artifact.ID{derived(artifact.KindEvidence, "note")},
		Authority: trainingprogram.ObjectiveDeclared,
	}
	declared, err := trainingprogram.NewObjective(spec)
	if err != nil {
		t.Fatal(err)
	}
	const alias = "objective.registered.forecast-heldout"
	observation := observedTraining(t, ctx, store, root, declared, alias)
	if _, err := PromoteObjectiveAdaptive(ctx, store, alias, []artifact.ID{observation}); err != nil {
		t.Fatal(err)
	}

	view, err := evaluation.CompileSFTEvaluationView(declared, training, heldout)
	if err != nil {
		t.Fatal(err)
	}
	suite := evaluation.NumericTargetSuite{Scorers: []evaluation.NumericScorerSpec{{Kind: evaluation.NumericAbsoluteError}}}
	for index, record := range view.Records {
		suite.Cases = append(suite.Cases, evaluation.NumericTargetCase{Record: record.ID, Values: []float64{float64(index), 1}})
	}
	plan, err := evaluation.CompileNumericTargetPlan(view, suite)
	if err != nil {
		t.Fatal(err)
	}
	forecast := func(offset float64) evaluation.NumericTargetReport {
		var observations []evaluation.NumericTargetObservation
		for _, target := range plan.Cases {
			values := slices.Clone(target.Values)
			for index := range values {
				values[index] += offset
			}
			observations = append(observations, evaluation.NumericTargetObservation{Record: target.Record, Values: values})
		}
		report, err := evaluation.ScoreNumericTargets(plan, observations)
		if err != nil {
			t.Fatal(err)
		}
		return report
	}
	base, trained, regressed := forecast(2), forecast(1), forecast(3)
	commit := func(key string, stage func(string) (artifact.Batch, error)) {
		t.Helper()
		batch, err := stage(key)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
			t.Fatal(err)
		}
	}
	commit("test-view", view.Batch)
	commit("test-plan", plan.Batch)
	for index, report := range []evaluation.NumericTargetReport{base, trained, regressed} {
		commit(fmt.Sprintf("test-report-%d", index), report.Batch)
	}
	judge := func(key string, objective trainingprogram.ObjectiveDocument, view evaluation.SFTEvaluationView, candidate evaluation.NumericTargetReport) artifact.ID {
		t.Helper()
		verdict, err := evaluation.JudgeHeldout(objective, view, plan, base, candidate, observation)
		if err != nil {
			t.Fatal(err)
		}
		commit(key, verdict.Batch)
		return verdict.ID
	}

	if _, err := PromoteObjectiveApproved(ctx, store, alias, []artifact.ID{judge("test-regressed", declared, view, regressed)}); err == nil {
		t.Fatal("a held-out verdict with no gain approved the objective")
	}
	renamedSpec := spec
	renamedSpec.Name = "forecast-heldout-other"
	renamed, err := trainingprogram.NewObjective(renamedSpec)
	if err != nil {
		t.Fatal(err)
	}
	renamedContent, err := renamed.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{Key: "test-renamed", Contents: []artifact.Content{renamedContent}}); err != nil {
		t.Fatal(err)
	}
	otherView, err := evaluation.CompileSFTEvaluationView(renamed, training, heldout)
	if err != nil {
		t.Fatal(err)
	}
	commit("test-other-view", otherView.Batch)
	otherPlan, err := evaluation.CompileNumericTargetPlan(otherView, suite)
	if err != nil {
		t.Fatal(err)
	}
	commit("test-other-plan", otherPlan.Batch)
	scoreOther := func(key string, offset float64) evaluation.NumericTargetReport {
		t.Helper()
		var observations []evaluation.NumericTargetObservation
		for _, target := range otherPlan.Cases {
			values := slices.Clone(target.Values)
			for index := range values {
				values[index] += offset
			}
			observations = append(observations, evaluation.NumericTargetObservation{Record: target.Record, Values: values})
		}
		report, err := evaluation.ScoreNumericTargets(otherPlan, observations)
		if err != nil {
			t.Fatal(err)
		}
		commit(key, report.Batch)
		return report
	}
	otherVerdict, err := evaluation.JudgeHeldout(renamed, otherView, otherPlan, scoreOther("test-other-base", 2), scoreOther("test-other-trained", 1), observation)
	if err != nil || !otherVerdict.Passed {
		t.Fatalf("the other objective's verdict = (passed=%t, %v)", otherVerdict.Passed, err)
	}
	commit("test-other-verdict", otherVerdict.Batch)
	if _, err := PromoteObjectiveApproved(ctx, store, alias, []artifact.ID{otherVerdict.ID}); err == nil {
		t.Fatal("a passed verdict judged for another objective's contract approved this one")
	}

	passed := judge("test-passed", declared, view, trained)
	approved, err := PromoteObjectiveApproved(ctx, store, alias, []artifact.ID{passed, commitBenchmarkReport(t, ctx, store, "passed", true)})
	if err != nil {
		t.Fatal(err)
	}
	if approved.Authority != trainingprogram.ObjectiveApproved || !slices.Contains(approved.Evidence, passed) {
		t.Fatalf("approved = %q, evidence carries the verdict %t", approved.Authority, slices.Contains(approved.Evidence, passed))
	}
}

// derivedIdentity names stand-in identities for the objective's declared
// components under prefix.
func derivedIdentity(t *testing.T, prefix string) func(artifact.Kind, string) artifact.ID {
	return func(kind artifact.Kind, role string) artifact.ID {
		t.Helper()
		id, err := artifact.IdentifyBytes(kind, []byte(prefix+role))
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
}

// observedTraining registers objective under alias, with a stand-in
// descriptor for each identity it names that the store does not hold, and
// records one succeeded training session grounded in it; it returns the
// session's observation.
func observedTraining(t *testing.T, ctx context.Context, store *overgodb.Store, root string, objective trainingprogram.ObjectiveDocument, alias string) artifact.ID {
	t.Helper()
	modelPath := filepath.Join(root, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}
	modelFile, err := os.Open(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	modelID, _, err := artifact.Identify(artifact.KindModel, modelFile)
	modelFile.Close()
	if err != nil {
		t.Fatal(err)
	}
	content, err := objective.Content()
	if err != nil {
		t.Fatal(err)
	}
	referenced := []artifact.ID{modelID, objective.Dataset, objective.Split, objective.Loss, objective.Evaluation}
	referenced = append(referenced, objective.Processors...)
	referenced = append(referenced, objective.Evidence...)
	for _, name := range []string{"precision", "placement", "memory", "checkpoint", "promotion"} {
		id, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("overgo/training-bootstrap/"+name))
		if err != nil {
			t.Fatal(err)
		}
		referenced = append(referenced, id)
	}
	var descriptors []artifact.Descriptor
	for _, id := range referenced {
		if _, held, err := store.Artifact(ctx, id); err != nil {
			t.Fatal(err)
		} else if !held {
			descriptors = append(descriptors, artifact.Descriptor{ID: id})
		}
	}
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key: "test-objective", Contents: []artifact.Content{content}, Artifacts: descriptors,
		Aliases: []artifact.AliasBinding{{Name: alias, Target: objective.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	recipeID, err := BootstrapObjectiveRecipe(ctx, store, modelPath, alias)
	if err != nil {
		t.Fatal(err)
	}
	observer, err := NewObserver(store, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := observer.Admit(ctx, modelID, recipeID); err != nil {
		t.Fatal(err)
	}
	observation, err := observer.Finish(ctx, modelID, recipeID, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	return observation
}

// commitBenchmarkReport commits one minimal typed campaign report so the
// approval gate has a real benchmark evaluation artifact to verify.
func commitBenchmarkReport(t *testing.T, ctx context.Context, store *overgodb.Store, name string, passed bool) artifact.ID {
	t.Helper()
	body, err := json.Marshal(map[string]any{"passed": passed})
	if err != nil {
		t.Fatal(err)
	}
	id, err := artifact.IdentifyBytes(artifact.KindEvaluation, body)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key: "test-evaluation-report-" + name,
		Contents: []artifact.Content{{
			Descriptor: artifact.Descriptor{
				ID: id, MediaType: "application/vnd.overgo.evaluation-campaign+json",
				Schema: "overgo/evaluation-campaign/v1", Size: uint64(len(body)),
			},
			Data: body,
		}},
	}); err != nil {
		t.Fatal(err)
	}
	return id
}
