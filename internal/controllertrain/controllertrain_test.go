package controllertrain

import (
	"fmt"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/workflowrecipe"
)

func TestCorpusRequiresExternalGroupAndTypedModule(t *testing.T) {
	spec := fixtureCorpusSpec()
	corpus, err := Compile(spec)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := corpus.PublicationBatch("controller/test-corpus")
	if err != nil || batch.Validate() != nil || corpus.Dataset().Kind() != artifact.KindDataset || corpus.Split().Kind() != artifact.KindDatasetShard {
		t.Fatalf("compiled corpus = (%s, %s, %v)", corpus.Dataset(), corpus.Split(), err)
	}
	leaked := fixtureCorpusSpec()
	leaked.Holdout[0].Group = leaked.Train[0].Group
	if _, err := Compile(leaked); err == nil {
		t.Fatal("controller corpus accepted cross-holdout group")
	}
	mismatch := fixtureCorpusSpec()
	mismatch.Train[1].Action.Module = workflowrecipe.ModuleProjectAudio
	if _, err := Compile(mismatch); err == nil {
		t.Fatal("controller corpus accepted modality/module mismatch")
	}
}

func TestDecisionRequiresMultiSeedGainAndExternalComparison(t *testing.T) {
	dataset := fixtureID(t, artifact.KindDataset, "dataset")
	split := fixtureID(t, artifact.KindDatasetShard, "split")
	champion := fixtureID(t, artifact.KindModel, "champion")
	candidate := fixtureID(t, artifact.KindModel, "candidate")
	incumbentEvaluation := fixtureID(t, artifact.KindEvaluation, "incumbent-evaluation")
	incumbentMetrics := SuiteMetrics{Loss: 2, ActionAccuracy: .125, ModalityAccuracy: .25, ValidActionRate: 1}
	evaluator := fixtureID(t, artifact.KindEvidence, "evaluator")
	seeds := make([]SeedEvidence, 3)
	for index := range seeds {
		seeds[index] = SeedEvidence{
			Seed: int64(index + 1), Model: fixtureID(t, artifact.KindModel, fmt.Sprintf("seed-%d", index)),
			Run:        fixtureID(t, artifact.KindRun, fmt.Sprintf("run-%d", index)),
			Evaluation: fixtureID(t, artifact.KindEvaluation, fmt.Sprintf("evaluation-%d", index)),
			Initial:    incumbentMetrics,
			Final:      SuiteMetrics{Loss: 1, ActionAccuracy: .5, ModalityAccuracy: .5, ValidActionRate: 1},
			Cost:       ResourceCost{WallNS: 1, PeakHostBytes: 1},
		}
	}
	seeds[1].Model = candidate
	budget, err := runrecord.NewBudget("queries", split, uint64(len(seeds)), fixtureID(t, artifact.KindEvidence, "budget-authority"))
	if err != nil {
		t.Fatal(err)
	}
	charges := make([]runrecord.BudgetCharge, len(seeds))
	for index := range charges {
		charges[index], err = runrecord.NewBudgetCharge(budget.ID, 1, evaluator, fmt.Sprintf("seed-%d", index))
		if err != nil {
			t.Fatal(err)
		}
	}
	spec := DecisionSpec{
		Dataset: dataset, Split: split, CurrentChampion: champion, Candidate: candidate, Seeds: seeds,
		Challengers: []Challenger{{
			Name: "current-champion", Model: champion, Evaluation: &incumbentEvaluation,
			Eligible: true, Metrics: &incumbentMetrics, Evaluator: evaluator, Split: split,
		}},
		Evaluator: evaluator, ObservedNoise: SuiteMetrics{}, Stochastic: true, Budget: budget, Charges: charges,
	}
	decision, err := Decide(spec)
	if err != nil || decision.State != DecisionPromote || decision.Rollback != champion {
		t.Fatalf("promotion = (%s, %s, %v)", decision.State, decision.Rollback, err)
	}
	content, err := decision.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseDecision(content.Data)
	if err != nil || parsed.ID != decision.ID {
		t.Fatalf("parsed promotion = (%s, %v)", parsed.ID, err)
	}
	spec.Seeds[0].Final.Loss = spec.Seeds[0].Initial.Loss
	refused, err := Decide(spec)
	if err != nil || refused.State != DecisionRefuse {
		t.Fatalf("weak promotion = (%s, %v)", refused.State, err)
	}
	spec.Challengers[0].Metrics = nil
	if _, err := Decide(spec); err == nil {
		t.Fatal("controller promotion accepted eligible challenger without metrics")
	}
}

func fixtureCorpusSpec() Spec {
	actions := []Action{
		{ScopeComponent, recipe.TaskGeneration, ModalityText, workflowrecipe.ModuleGenerate},
		{ScopeComponent, recipe.TaskProjection, ModalityImage, workflowrecipe.ModuleProjectImage},
		{ScopeComponent, recipe.TaskProjection, ModalityAudio, workflowrecipe.ModuleProjectAudio},
		{ScopeComponent, recipe.TaskProjection, ModalityVideo, workflowrecipe.ModuleProjectVideo},
		{ScopeWorkflow, recipe.TaskTraining, ModalityText, workflowrecipe.ModuleBatchDataset},
		{ScopeWorkflow, recipe.TaskTraining, ModalityImage, workflowrecipe.ModuleBatchDataset},
		{ScopeWorkflow, recipe.TaskTraining, ModalityAudio, workflowrecipe.ModuleBatchDataset},
		{ScopeWorkflow, recipe.TaskTraining, ModalityVideo, workflowrecipe.ModuleBatchDataset},
	}
	train, holdout := make([]Record, len(actions)), make([]Record, len(actions))
	for index, action := range actions {
		source := GitSource{Commit: "0123456789abcdef0123456789abcdef01234567", Path: "internal/workflowrecipe/catalog.go"}
		prompt := fmt.Sprintf("route action %d", index)
		train[index] = Record{ID: fmt.Sprintf("train-%d", index), Group: fmt.Sprintf("train-%d", index), Source: source, Prompt: prompt, Action: action}
		holdout[index] = Record{ID: fmt.Sprintf("holdout-%d", index), Group: fmt.Sprintf("holdout-%d", index), Source: source, Prompt: prompt, Action: action}
	}
	return Spec{Train: train, Holdout: holdout}
}

func fixtureID(t *testing.T, kind artifact.Kind, value string) artifact.ID {
	t.Helper()
	id, err := artifact.IdentifyBytes(kind, []byte(value))
	if err != nil {
		t.Fatal(err)
	}
	return id
}
