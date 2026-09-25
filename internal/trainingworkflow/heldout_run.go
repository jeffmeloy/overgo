package trainingworkflow

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

// HeldoutExample is one dataset record a model trains on or is judged on:
// its input, and its target values or label.
type HeldoutExample struct {
	// Record is the dataset record the example was read from; the runner
	// sets it before handing the example to Train or Predict.
	Record string
	Input  []float64
	Target []float64
	Label  string
}

// HeldoutPrediction is a model's answer for one example.
type HeldoutPrediction struct {
	Values []float64
	Label  string
}

// HeldoutRun trains a model under a registered objective and judges it on
// the objective's held-out split. Examples holds every record of both
// memberships by record identity; Train sees only the training records and
// Predict answers the held-out ones, once before training and once after.
type HeldoutRun struct {
	Alias     string
	ModelPath string
	Examples  map[string]HeldoutExample
	Train     func(context.Context, []HeldoutExample) error
	Predict   func(HeldoutExample) (HeldoutPrediction, error)
	// Host records the training session as host execution.
	Host bool
}

// HeldoutResult names what a held-out run published.
type HeldoutResult struct {
	Observation, View, Plan, Baseline, Candidate, Verdict artifact.ID
	Passed                                                bool
}

// RunHeldout trains through run.Train on the objective's training
// membership, records the training session, scores the base and trained
// model on its held-out membership, and publishes the view, plan, both
// reports and the verdict. Every held-out record must have an example: a
// view that skipped some would judge a chosen subset.
func RunHeldout(ctx context.Context, store *overgodb.Store, run HeldoutRun) (HeldoutResult, error) {
	objectiveID, bound, err := store.ResolveAlias(ctx, run.Alias)
	if err != nil {
		return HeldoutResult{}, err
	}
	if !bound {
		return HeldoutResult{}, fmt.Errorf("training workflow: objective alias %q is absent", run.Alias)
	}
	objective, err := trainingprogram.LoadObjective(ctx, store, objectiveID)
	if err != nil {
		return HeldoutResult{}, err
	}
	scorer, found := evaluation.HeldoutNumericScorer(objective.Metric)
	if !found {
		return HeldoutResult{}, fmt.Errorf("training workflow: no held-out scorer measures %q", objective.Metric)
	}
	training, heldout, err := ObjectiveMemberships(ctx, store, objective)
	if err != nil {
		return HeldoutResult{}, err
	}
	trainingExamples, err := membershipExamples(training, run.Examples)
	if err != nil {
		return HeldoutResult{}, err
	}
	view, err := evaluation.CompileSFTEvaluationView(objective, training, heldout)
	if err != nil {
		return HeldoutResult{}, err
	}
	suite := evaluation.NumericTargetSuite{Scorers: []evaluation.NumericScorerSpec{{Kind: scorer}}}
	for _, record := range view.Records {
		example, found := run.Examples[record.ID]
		if !found {
			return HeldoutResult{}, fmt.Errorf("training workflow: held-out record %q has no example", record.ID)
		}
		suite.Cases = append(suite.Cases, evaluation.NumericTargetCase{Record: record.ID, Values: slices.Clone(example.Target), Label: example.Label})
	}
	plan, err := evaluation.CompileNumericTargetPlan(view, suite)
	if err != nil {
		return HeldoutResult{}, err
	}
	baseline, err := predictHeldout(plan, run)
	if err != nil {
		return HeldoutResult{}, err
	}
	recipeID, err := BootstrapObjectiveRecipe(ctx, store, run.ModelPath, run.Alias)
	if err != nil {
		return HeldoutResult{}, err
	}
	modelID, err := identifyAndRecordWeights(ctx, store, run.ModelPath)
	if err != nil {
		return HeldoutResult{}, err
	}
	observer, err := NewObserver(store, run.Host)
	if err != nil {
		return HeldoutResult{}, err
	}
	if err := observer.Admit(ctx, modelID, recipeID); err != nil {
		return HeldoutResult{}, err
	}
	trainErr := run.Train(ctx, trainingExamples)
	observation, err := observer.Finish(ctx, modelID, recipeID, trainErr, uint64(len(trainingExamples)))
	if err := errors.Join(trainErr, err); err != nil {
		return HeldoutResult{}, err
	}
	candidate, err := predictHeldout(plan, run)
	if err != nil {
		return HeldoutResult{}, err
	}
	verdict, err := evaluation.JudgeHeldout(objective, view, plan, baseline, candidate, observation)
	if err != nil {
		return HeldoutResult{}, err
	}
	batch := artifact.Batch{Key: "training/heldout/" + verdict.ID.String()}
	for _, stage := range []func(string) (artifact.Batch, error){view.Batch, plan.Batch, baseline.Batch, candidate.Batch, verdict.Batch} {
		part, err := stage(batch.Key)
		if err != nil {
			return HeldoutResult{}, err
		}
		batch.Contents = append(batch.Contents, part.Contents...)
		batch.Lineage = append(batch.Lineage, part.Lineage...)
	}
	if _, err := artifact.Publish(ctx, store, batch); err != nil {
		return HeldoutResult{}, err
	}
	return HeldoutResult{
		Observation: observation, View: view.ID, Plan: plan.ID, Baseline: baseline.ID, Candidate: candidate.ID,
		Verdict: verdict.ID, Passed: verdict.Passed,
	}, nil
}

// ObjectiveMemberships loads the objective's training membership and the
// held-out membership of the same split: the other partition of its source
// under the same seed.
func ObjectiveMemberships(ctx context.Context, reader artifact.Reader, objective trainingprogram.ObjectiveDocument) (dataset.Membership, dataset.Membership, error) {
	training, found, err := dataset.LoadMembership(ctx, reader, objective.Split)
	if err != nil {
		return dataset.Membership{}, dataset.Membership{}, err
	}
	if !found {
		return dataset.Membership{}, dataset.Membership{}, fmt.Errorf("training workflow: objective %q names no committed training membership", objective.Name)
	}
	children, err := reader.Children(ctx, training.Source)
	if err != nil {
		return dataset.Membership{}, dataset.Membership{}, err
	}
	var heldout []dataset.Membership
	for _, edge := range children {
		membership, found, err := dataset.LoadMembership(ctx, reader, edge.Child)
		if err != nil || !found || membership.ID == training.ID || membership.Seed != training.Seed {
			continue
		}
		heldout = append(heldout, membership)
	}
	if len(heldout) != 1 {
		return dataset.Membership{}, dataset.Membership{}, fmt.Errorf("training workflow: objective %q split has %d held-out memberships, want one", objective.Name, len(heldout))
	}
	return training, heldout[0], nil
}

func membershipExamples(membership dataset.Membership, examples map[string]HeldoutExample) ([]HeldoutExample, error) {
	selected := make([]HeldoutExample, 0, len(membership.Records))
	for _, record := range membership.Records {
		example, found := examples[record.ID]
		if !found {
			return nil, fmt.Errorf("training workflow: training record %q has no example", record.ID)
		}
		example.Record = record.ID
		selected = append(selected, example)
	}
	return selected, nil
}

func predictHeldout(plan evaluation.NumericTargetPlan, run HeldoutRun) (evaluation.NumericTargetReport, error) {
	observations := make([]evaluation.NumericTargetObservation, 0, len(plan.Cases))
	for _, target := range plan.Cases {
		example := run.Examples[target.Record]
		example.Record = target.Record
		prediction, err := run.Predict(example)
		if err != nil {
			return evaluation.NumericTargetReport{}, fmt.Errorf("training workflow: held-out record %q: %w", target.Record, err)
		}
		observations = append(observations, evaluation.NumericTargetObservation{Record: target.Record, Values: prediction.Values, Label: prediction.Label})
	}
	return evaluation.ScoreNumericTargets(plan, observations)
}

// readAllDecodeWorkers decodes a one-time read sequentially: ReadAll serves
// evaluation and small splits, where parallel decode buys nothing.
const readAllDecodeWorkers = 1

// ReadAll reads every record of a materialized dataset once, processed, in
// stream order: the stream cycles epochs, so one batch the size of the
// dataset holds each record exactly once. It suits evaluation and small
// splits; training streams through a trainingdata.Batcher.
func ReadAll(ctx context.Context, materialized *trainingdata.Dataset) ([]trainingdata.Example, error) {
	stream, err := trainingdata.NewStream(materialized, nil)
	if err != nil {
		return nil, err
	}
	records := materialized.Records()
	batcher, err := trainingdata.NewBatcher(stream, trainingdata.BatchPolicy{Examples: records, MicrobatchExamples: records, DecodeWorkers: readAllDecodeWorkers})
	if err != nil {
		return nil, err
	}
	batch, err := batcher.Next(ctx)
	return batch.Examples, err
}
