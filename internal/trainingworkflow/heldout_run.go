package trainingworkflow

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

// HeldoutExample is one dataset record a model trains on or is judged on:
// its input, and its target values, label or accepted answers.
type HeldoutExample struct {
	// Record is the dataset record the example was read from; the runner
	// sets it before handing the example to Train or Predict.
	Record  string
	Input   []float64
	Target  []float64
	Label   string
	Answers []string
}

// HeldoutPrediction is a model's answer for one example: values, a label,
// or generated text.
type HeldoutPrediction struct {
	Values []float64
	Label  string
	Text   string
}

// HeldoutRun trains a model under a registered objective and judges it on
// the objective's held-out split. Examples holds every record of both
// memberships by record identity; Train sees only the training records, and
// the recipe the session activated for them, and Predict answers the
// held-out ones, once before training and once after.
type HeldoutRun struct {
	Alias     string
	ModelPath string
	Examples  map[string]HeldoutExample
	Train     func(ctx context.Context, recipe artifact.ID, shown []HeldoutExample) error
	Predict   func(HeldoutExample) (HeldoutPrediction, error)
	// Host records the training session as host execution.
	Host bool
}

// HeldoutResult names what a held-out run published.
type HeldoutResult struct {
	Observation, View, Plan, Baseline, Candidate, Verdict artifact.ID
	Passed                                                bool
}

// LoadObjectiveAlias loads the objective an alias names, refusing an alias
// the store does not bind.
func LoadObjectiveAlias(ctx context.Context, reader artifact.Reader, alias string) (trainingprogram.ObjectiveDocument, error) {
	id, bound, err := reader.ResolveAlias(ctx, alias)
	if err != nil {
		return trainingprogram.ObjectiveDocument{}, err
	}
	if !bound {
		return trainingprogram.ObjectiveDocument{}, fmt.Errorf("training workflow: objective alias %q is absent", alias)
	}
	return trainingprogram.LoadObjective(ctx, reader, id)
}

// RunHeldout trains through run.Train on the objective's training
// membership, records the training session, scores the base and trained
// model on its held-out membership, and publishes the view, plan, both
// reports and the verdict. Every held-out record must have an example: a
// view that skipped some would judge a chosen subset.
func RunHeldout(ctx context.Context, store *overgodb.Store, run HeldoutRun) (HeldoutResult, error) {
	objective, err := LoadObjectiveAlias(ctx, store, run.Alias)
	if err != nil {
		return HeldoutResult{}, err
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
	var cases []evaluation.HeldoutCase
	for _, record := range view.Records {
		example, found := run.Examples[record.ID]
		if !found {
			return HeldoutResult{}, fmt.Errorf("training workflow: held-out record %q has no example", record.ID)
		}
		cases = append(cases, evaluation.HeldoutCase{Record: record.ID, Values: example.Target, Label: example.Label, Answers: example.Answers})
	}
	plan, err := evaluation.CompileHeldoutPlan(objective, view, cases)
	if err != nil {
		return HeldoutResult{}, err
	}
	baseline, err := predictHeldout(plan, cases, run)
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
	trainErr := run.Train(ctx, recipeID, trainingExamples)
	observation, err := observer.Finish(ctx, modelID, recipeID, trainErr, uint64(len(trainingExamples)))
	if err := errors.Join(trainErr, err); err != nil {
		return HeldoutResult{}, err
	}
	candidate, err := predictHeldout(plan, cases, run)
	if err != nil {
		return HeldoutResult{}, err
	}
	return publishHeldout(ctx, store, objective, view, plan, baseline, candidate, observation)
}

// publishHeldout judges the base and trained models' reports on the
// objective's held-out view and publishes the view, plan, both reports and
// the verdict as one batch.
func publishHeldout(
	ctx context.Context,
	store artifact.Repository,
	objective trainingprogram.ObjectiveDocument,
	view evaluation.SFTEvaluationView,
	plan evaluation.HeldoutPlan,
	baseline, candidate evaluation.HeldoutReport,
	observation artifact.ID,
) (HeldoutResult, error) {
	verdict, err := evaluation.JudgeHeldout(objective, view, plan, baseline, candidate, observation)
	if err != nil {
		return HeldoutResult{}, err
	}
	batch, err := evaluation.HeldoutBatch(view, plan, baseline, candidate, verdict)
	if err != nil {
		return HeldoutResult{}, err
	}
	if _, err := artifact.Publish(ctx, store, batch); err != nil {
		return HeldoutResult{}, err
	}
	return HeldoutResult{
		Observation: observation, View: view.ID, Plan: plan.ID(), Baseline: baseline.ID(), Candidate: candidate.ID(),
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

// ObjectiveExamples materializes both of the objective's memberships through
// the one process the processor profile names, serving modality to itself,
// and decodes each processed record into its example, keyed by record. It
// also returns the identity of the materialized training membership, the
// stream a checkpoint records its position in.
func ObjectiveExamples(ctx context.Context, reader artifact.Reader, objective trainingprogram.ObjectiveDocument, profile []byte, modality recipecontract.Modality, process trainingdata.Processor, decode func(trainingdata.Example) (HeldoutExample, error)) (map[string]HeldoutExample, artifact.ID, error) {
	training, heldout, err := ObjectiveMemberships(ctx, reader, objective)
	if err != nil {
		return nil, artifact.ID{}, err
	}
	processor, err := artifact.IdentifyBytes(artifact.KindProfile, profile)
	if err != nil {
		return nil, artifact.ID{}, err
	}
	modalities := []recipecontract.Modality{modality}
	examples := map[string]HeldoutExample{}
	var stream artifact.ID
	for _, membership := range []dataset.Membership{training, heldout} {
		materialized, err := trainingdata.Materialize(ctx, reader, trainingdata.Authority{
			Dataset: objective.Dataset, Split: membership.ID, Processors: []artifact.ID{processor},
			Signature: recipecontract.ModalitySignature{Inputs: modalities, Outputs: modalities},
		}, []trainingdata.ProcessorBinding{{Artifact: processor, Modalities: modalities, Process: process}})
		if err != nil {
			return nil, artifact.ID{}, err
		}
		if membership.ID == training.ID {
			stream = materialized.Identity()
		}
		read, err := ReadAll(ctx, materialized)
		closeErr := materialized.Close()
		if err != nil || closeErr != nil {
			return nil, artifact.ID{}, fmt.Errorf("training workflow: %s membership: %w", membership.Partition, errors.Join(err, closeErr))
		}
		for _, record := range read {
			if examples[record.ID], err = decode(record); err != nil {
				return nil, artifact.ID{}, fmt.Errorf("training workflow: record %s: %w", record.ID, err)
			}
		}
	}
	return examples, stream, nil
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

func predictHeldout(plan evaluation.HeldoutPlan, cases []evaluation.HeldoutCase, run HeldoutRun) (evaluation.HeldoutReport, error) {
	observations := make([]evaluation.HeldoutObservation, 0, len(cases))
	for _, target := range cases {
		example := run.Examples[target.Record]
		example.Record = target.Record
		prediction, err := run.Predict(example)
		if err != nil {
			return evaluation.HeldoutReport{}, fmt.Errorf("training workflow: held-out record %q: %w", target.Record, err)
		}
		observations = append(observations, evaluation.HeldoutObservation{
			Record: target.Record, Values: prediction.Values, Label: prediction.Label, Text: prediction.Text,
		})
	}
	return plan.Score(observations)
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
