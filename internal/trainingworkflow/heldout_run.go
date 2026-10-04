package trainingworkflow

import (
	"context"
	"errors"
	"fmt"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
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
// the objective's held-out split. Data holds every record of both
// memberships as ObjectiveExamples read them; Train sees only the training
// records, and the recipe the session trains them under, and returns the
// checkpoint the trained model published; Predict answers the held-out
// records, once before training and once after.
type HeldoutRun struct {
	Alias     string
	ModelPath string
	// Recipe selects exact authority; zero uses the objective's bootstrapped recipe.
	Recipe  artifact.ID
	Data    ObjectiveData
	Train   func(ctx context.Context, recipe artifact.ID, shown []HeldoutExample) (trainingprogram.Checkpoint, error)
	Predict func(HeldoutExample) (HeldoutPrediction, error)
	// Load is how long the caller took to load what the run trains. It
	// loads before the run, whose untrained model answers the held-out
	// records first, so the session records the wall as its load phase
	// though its own wall begins at training.
	Load time.Duration
	// Host records the training session as host execution.
	Host bool
	// Proof re-verifies the recipe by the training run alone: neither model
	// is scored and no verdict publishes, so the newest verdict, which the
	// recipe's held-out quality reads, stays a full run's.
	Proof bool
}

// HeldoutResult names what a held-out run published: the session
// observation, the trained checkpoint, the view, plan, both reports and the
// verdict.
type HeldoutResult struct {
	Observation, Trained, View, Plan, Baseline, Candidate, Verdict artifact.ID
	Passed                                                         bool
}

// ObjectiveData is what a run reads of an objective's dataset through one
// processor: the example of each record of both memberships, keyed by
// record, and each membership's materialized identity. Training is the
// stream a checkpoint records its position in; Heldout is the exact
// evaluation inputs a held-out verdict binds.
type ObjectiveData struct {
	Examples          map[string]HeldoutExample
	Training, Heldout artifact.ID
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
// reports and the verdict, which binds the checkpoint training published
// and the held-out inputs both models read. Every held-out record must have
// an example: a view that skipped some would judge a chosen subset. A proof
// run stops once training records its session, and verifies the recipe on
// the session and the checkpoint alone, as the dense workflow does.
func RunHeldout(ctx context.Context, store *overgodb.Store, run HeldoutRun) (HeldoutResult, error) {
	modelID, objective, session, err := resolveHeldoutAuthority(ctx, store, run.ModelPath, run.Alias, run.Recipe)
	if err != nil {
		return HeldoutResult{}, err
	}
	return runHeldout(ctx, store, run, modelID, objective, session.Program.Definition().ID)
}

// resolveHeldoutAuthority resolves one model, objective alias and exact recipe before execution.
func resolveHeldoutAuthority(ctx context.Context, store *overgodb.Store, modelPath, alias string, recipeID artifact.ID) (artifact.ID, trainingprogram.ObjectiveDocument, SessionAuthority, error) {
	objective, err := LoadObjectiveAlias(ctx, store, alias)
	if err != nil {
		return artifact.ID{}, objective, SessionAuthority{}, err
	}
	modelID, err := identifyAndRecordWeights(ctx, store, modelPath)
	if err != nil {
		return modelID, objective, SessionAuthority{}, err
	}
	if !recipeID.Valid() {
		if recipeID, err = bootstrapRegisteredObjective(ctx, store, modelID, objective); err != nil {
			return modelID, objective, SessionAuthority{}, err
		}
	}
	session, err := ResolveSessionAuthority(ctx, store, modelID, recipeID)
	if err != nil {
		return modelID, objective, session, err
	}
	if bound, found := session.Program.Definition().PrimaryDependency(recipe.DependencyObjective); !found || bound != objective.ID {
		return modelID, objective, session, errors.New("training workflow: selected recipe objective differs from the registered objective")
	}
	return modelID, objective, session, nil
}

// runHeldout uses authority fixed before model execution, without re-resolving a mutable alias.
func runHeldout(ctx context.Context, store *overgodb.Store, run HeldoutRun, modelID artifact.ID, objective trainingprogram.ObjectiveDocument, recipeID artifact.ID) (HeldoutResult, error) {
	training, heldout, err := objectiveMemberships(ctx, store, objective)
	if err != nil {
		return HeldoutResult{}, err
	}
	trainingExamples, err := membershipExamples(training, run.Data.Examples)
	if err != nil {
		return HeldoutResult{}, err
	}
	view, err := evaluation.CompileSFTEvaluationView(objective, training, heldout)
	if err != nil {
		return HeldoutResult{}, err
	}
	var cases []evaluation.HeldoutCase
	for _, record := range view.Records {
		example, found := run.Data.Examples[record.ID]
		if !found {
			return HeldoutResult{}, fmt.Errorf("training workflow: held-out record %q has no example", record.ID)
		}
		cases = append(cases, evaluation.HeldoutCase{Record: record.ID, Values: example.Target, Label: example.Label, Answers: example.Answers})
	}
	plan, err := evaluation.CompileHeldoutPlan(objective, view, cases)
	if err != nil {
		return HeldoutResult{}, err
	}
	var baseline evaluation.HeldoutReport
	if !run.Proof {
		if baseline, err = predictHeldout(plan, cases, run); err != nil {
			return HeldoutResult{}, err
		}
	}
	observer, err := NewObserver(store, run.Host)
	if err != nil {
		return HeldoutResult{}, err
	}
	if err := observer.Admit(ctx, modelID, recipeID); err != nil {
		return HeldoutResult{}, err
	}
	observer.Phase(runrecord.PhaseLoad, run.Load)
	trained, trainErr := run.Train(ctx, recipeID, trainingExamples)
	// A cancelled run's outcome records too.
	observation, err := observer.Finish(context.WithoutCancel(ctx), modelID, recipeID, trainErr, uint64(len(trainingExamples)))
	if err := errors.Join(trainErr, err); err != nil {
		return HeldoutResult{}, err
	}
	if run.Proof {
		return HeldoutResult{Observation: observation, Trained: trained.ID()}, observer.Verify(ctx, recipeID, observation, trained.ID())
	}
	candidate, err := predictHeldout(plan, cases, run)
	if err != nil {
		return HeldoutResult{}, err
	}
	result, err := publishHeldout(ctx, store, objective, view, plan, baseline, candidate, evaluation.HeldoutTraining{
		Observation: observation, Model: modelID, Trained: trained, Inputs: run.Data.Heldout,
	})
	if err != nil {
		return HeldoutResult{}, err
	}
	return result, observer.Verify(ctx, recipeID, observation, trained.ID(), result.Verdict)
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
	training evaluation.HeldoutTraining,
) (HeldoutResult, error) {
	// A report is named by what it holds, so models that answer every held-out
	// record alike share one, and a verdict cannot judge a report against itself.
	if baseline.ID() == candidate.ID() {
		return HeldoutResult{}, errors.New("training workflow: the trained model answers every held-out record exactly as the base model does, so there is nothing to judge between them: train for more steps or on more data; the trained model is saved")
	}
	verdict, err := evaluation.JudgeHeldout(objective, view, plan, baseline, candidate, training)
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
		Observation: training.Observation, Trained: training.Trained.ID(), View: view.ID, Plan: plan.ID(),
		Baseline: baseline.ID(), Candidate: candidate.ID(), Verdict: verdict.ID, Passed: verdict.Passed,
	}, nil
}

// objectiveMemberships loads the objective's training membership and the
// held-out membership its published split pairs it with.
func objectiveMemberships(ctx context.Context, reader artifact.Reader, objective trainingprogram.ObjectiveDocument) (dataset.Membership, dataset.Membership, error) {
	training, found, err := dataset.LoadMembership(ctx, reader, objective.Split)
	if err != nil {
		return dataset.Membership{}, dataset.Membership{}, err
	}
	if !found {
		return dataset.Membership{}, dataset.Membership{}, fmt.Errorf("training workflow: objective %q names no committed training membership", objective.Name)
	}
	heldout, err := dataset.SplitPartner(ctx, reader, training.ID)
	if err != nil {
		return dataset.Membership{}, dataset.Membership{}, fmt.Errorf("training workflow: objective %q: %w", objective.Name, err)
	}
	return training, heldout, nil
}

// ObjectiveExamples materializes both of the objective's memberships through
// the one process the processor profile names, serving modality to itself,
// and decodes each processed record into its example.
func ObjectiveExamples(ctx context.Context, reader artifact.Reader, objective trainingprogram.ObjectiveDocument, profile []byte, modality recipecontract.Modality, process trainingdata.Processor, decode func(trainingdata.Example) (HeldoutExample, error)) (ObjectiveData, error) {
	training, heldout, err := objectiveMemberships(ctx, reader, objective)
	if err != nil {
		return ObjectiveData{}, err
	}
	processor, err := artifact.IdentifyBytes(artifact.KindProfile, profile)
	if err != nil {
		return ObjectiveData{}, err
	}
	modalities := []recipecontract.Modality{modality}
	data := ObjectiveData{Examples: map[string]HeldoutExample{}}
	for _, membership := range []dataset.Membership{training, heldout} {
		materialized, err := trainingdata.Materialize(ctx, reader, trainingdata.Authority{
			Dataset: objective.Dataset, Split: membership.ID, Processors: []artifact.ID{processor},
			Signature: recipecontract.ModalitySignature{Inputs: modalities, Outputs: modalities},
		}, []trainingdata.ProcessorBinding{{Artifact: processor, Modalities: modalities, Process: process}})
		if err != nil {
			return ObjectiveData{}, err
		}
		if membership.ID == training.ID {
			data.Training = materialized.Identity()
		} else {
			data.Heldout = materialized.Identity()
		}
		read, err := ReadAll(ctx, materialized)
		closeErr := materialized.Close()
		if err != nil || closeErr != nil {
			return ObjectiveData{}, fmt.Errorf("training workflow: %s membership: %w", membership.Partition, errors.Join(err, closeErr))
		}
		for _, record := range read {
			if data.Examples[record.ID], err = decode(record); err != nil {
				return ObjectiveData{}, fmt.Errorf("training workflow: record %s: %w", record.ID, err)
			}
		}
	}
	return data, nil
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
		example := run.Data.Examples[target.Record]
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
