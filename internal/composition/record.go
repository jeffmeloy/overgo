package composition

import (
	"context"
	"fmt"
	"math"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
)

var (
	viabilityEvaluatorContract = artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: "application/vnd.overgo.viability-evaluator+json", Schema: "overgo/viability-evaluator/v1",
	}
	viabilityVerdictContract = artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: "application/vnd.overgo.viability-verdict+json", Schema: "overgo/viability-verdict/v1",
	}
)

const (
	baselineHeldOutMetric = "baseline-heldout-loss"
	worstHeldOutMetric    = "worst-seed-heldout-loss"
)

type viabilityEvaluator struct {
	Recipe  artifact.ID `json:"recipe"`
	Dataset artifact.ID `json:"dataset"`
	Split   artifact.ID `json:"split"`
	Metrics []string    `json:"metrics"`
}

type viabilityVerdict struct {
	Evaluation artifact.ID `json:"evaluation"`
	Ship       bool        `json:"ship"`
	Reason     string      `json:"reason"`
}

// RecordChainViability commits the Tier-0 chain experiment as a typed
// generation record: both whole models as parents, the content-addressed chain
// recipe as the training plan (no components, no bridge -- a whole-model
// child), and a decision identity carrying the SHIP or REFUSE reason. The
// probe records refusals exactly as loudly as ships.
func RecordChainViability(store *repodb.Store, config ChainConfig, result ChainResult) (runrecord.GenerationRecord, error) {
	must := func(kind artifact.Kind, payload string) artifact.ID {
		id, err := artifact.IdentifyBytes(kind, []byte("tier0-chain/v1:"+payload))
		if err != nil {
			panic("composition: identify " + payload + ": " + err.Error())
		}
		return id
	}
	protocol := fmt.Sprintf("scorer=%s;drafter=%s;prefix=%d;draft=%d;target=%d;windows=%d",
		config.ScorerDir, config.DrafterDir, config.Prefix, config.Draft, config.Target, len(config.Windows))
	outcome := runrecord.OutcomeFailed
	if result.Ship {
		outcome = runrecord.OutcomeSucceeded
	}
	record, err := runrecord.NewGenerationRecord(runrecord.GenerationRecord{
		Parents: []artifact.ID{
			must(artifact.KindModel, "model:"+config.ScorerDir),
			must(artifact.KindModel, "model:"+config.DrafterDir),
		},
		Child:        must(artifact.KindModel, "chained:"+protocol),
		TrainingPlan: result.ChainRecipe,
		Dataset:      must(artifact.KindDataset, "windows:"+protocol),
		Split:        must(artifact.KindDatasetShard, "targets:"+protocol),
		Evaluator:    must(artifact.KindEvidence, "evaluator:displaced-heldout-ce-envelope"),
		Code:         must(artifact.KindEvidence, "code:"+protocol),
		Environment:  must(artifact.KindEvidence, "environment:host-reference"),
		Seeds:        []uint64{0},
		Budget:       must(artifact.KindEvidence, fmt.Sprintf("budget:windows=%d;draft=%d;greedy-deterministic", len(config.Windows), config.Draft)),
		Run:          must(artifact.KindRun, "run:"+protocol+";"+result.Reason),
		Outcome:      outcome,
		Decision:     must(artifact.KindEvidence, "decision:"+result.Reason),
	})
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	// The chain and baseline recipes are NOT in this batch: the workflow
	// runtime already committed them with full content facts, and a bare
	// descriptor for the same identity would conflict.
	static := append([]artifact.ID{record.Child,
		record.Dataset, record.Split, record.Evaluator, record.Code, record.Environment,
		record.Budget, record.Run, record.Decision}, record.Parents...)
	descriptors := make([]artifact.Descriptor, len(static))
	for index, id := range static {
		descriptors[index] = artifact.Descriptor{ID: id}
	}
	ctx := context.Background()
	if _, err := store.Commit(ctx, artifact.Batch{Key: "tier0-chain/facts/" + record.ID.String(), Artifacts: descriptors}); err != nil {
		return runrecord.GenerationRecord{}, err
	}
	batch, err := record.Batch("tier0-chain/generation/" + record.ID.String())
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		return runrecord.GenerationRecord{}, err
	}
	if err := recordChainLifecycle(ctx, store, record, result); err != nil {
		return runrecord.GenerationRecord{}, err
	}
	return record, nil
}

// recordChainLifecycle commits the probe's runtime history as the immutable
// experiment state-machine chain: proposed through evaluated to the verdict
// terminal, each transition bound to the generation record's own evidence.
// The probe runs to completion in-process, so the recorded chain is the
// already-elapsed history -- expiries are the bounds the run held, not
// forward-looking leases.
func recordChainLifecycle(
	ctx context.Context,
	store *repodb.Store,
	record runrecord.GenerationRecord,
	result ChainResult,
) error {
	expiry := time.Now().UTC().Add(time.Hour).Format(time.RFC3339Nano)
	transition := func(prior *runrecord.ExperimentLifecycle, state runrecord.ExperimentState, evidence artifact.ID, bounded bool) (runrecord.ExperimentLifecycle, error) {
		value := runrecord.ExperimentLifecycle{
			State: state, Experiment: record.ID, Evidence: evidence,
		}
		if prior != nil {
			value.Retry = prior.Retry
		}
		if bounded {
			value.HeartbeatExpiry = expiry
		}
		return runrecord.NewExperimentLifecycle(value, prior)
	}
	verdict := runrecord.ExperimentRefused
	if record.Outcome == runrecord.OutcomeSucceeded {
		verdict = runrecord.ExperimentPromoted
	}
	proposed, err := transition(nil, runrecord.ExperimentProposed, record.TrainingPlan, false)
	if err != nil {
		return err
	}
	admitted, err := transition(&proposed, runrecord.ExperimentAdmitted, record.Budget, false)
	if err != nil {
		return err
	}
	leased, err := transition(&admitted, runrecord.ExperimentLeased, record.Environment, true)
	if err != nil {
		return err
	}
	running, err := transition(&leased, runrecord.ExperimentRunning, record.Run, true)
	if err != nil {
		return err
	}
	evaluated, err := transition(&running, runrecord.ExperimentEvaluated, record.Evaluator, false)
	if err != nil {
		return err
	}
	terminal, err := transition(&evaluated, verdict, record.Decision, false)
	if err != nil {
		return err
	}
	contents := make([]artifact.Content, 0, 6)
	for _, lifecycle := range []runrecord.ExperimentLifecycle{proposed, admitted, leased, running, evaluated, terminal} {
		content, err := lifecycle.Content()
		if err != nil {
			return err
		}
		contents = append(contents, content)
	}
	_, err = store.Commit(ctx, artifact.Batch{
		Key: "tier0-chain/lifecycle/" + record.ID.String(), Contents: contents,
	})
	return err
}

// RecordViability: atomic recipe, run, evaluation, verdict, and lineage.
func RecordViability(store *repodb.Store, config Config, result Result) (runrecord.GenerationRecord, error) {
	if result.Target.Kind() != artifact.KindModel || result.Donor.Kind() != artifact.KindModel ||
		result.Component.Kind() != artifact.KindTensorSet || result.Recipe.Kind() != artifact.KindRecipe ||
		result.Dataset.Kind() != artifact.KindDataset || result.Split.Kind() != artifact.KindDatasetShard ||
		len(result.Outcomes) == 0 || result.definition.ID != result.Recipe ||
		result.WorstHeldOut < 0 || math.IsNaN(result.WorstHeldOut) || math.IsInf(result.WorstHeldOut, 0) {
		return runrecord.GenerationRecord{}, fmt.Errorf("composition: viability authority is incomplete")
	}
	protocol := fmt.Sprintf("target=%s;donor=%s;tensor=%s;layer=%d;steps=%d;seeds=%d;lrscale=%g",
		result.Target, result.Donor, result.DonorTensor, result.GraftLayer, config.Steps, len(config.Seeds), config.LRScale)
	seeds := make([]uint64, len(config.Seeds))
	for index, seed := range config.Seeds {
		seeds[index] = uint64(seed)
	}
	outcome := runrecord.OutcomeFailed
	failure := "held-out-gain-absent"
	if result.Ship {
		outcome = runrecord.OutcomeSucceeded
		failure = ""
	}
	bridge, err := artifact.JSONID(artifact.KindAdapter, struct {
		Recipe    artifact.ID `json:"recipe"`
		Component artifact.ID `json:"component"`
		Protocol  string      `json:"protocol"`
	}{result.Recipe, result.Component, protocol})
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	child, err := artifact.JSONID(artifact.KindModel, struct {
		Target artifact.ID `json:"target"`
		Donor  artifact.ID `json:"donor"`
		Bridge artifact.ID `json:"bridge"`
	}{result.Target, result.Donor, bridge})
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	run, err := runrecord.NewRun(result.Recipe, outcome,
		[]artifact.ID{result.Target, result.Donor, result.Component, result.Dataset, result.Split},
		[]artifact.ID{child, bridge}, failure)
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	evaluation, err := runrecord.NewEvaluation(result.Recipe, run.ID, result.Dataset, []runrecord.Metric{
		{Name: baselineHeldOutMetric, Value: result.Baseline, Direction: runrecord.DirectionMinimize},
		{Name: worstHeldOutMetric, Value: result.WorstHeldOut, Direction: runrecord.DirectionMinimize},
	})
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	evaluator, err := artifact.JSONContent(viabilityEvaluatorContract, viabilityEvaluator{
		Recipe: result.Recipe, Dataset: result.Dataset, Split: result.Split,
		Metrics: []string{baselineHeldOutMetric, worstHeldOutMetric},
	})
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	verdict, err := artifact.JSONContent(viabilityVerdictContract, viabilityVerdict{
		Evaluation: evaluation.ID, Ship: result.Ship, Reason: result.Reason,
	})
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	identifyEvidence := func(payload string) (artifact.ID, error) {
		return artifact.IdentifyBytes(artifact.KindEvidence, []byte("graft-probe/v2:"+payload))
	}
	code, err := identifyEvidence("code:" + protocol)
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	environment, err := identifyEvidence("environment:host-reference")
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	budget, err := identifyEvidence(fmt.Sprintf("budget:seeds=%d;steps=%d", len(seeds), config.Steps))
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	record, err := runrecord.NewGenerationRecord(runrecord.GenerationRecord{
		Parents:      []artifact.ID{result.Target, result.Donor},
		Child:        child,
		Components:   []artifact.ID{result.Component},
		Bridge:       bridge,
		TrainingPlan: result.Recipe,
		Dataset:      result.Dataset,
		Split:        result.Split,
		Evaluator:    evaluator.Descriptor.ID,
		Code:         code,
		Environment:  environment,
		Seeds:        seeds,
		Budget:       budget,
		Run:          run.ID,
		Outcome:      outcome,
		Decision:     verdict.Descriptor.ID,
	})
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	static := append([]artifact.ID{record.Child, record.Bridge, record.Dataset,
		record.Split, record.Code, record.Environment, record.Budget}, record.Parents...)
	static = append(static, record.Components...)
	descriptors := make([]artifact.Descriptor, len(static))
	for index, id := range static {
		descriptors[index] = artifact.Descriptor{ID: id}
	}
	recipeContent, err := result.definition.ArtifactContent()
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	runContent, err := run.Content()
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	evaluationContent, err := evaluation.Content()
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	recordContent, err := record.Content()
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	lineage := append(run.Lineage(), evaluation.Lineage()...)
	lineage = append(lineage, artifact.DependencyLineage(
		evaluator.Descriptor.ID, result.Recipe, result.Dataset, result.Split,
	)...)
	lineage = append(lineage, artifact.DependencyLineage(verdict.Descriptor.ID, evaluation.ID)...)
	lineage = append(lineage, record.Lineage()...)
	ctx := context.Background()
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "graft-probe/generation/" + record.ID.String(), Artifacts: descriptors,
		Contents: []artifact.Content{recipeContent, runContent, evaluationContent, evaluator, verdict, recordContent},
		Lineage:  lineage,
	}); err != nil {
		return runrecord.GenerationRecord{}, err
	}
	return record, nil
}
