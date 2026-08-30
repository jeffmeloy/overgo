package controlleraction

import (
	"bytes"
	"cmp"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/composition"
	"overgo/internal/fsatomic"
	"overgo/internal/modelartifact"
	"overgo/internal/modelmerge"
	"overgo/internal/modelrecipe"
	"overgo/internal/runrecord"
)

var executableCodeCommit = runrecord.ExecutableCodeCommit

const (
	materializationDevice  = "host"
	materializationBackend = "go"
	materializationPurpose = "candidate materialization"
)

// CandidateMaterializationResult reports one cross-domain realization and its
// immutable output locations. Commit is set only by a new publication;
// Recovered distinguishes an idempotent lookup without mislabeling HEAD.
type CandidateMaterializationResult struct {
	Materialization modelrecipe.CandidateMaterialization
	Commit          artifact.CommitID
	Directories     map[artifact.ID]string
	Recovered       bool
}

// MaterializeCandidateTrial consumes one persisted realize decision, replays
// the stored compilation, executes every primary and declared ablation through
// the existing Go owners, and publishes their complete typed closure once.
func MaterializeCandidateTrial(
	ctx context.Context,
	repository artifact.Repository,
	decisionID artifact.ID,
	outputRoot string,
) (CandidateMaterializationResult, error) {
	if ctx == nil || repository == nil || decisionID.Kind() != artifact.KindEvidence ||
		strings.TrimSpace(outputRoot) == "" {
		return CandidateMaterializationResult{}, errors.New("controller action: materialization authority is absent")
	}
	if err := ctx.Err(); err != nil {
		return CandidateMaterializationResult{}, err
	}
	root, err := filepath.Abs(outputRoot)
	if err != nil {
		return CandidateMaterializationResult{}, err
	}
	if recovered, found, err := recoverCandidateMaterialization(
		ctx, repository, decisionID, root,
	); err != nil || found {
		return recovered, err
	}
	decision, err := runrecord.RequireDriverDecision(ctx, repository, decisionID)
	if err != nil {
		return CandidateMaterializationResult{}, err
	}
	candidateID, admissionID, err := selectedRealizeCandidate(decision)
	if err != nil {
		return CandidateMaterializationResult{}, err
	}
	candidate, err := modelrecipe.RequireCandidate(ctx, repository, candidateID)
	if err != nil {
		return CandidateMaterializationResult{}, err
	}
	code, err := runrecord.RequireCodeRevision(ctx, repository, candidate.Spec().Code)
	if err != nil {
		return CandidateMaterializationResult{}, err
	}
	commit, err := executableCodeCommit(".")
	if err != nil || commit != code.Commit {
		return CandidateMaterializationResult{}, errors.Join(
			errors.New("controller action: executable code revision differs from candidate"), err,
		)
	}
	environment, err := requireCurrentCandidateEnvironment(ctx, repository, candidate.Spec().Environment)
	if err != nil {
		return CandidateMaterializationResult{}, err
	}
	compiled, err := modelrecipe.RequireStoredCandidateCompilation(
		ctx, repository, candidateID, admissionID, composition.SameBaseCandidatePlugin{},
	)
	if err != nil {
		return CandidateMaterializationResult{}, err
	}
	head, _ := repository.Head()
	if err := clioptions.EnsureOutputDirectory(root); err != nil {
		return CandidateMaterializationResult{}, err
	}

	accumulator := newMaterializationBatchAccumulator()
	arms := make([]modelrecipe.CandidateMaterializedArm, 0, len(compiled.Components)+len(compiled.Ablations))
	directories := make(map[artifact.ID]string, cap(arms))
	ablations := make(map[artifact.ID]modelrecipe.CandidateAblation, len(compiled.Ablations))
	for _, ablation := range compiled.Ablations {
		ablations[ablation.ID] = ablation
	}
	for _, component := range compiled.Components {
		primary, batches, directory, err := materializeCandidateArm(
			ctx, repository, root, code.Commit, environment.ID,
			candidateID, admissionID, compiled.Trial.ID, compiled.EvaluationPlan.ID,
			candidate.Spec().Code, component.ID,
			artifact.ID{}, artifact.ID{}, component.Realization,
		)
		if err != nil {
			return CandidateMaterializationResult{}, err
		}
		if err := accumulator.add(batches...); err != nil {
			return CandidateMaterializationResult{}, err
		}
		arms = append(arms, primary)
		directories[primary.Realization] = directory
		for _, ablationID := range component.Ablations {
			ablation, found := ablations[ablationID]
			if !found {
				return CandidateMaterializationResult{}, errors.New("controller action: compiled ablation is absent")
			}
			arm, batches, directory, err := materializeCandidateArm(
				ctx, repository, root, code.Commit, environment.ID,
				candidateID, admissionID, compiled.Trial.ID, compiled.EvaluationPlan.ID,
				candidate.Spec().Code, component.ID,
				ablation.ID, ablation.Omitted, ablation.Realization,
			)
			if err != nil {
				return CandidateMaterializationResult{}, err
			}
			if err := accumulator.add(batches...); err != nil {
				return CandidateMaterializationResult{}, err
			}
			arms = append(arms, arm)
			directories[arm.Realization] = directory
		}
	}
	latest, err := runrecord.RequireDriverDecision(ctx, repository, decision.ID)
	if err != nil || latest.ID != decision.ID {
		return CandidateMaterializationResult{}, errors.Join(
			errors.New("controller action: materialization decision changed during execution"), err,
		)
	}
	cost, err := selectedRealizeCost(latest, candidateID)
	if err != nil {
		return CandidateMaterializationResult{}, err
	}
	charge, err := runrecord.NewBudgetCharge(latest.Resources.Grant, cost, latest.ID, materializationPurpose)
	if err != nil {
		return CandidateMaterializationResult{}, err
	}
	materialization, err := modelrecipe.NewCandidateMaterialization(latest, compiled, charge, arms)
	if err != nil {
		return CandidateMaterializationResult{}, err
	}
	chargeBatch, err := charge.Batch("candidate-materialization/charge/" + charge.ID.DigestHex())
	if err != nil {
		return CandidateMaterializationResult{}, err
	}
	materializationContent, err := materialization.Content()
	if err != nil {
		return CandidateMaterializationResult{}, err
	}
	materializationBatch, err := artifact.NewDocumentBatch(
		"candidate-materialization/closure/"+materialization.ID.DigestHex(),
		[]artifact.Content{materializationContent}, materialization.Lineage(), nil,
	)
	if err != nil {
		return CandidateMaterializationResult{}, err
	}
	if err := accumulator.add(chargeBatch, materializationBatch); err != nil {
		return CandidateMaterializationResult{}, err
	}
	batch, err := accumulator.batch(
		"candidate-materialization/"+materialization.ID.DigestHex(), head,
	)
	if err != nil {
		return CandidateMaterializationResult{}, err
	}
	committed, err := artifact.CommitBatch(ctx, repository, batch)
	if err != nil {
		return CandidateMaterializationResult{}, err
	}
	return CandidateMaterializationResult{
		Materialization: materialization, Commit: committed, Directories: directories,
	}, nil
}

func recoverCandidateMaterialization(
	ctx context.Context,
	repository artifact.Repository,
	decision artifact.ID,
	root string,
) (CandidateMaterializationResult, bool, error) {
	edges, err := repository.Children(ctx, decision)
	if err != nil {
		return CandidateMaterializationResult{}, false, err
	}
	identities := make(map[artifact.ID]struct{})
	for _, edge := range edges {
		if edge.Parent != decision || edge.Relation != artifact.RelationDependsOn ||
			edge.Child.Kind() != artifact.KindEvidence {
			continue
		}
		descriptor, found, err := repository.Artifact(ctx, edge.Child)
		if err != nil {
			return CandidateMaterializationResult{}, false, err
		}
		if found && descriptor.MediaType == modelrecipe.CandidateMaterializationMediaType &&
			descriptor.Schema == modelrecipe.CandidateMaterializationSchema {
			identities[edge.Child] = struct{}{}
		}
	}
	if len(identities) == 0 {
		return CandidateMaterializationResult{}, false, nil
	}
	if len(identities) != 1 {
		return CandidateMaterializationResult{}, false, errors.New(
			"controller action: materialization recovery is ambiguous",
		)
	}
	var id artifact.ID
	for candidate := range identities {
		id = candidate
	}
	materialization, err := RequireCandidateMaterialization(ctx, repository, id)
	if err != nil {
		return CandidateMaterializationResult{}, false, err
	}
	directories := make(map[artifact.ID]string, len(materialization.Arms))
	for _, arm := range materialization.Arms {
		destination := filepath.Join(root, arm.Realization.DigestHex())
		observed, observeErr := modelartifact.FromSafetensorsPath(destination)
		if observeErr == nil && observed.Manifest.ID == arm.Model &&
			observed.TensorInventory.ID == arm.TensorInventory {
			directories[arm.Realization] = destination
			continue
		}
		available, err := artifact.AvailablePath(
			ctx, repository, arm.Model, artifact.LocationDirectory,
		)
		if err != nil {
			return CandidateMaterializationResult{}, false, errors.Join(observeErr, err)
		}
		directories[arm.Realization] = available
	}
	return CandidateMaterializationResult{
		Materialization: materialization, Directories: directories, Recovered: true,
	}, true, nil
}

// RequireCandidateMaterialization reopens a committed realization through
// every production owner and re-derives the complete closure. Historical
// decisions are checked as recorded facts; they are intentionally no longer
// executable after their resource charge commits.
func RequireCandidateMaterialization(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (modelrecipe.CandidateMaterialization, error) {
	if ctx == nil || reader == nil || id.Kind() != artifact.KindEvidence {
		return modelrecipe.CandidateMaterialization{}, errors.New(
			"controller action: candidate materialization authority is absent",
		)
	}
	materialization, err := modelrecipe.RequireCandidateMaterialization(ctx, reader, id)
	if err != nil {
		return modelrecipe.CandidateMaterialization{}, err
	}
	decision, err := runrecord.RequireRecordedDriverDecision(ctx, reader, materialization.Decision)
	if err != nil {
		return modelrecipe.CandidateMaterialization{}, err
	}
	compiled, err := modelrecipe.RequireStoredCandidateCompilation(
		ctx, reader, materialization.Candidate, materialization.Admission,
		composition.SameBaseCandidatePlugin{},
	)
	if err != nil {
		return modelrecipe.CandidateMaterialization{}, err
	}
	if compiled.Trial.ID != materialization.Trial ||
		compiled.EvaluationPlan.ID != materialization.EvaluationPlan ||
		compiled.Trial.Code != materialization.Code ||
		compiled.Trial.Environment != materialization.Environment {
		return modelrecipe.CandidateMaterialization{}, errors.New(
			"controller action: materialization compilation closure differs",
		)
	}
	charge, err := runrecord.RequireBudgetCharge(ctx, reader, materialization.ResourceCharge)
	if err != nil {
		return modelrecipe.CandidateMaterialization{}, err
	}
	if charge.Purpose != materializationPurpose {
		return modelrecipe.CandidateMaterialization{}, errors.New(
			"controller action: materialization charge purpose differs",
		)
	}
	code, err := runrecord.RequireCodeRevision(ctx, reader, materialization.Code)
	if err != nil {
		return modelrecipe.CandidateMaterialization{}, err
	}
	if _, err := runrecord.RequireEnvironment(ctx, reader, materialization.Environment); err != nil {
		return modelrecipe.CandidateMaterialization{}, err
	}
	for _, arm := range materialization.Arms {
		if err := requireCandidateMaterializedArm(
			ctx, reader, materialization, compiled, arm, code.Commit,
		); err != nil {
			return modelrecipe.CandidateMaterialization{}, err
		}
	}
	replayed, err := modelrecipe.NewCandidateMaterialization(
		decision, compiled, charge, materialization.Arms,
	)
	if err != nil {
		return modelrecipe.CandidateMaterialization{}, err
	}
	if replayed.ID != materialization.ID || !reflect.DeepEqual(replayed, materialization) {
		return modelrecipe.CandidateMaterialization{}, errors.New(
			"controller action: candidate materialization replay differs",
		)
	}
	return materialization, nil
}

func materializeCandidateArm(
	ctx context.Context,
	reader artifact.Reader,
	root, codeCommit string,
	environment, candidate, admission, trial, evaluation, code, component,
	ablation, omitted, realization artifact.ID,
) (modelrecipe.CandidateMaterializedArm, []artifact.Batch, string, error) {
	plan, err := composition.LoadOfflineTensorExecutionPlan(ctx, reader, realization)
	if err != nil {
		return modelrecipe.CandidateMaterializedArm{}, nil, "", err
	}
	sources, err := resolveMaterializationSources(ctx, reader, plan)
	if err != nil {
		return modelrecipe.CandidateMaterializedArm{}, nil, "", err
	}
	destination := filepath.Join(root, realization.DigestHex())
	executed, duration, inventory, err := executeMaterializationPlan(ctx, plan, sources, destination, root)
	if err != nil {
		return modelrecipe.CandidateMaterializedArm{}, nil, "", err
	}
	if duration <= 0 || executed.TensorCount != len(plan.Operations) ||
		executed.PeakResidentBytes == 0 || executed.PeakResidentBytes > plan.PeakResidentBytes {
		return modelrecipe.CandidateMaterializedArm{}, nil, "", errors.New(
			"controller action: materialization execution envelope differs",
		)
	}
	base, err := modelrecipe.ResolveModelDefinition(ctx, reader, plan.Inputs[0].Definition)
	if err != nil {
		return modelrecipe.CandidateMaterializedArm{}, nil, "", err
	}
	definition, err := modelrecipe.NewModelDefinitionDocument(base.Profile, inventory.TensorInventory, base.Spec)
	if err != nil {
		return modelrecipe.CandidateMaterializedArm{}, nil, "", err
	}
	resolved, err := definition.Resolve(base.Profile, inventory.TensorInventory)
	if err != nil {
		return modelrecipe.CandidateMaterializedArm{}, nil, "", err
	}
	output, err := composition.NewOfflineArtifactOutput(plan, inventory, definition)
	if err != nil {
		return modelrecipe.CandidateMaterializedArm{}, nil, "", err
	}
	inputIDs := materializationRunInputs(
		plan, candidate, admission, trial, evaluation, code, component, ablation, omitted,
	)
	outputIDs := materializationRunOutputs(inputIDs, output)
	run, err := runrecord.NewBoundRun(
		plan.ArtifactPlan, runrecord.OutcomeSucceeded, inputIDs,
		outputIDs,
		"", codeCommit, environment, uint64(duration),
		[]runrecord.PhaseMetric{{Phase: runrecord.PhaseBuild, DurationNS: uint64(duration)}},
	)
	if err != nil {
		return modelrecipe.CandidateMaterializedArm{}, nil, "", err
	}
	observation, err := composition.NewOfflineArtifactObservation(
		plan, output, run, uint64(duration), executed.PeakResidentBytes,
	)
	if err != nil {
		return modelrecipe.CandidateMaterializedArm{}, nil, "", err
	}
	modelBatch, err := resolved.Batch("candidate-materialization/model/"+inventory.Manifest.ID.DigestHex(), inventory)
	if err != nil {
		return modelrecipe.CandidateMaterializedArm{}, nil, "", err
	}
	for _, input := range plan.Inputs {
		if inventory.Manifest.ID != input.Model {
			modelBatch.Lineage = append(modelBatch.Lineage, artifact.Lineage{
				Child: inventory.Manifest.ID, Parent: input.Model, Relation: artifact.RelationDerivedFrom,
			})
		}
	}
	outputBatch, err := documentBatch(
		"candidate-materialization/output/"+output.ID.DigestHex(), output.Content, output.Lineage(),
	)
	if err != nil {
		return modelrecipe.CandidateMaterializedArm{}, nil, "", err
	}
	runBatch, err := run.Batch("candidate-materialization/run/" + run.ID.DigestHex())
	if err != nil {
		return modelrecipe.CandidateMaterializedArm{}, nil, "", err
	}
	observationBatch, err := documentBatch(
		"candidate-materialization/observation/"+observation.ID.DigestHex(),
		observation.Content, observation.Lineage(),
	)
	if err != nil {
		return modelrecipe.CandidateMaterializedArm{}, nil, "", err
	}
	return modelrecipe.CandidateMaterializedArm{
		Component: component, Ablation: ablation, Omitted: omitted, Realization: realization,
		Model: output.Model, TensorInventory: output.TensorInventory, ModelDefinition: output.ModelDefinition,
		Output: output.ID, Run: run.ID, Observation: observation.ID,
		TensorBytes: output.TensorBytes, StoredBytes: output.StoredBytes,
		PeakResidentBytes: executed.PeakResidentBytes,
	}, []artifact.Batch{modelBatch, outputBatch, runBatch, observationBatch}, destination, nil
}

func uniqueMaterializationIDs(values []artifact.ID) []artifact.ID {
	values = slices.DeleteFunc(slices.Clone(values), func(id artifact.ID) bool { return !id.Valid() })
	slices.SortFunc(values, artifact.CompareID)
	return slices.Compact(values)
}

func materializationRunInputs(
	plan composition.OfflineTensorExecutionPlan,
	candidate, admission, trial, evaluation, code, component, ablation, omitted artifact.ID,
) []artifact.ID {
	values := []artifact.ID{
		plan.ID, candidate, admission, trial, evaluation, code, component, ablation, omitted,
	}
	for _, input := range plan.Inputs {
		values = append(values, input.Model, input.Definition, input.Inventory)
	}
	return uniqueMaterializationIDs(values)
}

func materializationRunOutputs(
	inputs []artifact.ID,
	output composition.OfflineArtifactOutput,
) []artifact.ID {
	values := []artifact.ID{output.ID}
	for _, id := range []artifact.ID{output.Model, output.TensorInventory, output.ModelDefinition} {
		if !slices.Contains(inputs, id) {
			values = append(values, id)
		}
	}
	return uniqueMaterializationIDs(values)
}

func requireCandidateMaterializedArm(
	ctx context.Context,
	reader artifact.Reader,
	materialization modelrecipe.CandidateMaterialization,
	compiled modelrecipe.CandidateCompilation,
	arm modelrecipe.CandidateMaterializedArm,
	codeCommit string,
) error {
	plan, err := composition.LoadOfflineTensorExecutionPlan(ctx, reader, arm.Realization)
	if err != nil {
		return err
	}
	output, err := composition.RequireOfflineArtifactOutput(ctx, reader, arm.Output)
	if err != nil {
		return err
	}
	run, err := runrecord.RequireExactRun(ctx, reader, arm.Run)
	if err != nil {
		return err
	}
	observation, err := composition.RequireOfflineArtifactObservation(ctx, reader, arm.Observation)
	if err != nil {
		return err
	}
	inputs := materializationRunInputs(
		plan, materialization.Candidate, materialization.Admission, materialization.Trial,
		materialization.EvaluationPlan, materialization.Code, arm.Component, arm.Ablation, arm.Omitted,
	)
	expectedRun, err := runrecord.NewBoundRun(
		plan.ArtifactPlan, runrecord.OutcomeSucceeded, inputs,
		materializationRunOutputs(inputs, output), "", codeCommit, materialization.Environment,
		run.MeasuredNS, []runrecord.PhaseMetric{{Phase: runrecord.PhaseBuild, DurationNS: run.MeasuredNS}},
	)
	if err != nil {
		return err
	}
	if expectedRun.ID != run.ID {
		return errors.New("controller action: materialization run replay differs")
	}
	replayedObservation, err := composition.NewOfflineArtifactObservation(
		plan, output, run, run.MeasuredNS, arm.PeakResidentBytes,
	)
	if err != nil {
		return err
	}
	if replayedObservation.ID != observation.ID || observation.ID != arm.Observation ||
		observation.PeakResidentBytes != arm.PeakResidentBytes {
		return errors.New("controller action: materialization observation replay differs")
	}
	if output.ID != arm.Output || output.Model != arm.Model ||
		output.TensorInventory != arm.TensorInventory ||
		output.ModelDefinition != arm.ModelDefinition || output.TensorBytes != arm.TensorBytes ||
		output.StoredBytes != arm.StoredBytes {
		return errors.New("controller action: materialization output closure differs")
	}
	if err := requireMaterializationOutputLineage(ctx, reader, plan, output, run.ID); err != nil {
		return err
	}
	if err := requireMaterializedModel(ctx, reader, plan, output, run); err != nil {
		return err
	}
	for _, component := range compiled.Components {
		if component.ID == arm.Component && arm.PeakResidentBytes <= component.PeakResidentBytes {
			return nil
		}
	}
	return errors.New("controller action: materialization component envelope differs")
}

func requireProducedDocumentLineage(
	ctx context.Context,
	reader artifact.Reader,
	child artifact.ID,
	immutable []artifact.Lineage,
	requiredRun artifact.ID,
) ([]runrecord.Run, error) {
	actual, err := reader.Parents(ctx, child)
	if err != nil {
		return nil, err
	}
	fixed := slices.DeleteFunc(slices.Clone(actual), func(edge artifact.Lineage) bool {
		return edge.Relation == artifact.RelationProducedBy
	})
	if !sameMaterializationLineage(fixed, immutable) {
		return nil, errors.New("controller action: materialization immutable lineage differs")
	}
	return requireMaterializationProducers(ctx, reader, child, actual, requiredRun)
}

func requireMaterializationOutputLineage(
	ctx context.Context,
	reader artifact.Reader,
	plan composition.OfflineTensorExecutionPlan,
	output composition.OfflineArtifactOutput,
	requiredRun artifact.ID,
) error {
	producers, err := requireProducedDocumentLineage(
		ctx, reader, output.ID, output.Lineage(), requiredRun,
	)
	if err != nil {
		return err
	}
	for _, producer := range producers {
		if producer.Recipe != plan.ArtifactPlan || !slices.Contains(producer.Inputs, plan.ID) {
			return errors.New("controller action: materialization output producer differs")
		}
	}
	return nil
}

func requireMaterializedModel(
	ctx context.Context,
	reader artifact.Reader,
	plan composition.OfflineTensorExecutionPlan,
	output composition.OfflineArtifactOutput,
	run runrecord.Run,
) error {
	storedManifest, found, err := reader.Manifest(ctx, output.Model)
	if err != nil || !found {
		return errors.Join(errors.New("controller action: materialized model manifest is absent"), err)
	}
	if err := storedManifest.Validate(); err != nil {
		return err
	}
	modelParents, err := reader.Parents(ctx, output.Model)
	if err != nil {
		return err
	}
	contains := slices.DeleteFunc(slices.Clone(modelParents), func(edge artifact.Lineage) bool {
		return edge.Relation != artifact.RelationContains
	})
	if !sameMaterializationLineage(contains, storedManifest.Lineage()) {
		return errors.New("controller action: materialized model component lineage differs")
	}
	for _, edge := range modelParents {
		switch edge.Relation {
		case artifact.RelationContains, artifact.RelationProducedBy:
		case artifact.RelationDerivedFrom:
			parent, found, err := reader.Manifest(ctx, edge.Parent)
			if err != nil || !found || edge.Parent == output.Model || parent.Validate() != nil {
				return errors.Join(errors.New("controller action: materialized model derivation differs"), err)
			}
		default:
			return errors.New("controller action: materialized model has foreign lineage")
		}
	}
	for _, input := range plan.Inputs {
		if input.Model == output.Model {
			continue
		}
		if !slices.Contains(modelParents, artifact.Lineage{
			Child: output.Model, Parent: input.Model, Relation: artifact.RelationDerivedFrom,
		}) {
			return errors.New("controller action: materialized model source lineage is absent")
		}
	}
	requiredModelRun := artifact.ID{}
	if slices.Contains(run.Outputs, output.Model) {
		requiredModelRun = run.ID
	}
	producers, err := requireMaterializationProducers(
		ctx, reader, output.Model, modelParents, requiredModelRun,
	)
	if err != nil {
		return err
	}
	producerInputs := make(map[artifact.ID]struct{})
	for _, producer := range producers {
		for _, input := range producer.Inputs {
			producerInputs[input] = struct{}{}
		}
	}
	if len(producerInputs) != 0 {
		for _, edge := range modelParents {
			if edge.Relation == artifact.RelationDerivedFrom {
				if _, found := producerInputs[edge.Parent]; !found {
					return errors.New("controller action: materialized model derivation lacks a producer input")
				}
			}
		}
	}
	storedInventory, found, err := modelartifact.ReadTensorInventoryDocument(
		ctx, reader, output.TensorInventory,
	)
	if err != nil || !found || storedInventory.Owner != output.Model {
		return errors.Join(errors.New("controller action: materialized tensor inventory is absent"), err)
	}
	requiredInventoryRun := artifact.ID{}
	if slices.Contains(run.Outputs, output.TensorInventory) {
		requiredInventoryRun = run.ID
	}
	if _, err := requireProducedDocumentLineage(
		ctx, reader, storedInventory.ID, storedInventory.Lineage(), requiredInventoryRun,
	); err != nil {
		return err
	}
	resolved, err := modelrecipe.ResolveModelDefinition(ctx, reader, output.ModelDefinition)
	if err != nil {
		return err
	}
	if resolved.Document.Model != output.Model || resolved.Tensors.ID != output.TensorInventory {
		return errors.New("controller action: materialized model definition differs")
	}
	requiredDefinitionRun := artifact.ID{}
	if slices.Contains(run.Outputs, output.ModelDefinition) {
		requiredDefinitionRun = run.ID
	}
	if _, err := requireProducedDocumentLineage(
		ctx, reader, resolved.Document.ID, resolved.Document.Lineage(), requiredDefinitionRun,
	); err != nil {
		return err
	}
	directory, err := artifact.AvailablePath(ctx, reader, output.Model, artifact.LocationDirectory)
	if err != nil {
		return err
	}
	observed, err := modelartifact.FromSafetensorsPath(directory)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(observed.Manifest, storedManifest) ||
		!reflect.DeepEqual(observed.TensorInventory, storedInventory) {
		return errors.New("controller action: materialized model bytes differ")
	}
	for _, descriptor := range observed.Components {
		stored, found, err := reader.Artifact(ctx, descriptor.ID)
		if err != nil || !found || stored != descriptor {
			return errors.Join(errors.New("controller action: materialized component differs"), err)
		}
	}
	return nil
}

func requireMaterializationProducers(
	ctx context.Context,
	reader artifact.Reader,
	child artifact.ID,
	parents []artifact.Lineage,
	requiredRun artifact.ID,
) ([]runrecord.Run, error) {
	foundRequired := !requiredRun.Valid()
	producers := make([]runrecord.Run, 0)
	for _, edge := range parents {
		if edge.Relation != artifact.RelationProducedBy {
			continue
		}
		if edge.Child != child || edge.Parent.Kind() != artifact.KindRun {
			return nil, errors.New("controller action: materialization producer lineage differs")
		}
		producer, err := runrecord.RequireExactRun(ctx, reader, edge.Parent)
		if err != nil {
			return nil, err
		}
		if !slices.Contains(producer.Outputs, child) {
			return nil, errors.New("controller action: materialization producer omits output")
		}
		producers = append(producers, producer)
		foundRequired = foundRequired || edge.Parent == requiredRun
	}
	if !foundRequired {
		return nil, errors.New("controller action: required materialization producer is absent")
	}
	return producers, nil
}

func sameMaterializationLineage(left, right []artifact.Lineage) bool {
	left = slices.Clone(left)
	right = slices.Clone(right)
	compare := func(left, right artifact.Lineage) int {
		return cmp.Or(
			artifact.CompareID(left.Child, right.Child),
			artifact.CompareID(left.Parent, right.Parent),
			cmp.Compare(left.Relation, right.Relation),
		)
	}
	slices.SortFunc(left, compare)
	slices.SortFunc(right, compare)
	return slices.Equal(left, right)
}

func executeMaterializationPlan(
	ctx context.Context,
	plan composition.OfflineTensorExecutionPlan,
	sources []modelmerge.StreamingSource,
	destination, root string,
) (modelmerge.StreamingResult, time.Duration, modelartifact.Inventory, error) {
	var noMaterializationTime time.Duration
	info, statErr := os.Stat(destination)
	if errors.Is(statErr, os.ErrNotExist) {
		return executeFreshMaterializationPlan(ctx, plan, sources, destination, root)
	}
	if statErr != nil {
		return modelmerge.StreamingResult{}, noMaterializationTime, modelartifact.Inventory{}, statErr
	}
	if !info.IsDir() {
		return modelmerge.StreamingResult{}, noMaterializationTime, modelartifact.Inventory{}, errors.New(
			"controller action: materialization destination is not a directory",
		)
	}
	existing, err := modelartifact.FromSafetensorsPath(destination)
	if err != nil {
		return modelmerge.StreamingResult{}, noMaterializationTime, modelartifact.Inventory{}, err
	}
	verificationRoot, err := os.MkdirTemp(root, ".materialization-verify-")
	if err != nil {
		return modelmerge.StreamingResult{}, noMaterializationTime, modelartifact.Inventory{}, err
	}
	cleanup := func(cause error) error {
		return errors.Join(cause, os.RemoveAll(verificationRoot))
	}
	verification := filepath.Join(verificationRoot, "output")
	started := time.Now()
	result, err := modelmerge.ExecuteStreaming(ctx, plan, sources, verification)
	duration := time.Since(started)
	if err != nil {
		return modelmerge.StreamingResult{}, noMaterializationTime, modelartifact.Inventory{}, cleanup(err)
	}
	replayed, err := modelartifact.FromSafetensorsPath(verification)
	if err != nil {
		return modelmerge.StreamingResult{}, noMaterializationTime, modelartifact.Inventory{}, cleanup(err)
	}
	if !reflect.DeepEqual(existing.Manifest, replayed.Manifest) ||
		!reflect.DeepEqual(existing.TensorInventory, replayed.TensorInventory) ||
		!reflect.DeepEqual(existing.Components, replayed.Components) {
		return modelmerge.StreamingResult{}, noMaterializationTime, modelartifact.Inventory{}, cleanup(errors.New(
			"controller action: existing materialization differs from deterministic replay",
		))
	}
	if err := cleanup(nil); err != nil {
		return modelmerge.StreamingResult{}, noMaterializationTime, modelartifact.Inventory{}, err
	}
	result.Destination = destination
	return result, duration, existing, nil
}

func executeFreshMaterializationPlan(
	ctx context.Context,
	plan composition.OfflineTensorExecutionPlan,
	sources []modelmerge.StreamingSource,
	destination, root string,
) (modelmerge.StreamingResult, time.Duration, modelartifact.Inventory, error) {
	var noMaterializationTime time.Duration
	privateRoot, err := os.MkdirTemp(root, ".materialization-execute-")
	if err != nil {
		return modelmerge.StreamingResult{}, noMaterializationTime, modelartifact.Inventory{}, err
	}
	cleanup := func(cause error) error {
		return errors.Join(cause, os.RemoveAll(privateRoot))
	}
	privateDestination := filepath.Join(privateRoot, "output")
	started := time.Now()
	result, err := modelmerge.ExecuteStreaming(ctx, plan, sources, privateDestination)
	duration := time.Since(started)
	if err != nil {
		return modelmerge.StreamingResult{}, noMaterializationTime, modelartifact.Inventory{}, cleanup(err)
	}
	if err := os.Rename(privateDestination, destination); err != nil {
		return modelmerge.StreamingResult{}, noMaterializationTime, modelartifact.Inventory{}, cleanup(err)
	}
	if err := fsatomic.SyncDirectory(root); err != nil {
		return modelmerge.StreamingResult{}, noMaterializationTime, modelartifact.Inventory{}, cleanup(err)
	}
	inventory, err := modelartifact.FromSafetensorsPath(destination)
	if err != nil {
		return modelmerge.StreamingResult{}, noMaterializationTime, modelartifact.Inventory{}, cleanup(err)
	}
	if err := cleanup(nil); err != nil {
		return modelmerge.StreamingResult{}, noMaterializationTime, modelartifact.Inventory{}, err
	}
	result.Destination = destination
	return result, duration, inventory, nil
}

func resolveMaterializationSources(
	ctx context.Context,
	reader artifact.Reader,
	plan composition.OfflineTensorExecutionPlan,
) ([]modelmerge.StreamingSource, error) {
	sources := make([]modelmerge.StreamingSource, len(plan.Inputs))
	for index, input := range plan.Inputs {
		directory, err := artifact.AvailablePath(ctx, reader, input.Model, artifact.LocationDirectory)
		if err != nil {
			return nil, err
		}
		observed, err := modelartifact.FromSafetensorsPath(directory)
		if err != nil {
			return nil, err
		}
		manifest, found, err := reader.Manifest(ctx, input.Model)
		if err != nil || !found {
			return nil, errors.Join(errors.New("controller action: source model manifest is absent"), err)
		}
		if !slices.Equal(materializationComponents(manifest), observed.Manifest.Components) {
			return nil, fmt.Errorf("controller action: source model bytes differ: %s", input.Model)
		}
		for _, descriptor := range observed.Components {
			stored, found, err := reader.Artifact(ctx, descriptor.ID)
			if err != nil || !found || stored != descriptor {
				return nil, errors.Join(
					fmt.Errorf("controller action: source component differs: %s", descriptor.ID), err,
				)
			}
		}
		expected, found, err := modelartifact.ReadTensorInventoryDocument(ctx, reader, input.Inventory)
		if err != nil || !found || expected.Owner != input.Model ||
			expected.Format != observed.TensorInventory.Format ||
			!reflect.DeepEqual(expected.Tensors, observed.TensorInventory.Tensors) {
			return nil, errors.Join(
				fmt.Errorf("controller action: source tensor inventory differs: %s", input.Inventory), err,
			)
		}
		shards, err := materializationShardBindings(directory, observed)
		if err != nil {
			return nil, err
		}
		sources[index] = modelmerge.StreamingSource{Model: input.Model, Directory: directory, Shards: shards}
	}
	return sources, nil
}

func materializationShardBindings(
	directory string,
	inventory modelartifact.Inventory,
) ([]modelmerge.StreamingShard, error) {
	weights := make(map[artifact.ID]struct{})
	for _, component := range inventory.Manifest.Components {
		if component.Role == artifact.ComponentWeights || component.Role == artifact.ComponentWeightsShard {
			weights[component.Artifact] = struct{}{}
		}
	}
	bindings := make([]modelmerge.StreamingShard, 0, len(weights))
	for _, location := range inventory.Locations {
		if location.Kind != artifact.LocationFile {
			continue
		}
		if _, found := weights[location.Artifact]; !found {
			continue
		}
		name, err := filepath.Rel(directory, location.Value)
		if err != nil || name == "." || filepath.IsAbs(name) || name == ".." ||
			strings.HasPrefix(name, ".."+string(filepath.Separator)) {
			return nil, errors.Join(errors.New("controller action: source shard location escapes model"), err)
		}
		bindings = append(bindings, modelmerge.StreamingShard{
			Name: filepath.ToSlash(name), Artifact: location.Artifact,
		})
	}
	if len(bindings) == 0 {
		return nil, errors.New("controller action: source shard locations are absent")
	}
	slices.SortFunc(bindings, func(left, right modelmerge.StreamingShard) int {
		return strings.Compare(left.Name, right.Name)
	})
	return bindings, nil
}

func materializationComponents(manifest artifact.Manifest) []artifact.Component {
	return slices.DeleteFunc(slices.Clone(manifest.Components), func(component artifact.Component) bool {
		return component.Role != artifact.ComponentWeights &&
			component.Role != artifact.ComponentWeightsShard && component.Role != artifact.ComponentShardIndex
	})
}

func selectedRealizeCandidate(decision runrecord.DriverDecision) (artifact.ID, artifact.ID, error) {
	if decision.Stop != runrecord.DriverStopContinue || decision.Action != runrecord.DriverActionRealize ||
		decision.Need != runrecord.DriverNeedRealization || decision.Subject.Kind() != artifact.KindRecipe ||
		decision.Target != decision.Subject {
		return artifact.ID{}, artifact.ID{}, errors.New("controller action: driver decision does not select realization")
	}
	for _, option := range decision.Options {
		if option.Candidate == decision.Subject && option.State == runrecord.DriverCandidateEligible {
			return option.Candidate, option.Admission, nil
		}
	}
	return artifact.ID{}, artifact.ID{}, errors.New("controller action: selected candidate admission is absent")
}

func selectedRealizeCost(decision runrecord.DriverDecision, candidate artifact.ID) (uint64, error) {
	for _, option := range decision.Options {
		if option.Candidate != candidate || option.State != runrecord.DriverCandidateEligible {
			continue
		}
		for _, gap := range option.Missing {
			if gap.Need == runrecord.DriverNeedRealization && gap.Target == candidate &&
				gap.CostUnit == decision.Resources.Unit {
				return gap.CostUnits, nil
			}
		}
	}
	return 0, errors.New("controller action: selected realization cost is absent")
}

func requireCurrentCandidateEnvironment(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (runrecord.Environment, error) {
	expected, err := runrecord.RequireEnvironment(ctx, reader, id)
	if err != nil {
		return runrecord.Environment{}, err
	}
	current, err := runrecord.CurrentEnvironment(materializationDevice, materializationBackend)
	if err != nil {
		return runrecord.Environment{}, err
	}
	if current != expected {
		return runrecord.Environment{}, errors.New("controller action: materialization environment differs")
	}
	return current, nil
}

func documentBatch(
	key string,
	content func() (artifact.Content, error),
	lineage []artifact.Lineage,
) (artifact.Batch, error) {
	document, err := content()
	if err != nil {
		return artifact.Batch{}, err
	}
	return artifact.NewDocumentBatch(key, []artifact.Content{document}, lineage, nil)
}

type materializationBatchAccumulator struct {
	descriptors map[artifact.ID]artifact.Descriptor
	contents    map[artifact.ID]artifact.Content
	manifests   map[artifact.ID]artifact.Manifest
	lineage     map[artifact.Lineage]struct{}
	causality   map[artifact.ID]artifact.CausalLink
	locations   map[artifact.LocationEvent]struct{}
}

func newMaterializationBatchAccumulator() *materializationBatchAccumulator {
	return &materializationBatchAccumulator{
		descriptors: make(map[artifact.ID]artifact.Descriptor),
		contents:    make(map[artifact.ID]artifact.Content), manifests: make(map[artifact.ID]artifact.Manifest),
		lineage: make(map[artifact.Lineage]struct{}), causality: make(map[artifact.ID]artifact.CausalLink),
		locations: make(map[artifact.LocationEvent]struct{}),
	}
}

func (accumulator *materializationBatchAccumulator) add(batches ...artifact.Batch) error {
	for _, batch := range batches {
		if len(batch.Aliases) != 0 {
			return errors.New("controller action: candidate materialization cannot mutate aliases")
		}
		for _, descriptor := range batch.Artifacts {
			if prior, found := accumulator.descriptors[descriptor.ID]; found && prior != descriptor {
				return fmt.Errorf("controller action: conflicting descriptor %s", descriptor.ID)
			}
			accumulator.descriptors[descriptor.ID] = descriptor
		}
		for _, content := range batch.Contents {
			if prior, found := accumulator.contents[content.Descriptor.ID]; found &&
				(prior.Descriptor != content.Descriptor || !bytes.Equal(prior.Data, content.Data)) {
				return fmt.Errorf("controller action: conflicting content %s", content.Descriptor.ID)
			}
			accumulator.contents[content.Descriptor.ID] = content.Clone()
		}
		for _, manifest := range batch.Manifests {
			if prior, found := accumulator.manifests[manifest.ID]; found && !reflect.DeepEqual(prior, manifest) {
				return fmt.Errorf("controller action: conflicting manifest %s", manifest.ID)
			}
			accumulator.manifests[manifest.ID] = manifest.Clone()
		}
		for _, edge := range batch.Lineage {
			accumulator.lineage[edge] = struct{}{}
		}
		for _, link := range batch.Causality {
			if prior, found := accumulator.causality[link.Execution]; found && !prior.Equal(link) {
				return fmt.Errorf("controller action: conflicting causality %s", link.Execution)
			}
			accumulator.causality[link.Execution] = link.Clone()
		}
		for _, location := range batch.Locations {
			accumulator.locations[location] = struct{}{}
		}
	}
	return nil
}

func (accumulator *materializationBatchAccumulator) batch(
	key string,
	head artifact.CommitID,
) (artifact.Batch, error) {
	batch := artifact.Batch{Key: key, ExpectedHead: &head}
	for _, descriptor := range accumulator.descriptors {
		batch.Artifacts = append(batch.Artifacts, descriptor)
	}
	for _, content := range accumulator.contents {
		batch.Contents = append(batch.Contents, content)
	}
	for _, manifest := range accumulator.manifests {
		batch.Manifests = append(batch.Manifests, manifest)
	}
	for edge := range accumulator.lineage {
		batch.Lineage = append(batch.Lineage, edge)
	}
	for _, link := range accumulator.causality {
		batch.Causality = append(batch.Causality, link)
	}
	for location := range accumulator.locations {
		batch.Locations = append(batch.Locations, location)
	}
	slices.SortFunc(batch.Artifacts, func(left, right artifact.Descriptor) int {
		return artifact.CompareID(left.ID, right.ID)
	})
	slices.SortFunc(batch.Contents, func(left, right artifact.Content) int {
		return artifact.CompareID(left.Descriptor.ID, right.Descriptor.ID)
	})
	slices.SortFunc(batch.Manifests, func(left, right artifact.Manifest) int {
		return artifact.CompareID(left.ID, right.ID)
	})
	slices.SortFunc(batch.Lineage, func(left, right artifact.Lineage) int {
		return cmp.Or(
			artifact.CompareID(left.Child, right.Child), artifact.CompareID(left.Parent, right.Parent),
			cmp.Compare(left.Relation, right.Relation),
		)
	})
	slices.SortFunc(batch.Causality, func(left, right artifact.CausalLink) int {
		return artifact.CompareID(left.Execution, right.Execution)
	})
	slices.SortFunc(batch.Locations, func(left, right artifact.LocationEvent) int {
		return cmp.Or(
			artifact.CompareID(left.Artifact, right.Artifact), cmp.Compare(left.Kind, right.Kind),
			strings.Compare(left.Value, right.Value), cmp.Compare(left.Action, right.Action),
		)
	})
	if err := batch.Validate(); err != nil {
		return artifact.Batch{}, err
	}
	return batch, nil
}
