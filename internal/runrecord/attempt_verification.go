package runrecord

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// AttemptGateVerification is the complete immutable authority chain for one
// successful plan attempt. The preparation is intentionally introduced before
// the other records; Gate, Run, Finalization, and the Attempt itself must share
// one atomic store introduction.
type AttemptGateVerification struct {
	Gate         GateResult
	Run          Run
	Preparation  GateLifecycle
	Finalization GateLifecycle
}

// VerifyAttemptGate requires the exact stored successful attempt and follows
// typed lineage to its one successful gate run and lifecycle finalization.
func VerifyAttemptGate(
	ctx context.Context,
	store *overgodb.Store,
	attempt AttemptRecord,
) (AttemptGateVerification, error) {
	if ctx == nil || store == nil {
		return AttemptGateVerification{}, errors.New("run record: attempt gate verification requires the store")
	}
	if err := attempt.ValidateIdentity(); err != nil {
		return AttemptGateVerification{}, err
	}
	stored, err := RequireAttemptRecord(ctx, store, attempt.ID)
	if err != nil {
		return AttemptGateVerification{}, err
	}
	// Both values validate against the same content-addressed identity, so an
	// equal ID proves that the caller supplied the exact stored value.
	if stored.ID != attempt.ID || stored.Outcome != OutcomeSucceeded || stored.Failure != "" {
		return AttemptGateVerification{}, errors.New("run record: attempt is not the exact stored success")
	}

	gate, err := RequireGateResult(ctx, store, stored.Result)
	if err != nil {
		return AttemptGateVerification{}, err
	}
	if gate.ID != stored.Result || gate.Recipe != stored.Recipe || gate.CodeCommit != stored.CodeCommit ||
		gate.Outcome != stored.Outcome || gate.Failure != stored.Failure {
		return AttemptGateVerification{}, errors.New("run record: attempt and gate authorities differ")
	}
	if !successfulGateStep(gate, "commit") {
		return AttemptGateVerification{}, errors.New("run record: attempt gate lacks a successful commit step")
	}

	verification, err := uniqueGateRun(ctx, store, gate)
	if err != nil {
		return AttemptGateVerification{}, err
	}
	finalization, preparation, err := uniqueGateFinalization(ctx, store, gate)
	if err != nil {
		return AttemptGateVerification{}, err
	}
	preparationFinalization, found, err := GateFinalizationForPreparation(ctx, store, preparation.ID)
	if err != nil {
		return AttemptGateVerification{}, err
	}
	if !found || preparationFinalization.ID != finalization.ID {
		return AttemptGateVerification{}, errors.New("run record: gate preparation does not have one exact finalization")
	}
	preparationIntroduction, found, err := store.ArtifactIntroduction(ctx, preparation.ID)
	if err != nil {
		return AttemptGateVerification{}, err
	}
	if !found {
		return AttemptGateVerification{}, errors.New("run record: gate preparation introduction is absent")
	}
	environmentIntroduction, found, err := store.ArtifactIntroduction(ctx, preparation.Environment)
	if err != nil {
		return AttemptGateVerification{}, err
	}
	if !found {
		return AttemptGateVerification{}, errors.New("run record: gate environment introduction is absent")
	}
	if environmentIntroduction.Sequence > preparationIntroduction.Sequence {
		return AttemptGateVerification{}, errors.New("run record: gate environment was published after its preparation")
	}

	ids := []artifact.ID{stored.ID, gate.ID, verification.Run.ID, finalization.ID}
	introductions := make([]overgodb.ArtifactIntroduction, 0, len(ids))
	for _, id := range ids {
		introduction, found, introductionErr := store.ArtifactIntroduction(ctx, id)
		if introductionErr != nil {
			return AttemptGateVerification{}, introductionErr
		}
		if !found {
			return AttemptGateVerification{}, fmt.Errorf("run record: artifact introduction is absent: %s", id)
		}
		introductions = append(introductions, introduction)
	}
	for _, introduction := range introductions[1:] {
		if introduction.Commit != introductions[0].Commit || introduction.Sequence != introductions[0].Sequence {
			return AttemptGateVerification{}, errors.New("run record: attempt gate lifecycle was not published atomically")
		}
	}
	if preparationIntroduction.Sequence >= introductions[0].Sequence {
		return AttemptGateVerification{}, errors.New("run record: gate preparation was not published before the final attempt batch")
	}
	return AttemptGateVerification{
		Gate: gate, Run: verification.Run, Preparation: preparation, Finalization: finalization,
	}, nil
}

func successfulGateStep(gate GateResult, name string) bool {
	for _, step := range gate.Steps {
		if step.Name == name {
			return step.Outcome == StepSucceeded
		}
	}
	return false
}

func uniqueGateRun(ctx context.Context, store *overgodb.Store, gate GateResult) (Verification, error) {
	edges, err := store.Parents(ctx, gate.ID)
	if err != nil {
		return Verification{}, err
	}
	var verified []Verification
	for _, edge := range edges {
		if edge.Child != gate.ID || edge.Relation != artifact.RelationProducedBy || edge.Parent.Kind() != artifact.KindRun {
			continue
		}
		content, found, readErr := artifact.ReadContent(ctx, store, edge.Parent)
		if readErr != nil {
			return Verification{}, readErr
		}
		if !found || content.Descriptor.MediaType != RunMediaType ||
			content.Descriptor.Schema != RunSchema && content.Descriptor.Schema != LegacyRunSchema {
			continue
		}
		run, parseErr := ParseRun(content.Data)
		if parseErr != nil || run.ID != edge.Parent {
			if parseErr != nil {
				return Verification{}, parseErr
			}
			return Verification{}, errors.New("run record: run lineage identity mismatch")
		}
		candidate, verifyErr := VerifyGateRun(ctx, store, gate.Recipe, gate.ID, run.ID)
		if verifyErr != nil {
			return Verification{}, fmt.Errorf("run record: typed run lineage contradicts its gate: %w", verifyErr)
		}
		verified = append(verified, candidate)
	}
	if len(verified) != 1 {
		return Verification{}, fmt.Errorf("run record: gate requires exactly one bound successful run, found %d", len(verified))
	}
	return verified[0], nil
}

func uniqueGateFinalization(
	ctx context.Context,
	store *overgodb.Store,
	gate GateResult,
) (GateLifecycle, GateLifecycle, error) {
	edges, err := store.Children(ctx, gate.ID)
	if err != nil {
		return GateLifecycle{}, GateLifecycle{}, err
	}
	var finalizations []GateLifecycle
	for _, edge := range edges {
		if edge.Parent != gate.ID || edge.Relation != artifact.RelationDependsOn {
			continue
		}
		lifecycle, typed, candidateErr := readGateLifecycleCandidate(ctx, store, edge.Child)
		if candidateErr != nil {
			return GateLifecycle{}, GateLifecycle{}, candidateErr
		}
		if !typed {
			continue
		}
		if lifecycle.State != GateFinalized || lifecycle.Result == nil || *lifecycle.Result != gate.ID {
			return GateLifecycle{}, GateLifecycle{}, errors.New("run record: gate lifecycle lineage contradicts its result")
		}
		finalizations = append(finalizations, lifecycle)
	}
	finalizations = canonicalLifecycles(finalizations)
	if len(finalizations) != 1 {
		return GateLifecycle{}, GateLifecycle{}, fmt.Errorf(
			"run record: gate requires exactly one typed finalization, found %d", len(finalizations),
		)
	}
	finalization := finalizations[0]
	if finalization.Outcome != OutcomeSucceeded || finalization.CodeCommit != gate.CodeCommit ||
		finalization.Environment != gate.Environment || finalization.Preparation == nil {
		return GateLifecycle{}, GateLifecycle{}, errors.New("run record: gate finalization does not match the successful result")
	}
	preparation, err := RequireGateLifecycle(ctx, store, *finalization.Preparation)
	if err != nil {
		return GateLifecycle{}, GateLifecycle{}, err
	}
	if preparation.State != GatePrepared || finalization.TreeKey != preparation.TreeKey ||
		finalization.Environment != preparation.Environment || finalization.Started != preparation.Started {
		return GateLifecycle{}, GateLifecycle{}, errors.New("run record: gate finalization contradicts its preparation")
	}
	return finalization, preparation, nil
}

// AttemptsForRecipe follows direct recipe lineage and returns exact typed
// attempt candidates in deterministic identity order.
func AttemptsForRecipe(
	ctx context.Context,
	store *overgodb.Store,
	recipeID artifact.ID,
) ([]AttemptRecord, error) {
	if ctx == nil || store == nil || recipeID.Kind() != artifact.KindRecipe {
		return nil, errors.New("run record: attempts for recipe require the store and a recipe")
	}
	edges, err := store.Children(ctx, recipeID)
	if err != nil {
		return nil, err
	}
	return attemptsFromEdges(ctx, store, recipeID, false, edges)
}

// AttemptsForPreparation follows preparation -> finalization -> gate-result
// lineage and returns exact typed attempt candidates in deterministic order.
func AttemptsForPreparation(
	ctx context.Context,
	store *overgodb.Store,
	preparationID artifact.ID,
) ([]AttemptRecord, error) {
	finalization, found, err := GateFinalizationForPreparation(ctx, store, preparationID)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, nil
	}
	if finalization.Result == nil {
		return nil, errors.New("run record: finalization lacks its gate result")
	}
	edges, err := store.Children(ctx, *finalization.Result)
	if err != nil {
		return nil, err
	}
	attempts, err := attemptsFromEdges(ctx, store, *finalization.Result, true, edges)
	if err != nil {
		return nil, err
	}
	byID := make(map[artifact.ID]AttemptRecord)
	for _, attempt := range attempts {
		byID[attempt.ID] = attempt
	}
	return sortedAttempts(byID), nil
}

func gateFinalizationsForPreparation(
	ctx context.Context,
	store *overgodb.Store,
	preparationID artifact.ID,
) ([]GateLifecycle, error) {
	if ctx == nil || store == nil {
		return nil, errors.New("run record: gate finalizations require the store")
	}
	preparation, err := RequireGateLifecycle(ctx, store, preparationID)
	if err != nil {
		return nil, err
	}
	if preparation.State != GatePrepared {
		return nil, errors.New("run record: gate finalizations require a preparation")
	}
	edges, err := store.Children(ctx, preparationID)
	if err != nil {
		return nil, err
	}
	var finalizations []GateLifecycle
	for _, edge := range edges {
		if edge.Parent != preparationID || edge.Relation != artifact.RelationDerivedFrom {
			continue
		}
		lifecycle, typed, candidateErr := readGateLifecycleCandidate(ctx, store, edge.Child)
		if candidateErr != nil {
			return nil, candidateErr
		}
		if !typed {
			continue
		}
		if lifecycle.State != GateFinalized || lifecycle.Preparation == nil || *lifecycle.Preparation != preparationID ||
			lifecycle.TreeKey != preparation.TreeKey || lifecycle.Environment != preparation.Environment ||
			lifecycle.Started != preparation.Started {
			return nil, errors.New("run record: finalization lineage contradicts its preparation")
		}
		finalizations = append(finalizations, lifecycle)
	}
	return canonicalLifecycles(finalizations), nil
}

// GateFinalizationForPreparation returns the sole canonical finalization for a
// preparation. Absence is explicit; ambiguous append-only authority is always
// an error so callers cannot accidentally select one by iteration order.
func GateFinalizationForPreparation(
	ctx context.Context,
	store *overgodb.Store,
	preparationID artifact.ID,
) (GateLifecycle, bool, error) {
	finalizations, err := gateFinalizationsForPreparation(ctx, store, preparationID)
	if err != nil {
		return GateLifecycle{}, false, err
	}
	var finalization GateLifecycle
	found := false
	for _, candidate := range finalizations {
		if found {
			return GateLifecycle{}, false, fmt.Errorf(
				"run record: preparation %s has %d canonical finalizations",
				preparationID, len(finalizations),
			)
		}
		finalization, found = candidate, true
	}
	return finalization, found, nil
}

func attemptsFromEdges(
	ctx context.Context,
	store *overgodb.Store,
	parent artifact.ID,
	resultParent bool,
	edges []artifact.Lineage,
) ([]AttemptRecord, error) {
	byID := make(map[artifact.ID]AttemptRecord)
	for _, edge := range edges {
		if edge.Parent != parent || edge.Relation != artifact.RelationDependsOn {
			continue
		}
		attempt, typed, err := readAttemptCandidate(ctx, store, edge.Child)
		if err != nil {
			return nil, err
		}
		if !typed {
			continue
		}
		matches := attempt.Recipe == parent
		if resultParent {
			matches = attempt.Result == parent
		}
		if !matches {
			return nil, errors.New("run record: attempt lineage contradicts its authority")
		}
		byID[attempt.ID] = attempt
	}
	return sortedAttempts(byID), nil
}

func readAttemptCandidate(
	ctx context.Context,
	store *overgodb.Store,
	id artifact.ID,
) (AttemptRecord, bool, error) {
	content, found, err := artifact.ReadContent(ctx, store, id)
	if err != nil || !found {
		return AttemptRecord{}, false, err
	}
	if content.Descriptor.ID.Kind() != attemptContract.Kind || content.Descriptor.MediaType != attemptContract.MediaType ||
		content.Descriptor.Schema != attemptContract.Schema {
		return AttemptRecord{}, false, nil
	}
	attempt, err := attemptCodec.Parse(content.Data)
	if err != nil {
		return AttemptRecord{}, false, err
	}
	if attempt.ID != id {
		return AttemptRecord{}, false, errors.New("run record: attempt candidate identity mismatch")
	}
	return attempt, true, nil
}

func readGateLifecycleCandidate(
	ctx context.Context,
	store *overgodb.Store,
	id artifact.ID,
) (GateLifecycle, bool, error) {
	content, found, err := artifact.ReadContent(ctx, store, id)
	if err != nil || !found {
		return GateLifecycle{}, false, err
	}
	if content.Descriptor.ID.Kind() != artifact.KindEvidence || content.Descriptor.MediaType != GateLifecycleMediaType ||
		content.Descriptor.Schema != GateLifecycleSchema {
		return GateLifecycle{}, false, nil
	}
	lifecycle, err := ParseGateLifecycle(content.Data)
	if err != nil {
		return GateLifecycle{}, false, err
	}
	if lifecycle.ID != id {
		return GateLifecycle{}, false, errors.New("run record: gate lifecycle candidate identity mismatch")
	}
	return lifecycle, true, nil
}

func sortedAttempts(byID map[artifact.ID]AttemptRecord) []AttemptRecord {
	result := slices.Collect(maps.Values(byID))
	slices.SortFunc(result, func(left, right AttemptRecord) int {
		return artifact.CompareID(left.ID, right.ID)
	})
	return result
}

func canonicalLifecycles(values []GateLifecycle) []GateLifecycle {
	slices.SortFunc(values, func(left, right GateLifecycle) int {
		return artifact.CompareID(left.ID, right.ID)
	})
	result := values[:0]
	for _, value := range values {
		if len(result) == 0 || result[len(result)-1].ID != value.ID {
			result = append(result, value)
		}
	}
	return result
}
