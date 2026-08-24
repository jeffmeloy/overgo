package runrecord

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
)

// Verification defines completed gate and its bound run.
type Verification struct {
	Gate GateResult
	Run  Run
}

// VerifyGateRun loads and binds a successful verifier gate and run pair.
func VerifyGateRun(
	ctx context.Context,
	store artifact.Reader,
	recipeID, gateID, runID artifact.ID,
) (Verification, error) {
	return verifyGateRunOutcome(ctx, store, recipeID, gateID, runID, OutcomeSucceeded)
}

// VerifyFailedGateRun loads and binds a failed verifier gate and run pair.
func VerifyFailedGateRun(
	ctx context.Context,
	store artifact.Reader,
	recipeID, gateID, runID artifact.ID,
) (Verification, error) {
	return verifyGateRunOutcome(ctx, store, recipeID, gateID, runID, OutcomeFailed)
}

func verifyGateRunOutcome(
	ctx context.Context,
	store artifact.Reader,
	recipeID, gateID, runID artifact.ID,
	want Outcome,
) (Verification, error) {
	gate, err := gateCodec.Require(ctx, store, gateID)
	if err != nil {
		return Verification{}, err
	}
	run, err := runCodec.Require(ctx, store, runID)
	if err != nil {
		return Verification{}, err
	}
	if gate.ID != gateID || run.ID != runID || gate.Recipe != recipeID || run.Recipe != recipeID {
		return Verification{}, errors.New("run record: verifier identity mismatch")
	}
	if gate.Outcome != want || run.Outcome != want {
		return Verification{}, errors.New("run record: verifier outcome mismatch")
	}
	if gate.Environment != run.Environment || gate.CodeCommit != run.CodeCommit ||
		!slices.Contains(run.Outputs, gate.ID) {
		return Verification{}, errors.New("run record: gate and run are not bound")
	}
	return Verification{Gate: gate, Run: run}, nil
}

// VerifyEvidence resolves one successful gate and run pair from lifecycle evidence.
func VerifyEvidence(
	ctx context.Context,
	store artifact.Reader,
	recipeID artifact.ID,
	evidence []artifact.ID,
) (Verification, error) {
	var gates, runs []artifact.ID
	for _, id := range evidence {
		content, ok, err := artifact.ReadContent(ctx, store, id)
		if err != nil {
			return Verification{}, err
		}
		if !ok {
			continue
		}
		descriptor := content.Descriptor
		switch {
		case descriptor.ID.Kind() == runContract.Kind &&
			descriptor.MediaType == runContract.MediaType && descriptor.Schema == runContract.Schema:
			runs = append(runs, id)
		case descriptor.ID.Kind() == gateContract.Kind &&
			descriptor.MediaType == gateContract.MediaType && descriptor.Schema == gateContract.Schema:
			gates = append(gates, id)
		}
	}
	for _, gate := range gates {
		for _, run := range runs {
			if verified, err := VerifyGateRun(ctx, store, recipeID, gate, run); err == nil {
				return verified, nil
			}
		}
	}
	return Verification{}, errors.New("run record: successful verifier gate/run evidence is absent")
}
