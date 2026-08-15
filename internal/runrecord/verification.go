package runrecord

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
)

// Verification: completed gate and its bound run.
type Verification struct {
	Gate GateResult
	Run  Run
}

// VerifyGateRun: load and bind a successful verifier gate/run pair.
func VerifyGateRun(
	ctx context.Context,
	store artifact.Reader,
	recipeID, gateID, runID artifact.ID,
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
	if gate.Outcome != OutcomeSucceeded || run.Outcome != OutcomeSucceeded {
		return Verification{}, errors.New("run record: verifier did not succeed")
	}
	if gate.Environment != run.Environment || gate.CodeCommit != run.CodeCommit ||
		!slices.Contains(run.Outputs, gate.ID) {
		return Verification{}, errors.New("run record: gate and run are not bound")
	}
	return Verification{Gate: gate, Run: run}, nil
}

// VerifyEvidence: resolve one successful gate/run pair from lifecycle evidence.
func VerifyEvidence(
	ctx context.Context,
	store artifact.Reader,
	recipeID artifact.ID,
	evidence []artifact.ID,
) (Verification, error) {
	var gates, runs []artifact.ID
	for _, id := range evidence {
		content, ok, err := store.Content(ctx, id)
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
