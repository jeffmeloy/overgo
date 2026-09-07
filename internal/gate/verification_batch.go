package gate

import (
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"time"

	"overgo/internal/automationcheck"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
)

func planVerificationBatch(repo, reference string) (*plan.VerificationBatch, error) {
	// Inspection without a dispatched row has no subordinate acceptance.
	if reference == "" {
		return nil, nil
	}
	document, err := plan.Load(filepath.Join(repo, plan.Path))
	if err != nil {
		return nil, err
	}
	if err := plan.Validate(document); err != nil {
		return nil, err
	}
	for _, item := range document.Items {
		for _, step := range item.Steps {
			if item.ID+"/"+step.ID == reference {
				return step.VerificationBatch, nil
			}
		}
	}
	return nil, errors.New("acceptance: dispatched row is absent from the candidate plan")
}

func (g *gateContext) batchAcceptanceChecks(checks []automationcheck.Check, batch *plan.VerificationBatch) ([]automationcheck.Check, error) {
	if batch == nil {
		return checks, nil
	}
	index := slices.IndexFunc(checks, func(check automationcheck.Check) bool { return check.Descriptor.Name == "acceptance" })
	if index < 0 {
		return nil, errors.New("acceptance: batch requires the parent integration check")
	}
	parent := &checks[index].Descriptor
	// Declared flush bounds: accepted checkpoints accumulate per key; the
	// decision and its reason join the audit so gate cost per flushed batch is
	// attributable. Absent declaration -> no accumulation.
	var accumulator *plan.BatchAccumulator
	if batch.Flush != nil {
		var err error
		if accumulator, err = plan.NewBatchAccumulator(*batch.Flush); err != nil {
			return nil, fmt.Errorf("acceptance: %w", err)
		}
	}
	// Checkpoint memo slots: computed once from the declared verifies so the
	// verification loop can reuse accepted checkpoint evidence across runs.
	memos, err := g.checkpointMemoInputs(batch)
	if err != nil {
		return nil, fmt.Errorf("acceptance: %w", err)
	}
	g.checkpointMemos = memos
	g.audit = append(g.audit, fmt.Sprintf("checkpoint memo: key=%s checkpoints=%d", checkpointMemoKey(g.planRef, batch), len(memos)))
	var added []automationcheck.Check
	for _, checkpoint := range batch.Checkpoints {
		evidence, err := runrecord.FormatCompletionAcceptanceEvidence(
			testevidence.CurrentVerifyPolicy, g.planRef, checkpoint.Verify,
		)
		if err != nil {
			return nil, err
		}
		// Bind the declared verifier even when execution reuses a checkpoint.
		// This is a contract, not a passing verdict: terminal evidence must
		// independently satisfy the manifest's commit barrier.
		g.stepEvidence[checkpoint.GateCheckName()] = evidence
		check := gateCheck(checkpoint.GateCheckName(), runrecord.PhaseTest, func() (bool, error) {
			if err := g.verifyAcceptedCandidate(checkpoint.Verify, false); err != nil {
				return false, fmt.Errorf("checkpoint %s: %w", checkpoint.ID, err)
			}
			if accumulator != nil {
				state, flush, reason, err := accumulator.Add(batch.Flush.Key, int64(len(evidence)), time.Now())
				if err != nil {
					return false, fmt.Errorf("checkpoint %s: %w", checkpoint.ID, err)
				}
				g.audit = append(g.audit, fmt.Sprintf("batch flush: checkpoint=%s key=%s size=%d bytes=%d flush=%t reason=%q",
					checkpoint.ID, state.Key, state.Size, state.Bytes, flush, reason))
			}
			return false, nil
		})
		// Run subordinate verifiers serially: each owns the temporary worktree
		// and gate evidence maps. Their declared graph remains the batch contract.
		check.Descriptor.Dependencies = slices.Clone(parent.Dependencies)
		parent.Dependencies = []string{checkpoint.GateCheckName()}
		added = append(added, check)
	}
	return slices.Insert(checks, index, added...), nil
}
