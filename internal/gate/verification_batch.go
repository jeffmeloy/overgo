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
	g.verificationBatch = batch
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
	var pending plan.BatchState
	// Checkpoint memo slots: computed once from the declared verifies so the
	// verification loop can reuse accepted checkpoint evidence across runs.
	memos, err := g.checkpointMemoInputs(batch)
	if err != nil {
		return nil, fmt.Errorf("acceptance: %w", err)
	}
	g.checkpointMemos = memos
	g.note(fmt.Sprintf("checkpoint memo: key=%s checkpoints=%d", checkpointMemoKey(g.planRef, batch), len(memos)))
	required, err := focusedCheckpoints(batch, g.checkpoint)
	if err != nil {
		return nil, err
	}
	base := slices.Clone(parent.Dependencies)
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
			if batch.Flush != nil {
				bytes := int64(len(evidence))
				state, flush, reason, err := batch.Flush.Advance(pending, &bytes, time.Now(), false)
				if err != nil {
					return false, fmt.Errorf("checkpoint %s: %w", checkpoint.ID, err)
				}
				g.note(fmt.Sprintf("batch flush: checkpoint=%s key=%s size=%d bytes=%d flush=%t reason=%q",
					checkpoint.ID, state.Key, state.Size, state.Bytes, flush, reason))
				pending = state
				if flush {
					pending = plan.BatchState{}
				}
			}
			return false, nil
		})
		// Run subordinate verifiers serially: each owns the temporary worktree
		// and gate evidence maps. Their declared graph remains the batch contract.
		check.Descriptor.Dependencies = slices.Clone(parent.Dependencies)
		if required[checkpoint.ID] {
			// A focused publication runs the checkpoint after its declared
			// predecessors alone; the serial chain through the deferred
			// siblings would otherwise hold it.
			check.Descriptor.Dependencies = slices.Clone(base)
			for _, predecessor := range checkpoint.DependsOn {
				check.Descriptor.Dependencies = append(check.Descriptor.Dependencies, plan.VerificationCheckpoint{ID: predecessor}.GateCheckName())
			}
		}
		parent.Dependencies = []string{checkpoint.GateCheckName()}
		added = append(added, check)
	}
	return slices.Insert(checks, index, added...), nil
}

// focusedCheckpoints names the checkpoint a publication runs and its
// declared predecessors, transitively; an empty focus names none.
func focusedCheckpoints(batch *plan.VerificationBatch, focus string) (map[string]bool, error) {
	required := map[string]bool{}
	if focus == "" {
		return required, nil
	}
	if batch == nil {
		return nil, fmt.Errorf("checkpoint %s: the dispatched step declares no verification batch", focus)
	}
	byID := make(map[string]plan.VerificationCheckpoint, len(batch.Checkpoints))
	for _, checkpoint := range batch.Checkpoints {
		byID[checkpoint.ID] = checkpoint
	}
	if _, found := byID[focus]; !found {
		return nil, fmt.Errorf("checkpoint %s: not declared by the dispatched step's verification batch", focus)
	}
	queue := []string{focus}
	for len(queue) != 0 {
		id := queue[0]
		queue = queue[1:]
		if required[id] {
			continue
		}
		required[id] = true
		queue = append(queue, byID[id].DependsOn...)
	}
	return required, nil
}

// checkpointDeferrals lists the checks a focused publication leaves to the
// step's cumulative gate: the step acceptance, the dependent suites, the
// lanes, the commit and every checkpoint the focus does not require. None
// of them is marked satisfied; the batch ledger keeps them outstanding.
func (g *gateContext) checkpointDeferrals(definitions []automationcheck.Check) (map[string]bool, error) {
	if g.checkpoint == "" {
		return nil, nil
	}
	required, err := focusedCheckpoints(g.verificationBatch, g.checkpoint)
	if err != nil {
		return nil, err
	}
	cumulative := map[string]bool{
		"acceptance": true, testRestCheckName: true, testDeviceCheckName: true, "device": true, automationcheck.WebUICheckName: true, "commit": true,
	}
	for _, checkpoint := range g.verificationBatch.Checkpoints {
		if !required[checkpoint.ID] {
			cumulative[checkpoint.GateCheckName()] = true
		}
	}
	deferred := map[string]bool{}
	for _, definition := range definitions {
		if cumulative[definition.Descriptor.Name] {
			deferred[definition.Descriptor.Name] = true
		}
	}
	return deferred, nil
}
