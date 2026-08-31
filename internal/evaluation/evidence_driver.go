package evaluation

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/composition"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// EvidenceDriverCandidate binds one immutable candidate to the exact admission
// decision that made it eligible for driver consideration.
type EvidenceDriverCandidate struct {
	Candidate artifact.ID `json:"candidate"`
	Admission artifact.ID `json:"admission"`
}

// EvidenceDriverRequest is the wire-safe input to one evidence-derived driver
// decision. It carries identities only; every typed owner is reloaded below.
type EvidenceDriverRequest struct {
	Goal              artifact.ID               `json:"goal"`
	Causal            runrecord.CausalContext   `json:"causal"`
	Head              artifact.CommitID         `json:"head"`
	Incumbent         artifact.ID               `json:"incumbent"`
	Candidates        []EvidenceDriverCandidate `json:"candidates,omitempty"`
	InteractionBudget artifact.ID               `json:"interaction_budget"`
	ResourceBudget    artifact.ID               `json:"resource_budget"`
}

// CompileEvidenceDriverDecision replays the current incumbent and every
// candidate admission, then delegates all selection, budget, and stop
// derivation to runrecord. It publishes and executes nothing.
func CompileEvidenceDriverDecision(
	ctx context.Context,
	reader artifact.Reader,
	request EvidenceDriverRequest,
) (runrecord.DriverDecision, error) {
	if ctx == nil || reader == nil {
		return runrecord.DriverDecision{}, errors.New("evaluation: evidence driver authority is absent")
	}
	if err := ctx.Err(); err != nil {
		return runrecord.DriverDecision{}, err
	}
	if !request.Head.Valid() || request.Causal.Validate() != nil ||
		len(request.Candidates) >= runrecord.MaximumAttemptPopulation {
		return runrecord.DriverDecision{}, errors.New("evaluation: invalid evidence driver request")
	}
	incumbent, err := requireEvidenceDriverIncumbent(ctx, reader, request.Incumbent, request.Goal)
	if err != nil {
		return runrecord.DriverDecision{}, err
	}
	options := make([]runrecord.DriverOption, 1, len(request.Candidates)+1)
	options[0] = incumbent
	seenCandidates := make(map[artifact.ID]struct{}, len(request.Candidates))
	seenAdmissions := make(map[artifact.ID]struct{}, len(request.Candidates))
	adapters := append(modelrecipe.CandidateAdmissionAdapters(), composition.CandidateAdmissionAdapter())
	for _, reference := range request.Candidates {
		if reference.Candidate.Kind() != artifact.KindRecipe || reference.Admission.Kind() != artifact.KindEvidence {
			return runrecord.DriverDecision{}, errors.New("evaluation: invalid evidence driver candidate authority")
		}
		_, duplicateCandidate := seenCandidates[reference.Candidate]
		_, duplicateAdmission := seenAdmissions[reference.Admission]
		if duplicateCandidate || duplicateAdmission {
			return runrecord.DriverDecision{}, errors.New("evaluation: duplicate evidence driver candidate authority")
		}
		seenCandidates[reference.Candidate] = struct{}{}
		seenAdmissions[reference.Admission] = struct{}{}

		candidate, err := modelrecipe.RequireCandidate(ctx, reader, reference.Candidate)
		if err != nil {
			return runrecord.DriverDecision{}, err
		}
		admission, err := runrecord.RequireReplayedCandidateAdmission(
			ctx, reader, reference.Admission, candidate, adapters...,
		)
		if err != nil || admission.Candidate != candidate.ID() {
			return runrecord.DriverDecision{}, errors.Join(
				errors.New("evaluation: evidence driver candidate admission is foreign or stale"), err,
			)
		}
		if _, err := modelrecipe.RequireCandidateEvaluationIntent(ctx, reader, candidate); err != nil {
			return runrecord.DriverDecision{}, errors.Join(
				errors.New("evaluation: admitted candidate evaluation intent does not replay"), err,
			)
		}
		if err := requireEvidenceDriverCandidateGoal(ctx, reader, request.Goal, candidate); err != nil {
			return runrecord.DriverDecision{}, err
		}
		spec := candidate.Spec()
		options = append(options, runrecord.DriverOption{
			Candidate: candidate.ID(), Admission: admission.ID,
			State: runrecord.DriverCandidateEligible,
			Missing: []runrecord.DriverEvidenceGap{{
				Need: runrecord.DriverNeedRealization, Target: candidate.ID(),
				CostUnits: spec.Prediction.Cost, CostUnit: spec.CostUnit, CostAuthority: candidate.ID(),
			}},
		})
	}

	return runrecord.NewDriverDecision(ctx, reader, runrecord.DriverDecisionFacts{
		Goal: request.Goal, Causal: request.Causal, Head: request.Head, Options: options,
		InteractionBudget: runrecord.DriverBudgetReference{Grant: request.InteractionBudget},
		ResourceBudget:    runrecord.DriverBudgetReference{Grant: request.ResourceBudget},
	})
}

func requireEvidenceDriverCandidateGoal(
	ctx context.Context,
	reader artifact.Reader,
	goal artifact.ID,
	candidate modelrecipe.Candidate,
) error {
	spec := candidate.Spec()
	for _, reference := range spec.References {
		if reference.Role != modelrecipe.CandidateReferenceBaseline || reference.Subject != spec.Parent {
			continue
		}
		decision, err := recipe.RequireDecision(ctx, reader, reference.Evidence)
		if err != nil {
			return err
		}
		if slices.Contains(decision.Evidence, goal) {
			return nil
		}
	}
	return errors.New("evaluation: admitted candidate baseline is unrelated to the evidence driver goal")
}

func requireEvidenceDriverIncumbent(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
	goal artifact.ID,
) (runrecord.DriverOption, error) {
	if id.Kind() != artifact.KindEvidence {
		return runrecord.DriverOption{}, errors.New("evaluation: invalid evidence driver incumbent authority")
	}
	probe, err := RequireCapabilityProbeResult(ctx, reader, id)
	if err != nil {
		return runrecord.DriverOption{}, err
	}
	if probe.Outcome != runrecord.OutcomeSucceeded ||
		(probe.Entry != ProductionEntryLocalPlacement && probe.Entry != ProductionEntryPeerPlacement) {
		return runrecord.DriverOption{}, errors.New("evaluation: evidence driver incumbent is not a successful model placement")
	}
	if goal != probe.CheckDecision {
		return runrecord.DriverOption{}, errors.New("evaluation: evidence driver goal differs from the incumbent production check")
	}
	testCase, err := activationCaseCodec.Require(ctx, reader, probe.Case)
	if err != nil {
		return runrecord.DriverOption{}, err
	}
	activation, active, err := modelrecipe.ActiveRecord(ctx, reader, probe.Model, testCase.Task)
	if err != nil || !active {
		return runrecord.DriverOption{}, errors.Join(errors.New("evaluation: evidence driver incumbent is no longer active"), err)
	}
	selection, err := modelrecipe.ResolveCapabilityEvidenceSelector(ctx, reader, modelrecipe.CapabilityEvidenceSelector{
		Alias: probe.Alias, Task: testCase.Task, Session: probe.Session, Compatibility: probe.Compatibility,
	})
	if err != nil {
		return runrecord.DriverOption{}, err
	}
	if activation.Definition.ID != probe.Recipe || activation.Definition.Model != probe.Model ||
		selection.Identity != probe.Selection || selection.Activation.Definition.ID != activation.Definition.ID ||
		selection.Activation.Event.ID != activation.Event.ID {
		return runrecord.DriverOption{}, errors.New("evaluation: evidence driver incumbent recipe or lifecycle is stale")
	}
	placement, err := capabilityruntime.ResolveExactCapabilityPlacement(ctx, reader, probe.ModelCapability, selection)
	if err != nil {
		return runrecord.DriverOption{}, err
	}
	if placement.ID != probe.Placement {
		return runrecord.DriverOption{}, errors.New("evaluation: evidence driver incumbent placement is stale")
	}
	return runrecord.DriverOption{
		Recipe: activation.Definition.ID, Lifecycle: activation.Event.ID,
		State: runrecord.DriverIncumbentActive, Placement: placement.ID,
		Evidence: []artifact.ID{probe.ID},
	}, nil
}
