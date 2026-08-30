package evaluation

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/executionfailure"
	"overgo/internal/runrecord"
)

const (
	trajectoryHealthDecisionMediaType = "application/vnd.overgo.trajectory-health-decision+json"
	trajectoryHealthDecisionSchema    = "overgo/trajectory-health-decision/v1"

	trajectoryRuleExactNoProgress = "exact-no-progress"
	trajectoryRuleNewEvidence     = "new-evidence"
	trajectoryRuleChangedAction   = "changed-action"
)

// TrajectoryHealthDecision is a durable supervision result over two exact
// interaction traces and the current stage receipts that close their tool
// results. Lifecycle owners decide what to execute from its disposition.
type TrajectoryHealthDecision struct {
	Version         uint16                    `json:"version"`
	Previous        artifact.ID               `json:"previous"`
	Current         artifact.ID               `json:"current"`
	PreviousReceipt artifact.ID               `json:"previous_receipt"`
	CurrentReceipt  artifact.ID               `json:"current_receipt"`
	Progress        []artifact.ID             `json:"progress,omitempty"`
	Decision        executionfailure.Decision `json:"decision"`
	Rule            string                    `json:"rule"`
	ID              artifact.ID               `json:"-"`
}

var trajectoryHealthDecisionCodec = artifact.JSONDocumentCodec(
	"trajectory health decision", artifact.KindEvidence,
	trajectoryHealthDecisionMediaType, trajectoryHealthDecisionSchema,
	canonicalizeTrajectoryHealthDecision,
	func(value TrajectoryHealthDecision) artifact.ID { return value.ID },
	func(value *TrajectoryHealthDecision, id artifact.ID) { value.ID = id },
	func(value TrajectoryHealthDecision) TrajectoryHealthDecision {
		value.Progress = slices.Clone(value.Progress)
		return value
	},
)

type trajectoryExchange struct {
	Trace     runrecord.InteractionTrace
	Receipt   runrecord.StageReceipt
	Manual    artifact.ID
	Arguments artifact.ID
	State     artifact.ID
	Result    artifact.ID
}

// SuperviseTrajectory derives both observations from stored traces and their
// exact current receipt closure, then commits one evidence decision at the
// verified store head. Callers cannot inject progress identities.
func SuperviseTrajectory(
	ctx context.Context,
	repository artifact.Repository,
	previousID, currentID artifact.ID,
) (TrajectoryHealthDecision, artifact.CommitID, error) {
	if ctx == nil || repository == nil {
		return TrajectoryHealthDecision{}, artifact.CommitID{}, errors.New("evaluation: trajectory supervision authority is absent")
	}
	head, _ := repository.Head()
	previous, err := requireTrajectoryExchange(ctx, repository, previousID)
	if err != nil {
		return TrajectoryHealthDecision{}, artifact.CommitID{}, err
	}
	current, err := requireTrajectoryExchange(ctx, repository, currentID)
	if err != nil {
		return TrajectoryHealthDecision{}, artifact.CommitID{}, err
	}
	if previous.Trace.ID == current.Trace.ID || previous.Receipt.ID == current.Receipt.ID ||
		previous.Trace.TaskContract != current.Trace.TaskContract || previous.Trace.Model != current.Trace.Model ||
		previous.Trace.Recipe != current.Trace.Recipe {
		return TrajectoryHealthDecision{}, artifact.CommitID{}, errors.New("evaluation: supervised trajectories address different sessions")
	}
	progress, err := requireTrajectoryProgress(ctx, repository, previous, current)
	if err != nil {
		return TrajectoryHealthDecision{}, artifact.CommitID{}, err
	}
	decision, rule := executionfailure.DecisionResume, trajectoryRuleChangedAction
	sameAction := previous.Manual == current.Manual && previous.Arguments == current.Arguments &&
		previous.State == current.State && previous.Result == current.Result
	if len(progress) != 0 {
		rule = trajectoryRuleNewEvidence
	} else if sameAction {
		decision, rule = executionfailure.DecisionRetireSession, trajectoryRuleExactNoProgress
	}
	value, err := trajectoryHealthDecisionCodec.New(TrajectoryHealthDecision{
		Version:  artifact.InitialDocumentVersion,
		Previous: previous.Trace.ID, Current: current.Trace.ID,
		PreviousReceipt: previous.Receipt.ID, CurrentReceipt: current.Receipt.ID,
		Progress: progress, Decision: decision, Rule: rule,
	})
	if err != nil {
		return TrajectoryHealthDecision{}, artifact.CommitID{}, err
	}
	content, err := trajectoryHealthDecisionCodec.Content(value)
	if err != nil {
		return TrajectoryHealthDecision{}, artifact.CommitID{}, err
	}
	batch, err := artifact.NewDocumentBatch(
		"evaluation/trajectory-health/"+value.ID.String(), []artifact.Content{content}, value.Lineage(), nil,
	)
	if err != nil {
		return TrajectoryHealthDecision{}, artifact.CommitID{}, err
	}
	batch.ExpectedHead = &head
	commit, err := artifact.CommitBatch(ctx, repository, batch)
	return value, commit, err
}

// RequireTrajectoryHealthDecision loads the codec-backed immutable result and
// refuses a body whose stored dependency closure differs from its references.
func RequireTrajectoryHealthDecision(ctx context.Context, reader artifact.Reader, id artifact.ID) (TrajectoryHealthDecision, error) {
	return trajectoryHealthDecisionCodec.RequireExactLineage(ctx, reader, id, TrajectoryHealthDecision.Lineage)
}

// Lineage closes the decision over traces, receipts, and typed progress.
func (value TrajectoryHealthDecision) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.Previous, value.Current, value.PreviousReceipt, value.CurrentReceipt}
	parents = append(parents, value.Progress...)
	return artifact.DependencyLineage(value.ID, parents...)
}

func requireTrajectoryExchange(ctx context.Context, reader artifact.Reader, id artifact.ID) (trajectoryExchange, error) {
	trace, err := runrecord.RequireInteractionTrace(ctx, reader, id)
	if err != nil || trace.TaskContract.Kind() != artifact.KindRecipe || trace.Terminal == "" {
		return trajectoryExchange{}, errors.Join(errors.New("evaluation: trajectory is not a complete agent trace"), err)
	}
	if _, err := runrecord.RequireInteractionTranscript(ctx, reader, trace.Request); err != nil {
		return trajectoryExchange{}, errors.Join(errors.New("evaluation: trajectory request state is absent"), err)
	}
	calls, failures, err := trace.ToolExchanges()
	if err != nil || failures != 0 || len(calls) != 1 {
		return trajectoryExchange{}, errors.Join(errors.New("evaluation: trajectory lacks one successful exact tool exchange"), err)
	}
	call := calls[0]
	manual, err := agenttool.RequireManual(ctx, reader, call.Manual)
	if err != nil || manual.Name != call.Name || !slices.Contains(trace.ToolManuals, manual.ID) {
		return trajectoryExchange{}, errors.Join(errors.New("evaluation: trajectory manual differs"), err)
	}
	arguments, err := runrecord.AttemptArgumentContent([]byte(call.Arguments))
	if err != nil {
		return trajectoryExchange{}, err
	}
	if content, found, readErr := artifact.ReadContent(ctx, reader, arguments.Descriptor.ID); readErr != nil || !found ||
		content.Descriptor != arguments.Descriptor || !slices.Equal(content.Data, arguments.Data) {
		return trajectoryExchange{}, errors.Join(errors.New("evaluation: trajectory arguments are not stored exactly"), readErr)
	}
	result, err := trajectoryToolResult(trace, call.ID)
	if err != nil {
		return trajectoryExchange{}, err
	}
	receipt, err := requireTrajectoryStageReceipt(ctx, reader, trace, manual.ID, arguments.Descriptor.ID, result)
	if err != nil {
		return trajectoryExchange{}, err
	}
	return trajectoryExchange{
		Trace: trace, Receipt: receipt, Manual: manual.ID, Arguments: arguments.Descriptor.ID,
		State: trace.Request, Result: result,
	}, nil
}

func trajectoryToolResult(trace runrecord.InteractionTrace, callID string) (artifact.ID, error) {
	for _, event := range trace.Events {
		if event.Kind == runrecord.InteractionEventToolResult && event.Message.ToolCallID == callID {
			id, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte(event.Message.Content))
			if err != nil {
				return artifact.ID{}, err
			}
			return id, nil
		}
	}
	return artifact.ID{}, errors.New("evaluation: trajectory tool result is absent")
}

func requireTrajectoryStageReceipt(
	ctx context.Context,
	reader artifact.Reader,
	trace runrecord.InteractionTrace,
	manual, arguments, result artifact.ID,
) (runrecord.StageReceipt, error) {
	var matched runrecord.StageReceipt
	for _, id := range trace.FinalArtifacts {
		content, found, err := artifact.ReadContent(ctx, reader, id)
		if err != nil || !found {
			continue
		}
		receipt, parseErr := runrecord.ParseStageReceipt(content.Data)
		if parseErr != nil || receipt.ID != id {
			continue
		}
		current, requireErr := requireCurrentStageReceipt(ctx, reader, receipt.ID)
		if requireErr != nil || current.Operation != trace.Operation || current.Recipe != trace.Recipe ||
			current.State != runrecord.StageCompleted || !stageBindingsContain(current.Inputs, manual) ||
			!stageBindingsContain(current.Inputs, arguments) || !stageBindingsContain(current.Outputs, result) {
			continue
		}
		if matched.ID.Valid() {
			return runrecord.StageReceipt{}, errors.New("evaluation: trajectory has ambiguous receipt closure")
		}
		matched = current
	}
	if !matched.ID.Valid() {
		return runrecord.StageReceipt{}, errors.New("evaluation: trajectory lacks current receipt closure")
	}
	return matched, nil
}

func requireTrajectoryProgress(
	ctx context.Context,
	reader artifact.Reader,
	previous, current trajectoryExchange,
) ([]artifact.ID, error) {
	newResolutions := slices.DeleteFunc(slices.Clone(current.Trace.Resolutions), func(id artifact.ID) bool {
		return slices.Contains(previous.Trace.Resolutions, id)
	})
	progress := make([]artifact.ID, 0, len(newResolutions))
	for _, id := range newResolutions {
		resolution, err := runrecord.RequireAgentObligationResolution(ctx, reader, id)
		if err != nil {
			return nil, errors.Join(errors.New("evaluation: trajectory progress resolution is not typed"), err)
		}
		if !slices.Contains(previous.Trace.Obligations, resolution.Obligation) ||
			!slices.Contains(current.Trace.Obligations, resolution.Obligation) ||
			!slices.Contains(resolution.Evidence, current.Result) ||
			!slices.Contains(resolution.Evidence, current.Receipt.ID) {
			return nil, errors.New("evaluation: trajectory resolution is outside the exact evidence closure")
		}
		obligation, err := runrecord.RequireAgentObligation(ctx, reader, resolution.Obligation)
		if err != nil || !runrecord.AgentObligationSatisfied(obligation, []runrecord.AgentObligationResolution{resolution}) {
			return nil, errors.Join(errors.New("evaluation: trajectory resolution does not satisfy its exact obligation"), err)
		}
		for _, evidence := range resolution.Evidence {
			if _, found, requireErr := reader.Artifact(ctx, evidence); requireErr != nil || !found {
				return nil, errors.Join(errors.New("evaluation: trajectory resolution evidence is absent"), requireErr)
			}
		}
		progress = append(progress, id)
	}
	slices.SortFunc(progress, artifact.CompareID)
	return progress, nil
}

func canonicalizeTrajectoryHealthDecision(value *TrajectoryHealthDecision) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Previous.Kind() != artifact.KindEvidence || value.Current.Kind() != artifact.KindEvidence ||
		value.PreviousReceipt.Kind() != artifact.KindEvidence || value.CurrentReceipt.Kind() != artifact.KindEvidence ||
		value.Previous == value.Current || value.PreviousReceipt == value.CurrentReceipt ||
		!executionfailure.ValidDecision(value.Decision) {
		return errors.New("evaluation: invalid trajectory health decision")
	}
	value.Progress = slices.Clone(value.Progress)
	slices.SortFunc(value.Progress, artifact.CompareID)
	value.Progress = slices.Compact(value.Progress)
	if slices.ContainsFunc(value.Progress, func(id artifact.ID) bool { return id.Kind() != artifact.KindEvidence }) {
		return errors.New("evaluation: invalid trajectory progress")
	}
	switch value.Rule {
	case trajectoryRuleExactNoProgress:
		if value.Decision != executionfailure.DecisionRetireSession || len(value.Progress) != 0 {
			return errors.New("evaluation: invalid no-progress trajectory decision")
		}
	case trajectoryRuleNewEvidence:
		if value.Decision != executionfailure.DecisionResume || len(value.Progress) == 0 {
			return errors.New("evaluation: invalid progress trajectory decision")
		}
	case trajectoryRuleChangedAction:
		if value.Decision != executionfailure.DecisionResume || len(value.Progress) != 0 {
			return errors.New("evaluation: invalid changed-action trajectory decision")
		}
	default:
		return errors.New("evaluation: foreign trajectory supervision rule")
	}
	return nil
}
