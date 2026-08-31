package runrecord

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
)

// CapabilityEpisodeAuthoritySource names the exact attempt and trajectory that
// the run-record owner must load before a dataset projection can be compiled.
type CapabilityEpisodeAuthoritySource struct {
	Attempt    artifact.ID `json:"attempt"`
	Trajectory artifact.ID `json:"trajectory"`
}

// BuildCapabilityEpisodeProjection loads canonical source records, validates
// their stored lineage and cross-links, and passes only immutable structural
// facts to the rebuildable dataset projection.
func BuildCapabilityEpisodeProjection(
	ctx context.Context,
	reader artifact.Reader,
	bounds dataset.CapabilityEpisodeProjectionBounds,
	sources []CapabilityEpisodeAuthoritySource,
) (dataset.CapabilityEpisodeProjection, error) {
	if ctx == nil || reader == nil || len(sources) == 0 {
		return dataset.CapabilityEpisodeProjection{}, errors.New("run record: capability episode source authority is absent")
	}
	inputs := make([]dataset.CapabilityEpisodeInput, len(sources))
	for index, source := range sources {
		attempt, err := RequireAttemptRecord(ctx, reader, source.Attempt)
		if err != nil {
			return dataset.CapabilityEpisodeProjection{}, err
		}
		trace, err := RequireInteractionTrace(ctx, reader, source.Trajectory)
		if err != nil {
			return dataset.CapabilityEpisodeProjection{}, err
		}
		gate, err := RequireGateResult(ctx, reader, attempt.Result)
		if err != nil {
			return dataset.CapabilityEpisodeProjection{}, err
		}
		if err := requireStoredLineageIncludes(ctx, reader, attempt.ID, attempt.Lineage()); err != nil {
			return dataset.CapabilityEpisodeProjection{}, err
		}
		if err := requireStoredLineageIncludes(ctx, reader, trace.ID, trace.Lineage()); err != nil {
			return dataset.CapabilityEpisodeProjection{}, err
		}
		if err := requireStoredLineageIncludes(ctx, reader, gate.ID, gate.Lineage()); err != nil {
			return dataset.CapabilityEpisodeProjection{}, err
		}
		if attempt.Trajectory != trace.ID || attempt.TaskContract != trace.TaskContract ||
			attempt.StrategyID != trace.Strategy || attempt.Result != gate.ID ||
			attempt.Recipe != gate.Recipe || attempt.Environment != gate.Environment ||
			attempt.CodeCommit != gate.CodeCommit || attempt.Outcome != gate.Outcome ||
			attempt.Failure != gate.Failure {
			return dataset.CapabilityEpisodeProjection{}, errors.New("run record: cross-linked capability episode authority")
		}
		events, err := datasetInteractionProtocolEvents(trace)
		if err != nil {
			return dataset.CapabilityEpisodeProjection{}, err
		}
		inputs[index] = dataset.CapabilityEpisodeInput{
			Attempt: attempt.ID, Trajectory: trace.ID, AttemptTrajectory: attempt.Trajectory,
			AttemptTaskContract: attempt.TaskContract, TraceTaskContract: trace.TaskContract,
			AttemptStrategy: attempt.StrategyID, TraceStrategy: trace.Strategy,
			AttemptRecipe: gate.Recipe, TraceRecipe: trace.Recipe, Model: trace.Model,
			Environment: gate.Environment, Result: gate.ID, Request: trace.Request,
			Operation: trace.Operation, AttemptOutcome: string(attempt.Outcome), TerminalOutcome: string(trace.Terminal),
			WallNS: attempt.WallNS, CostUnits: attempt.CostUnits, References: datasetCapabilityEpisodeReferences(trace),
			Events: events,
		}
	}
	return dataset.CompileCapabilityEpisodeProjection(bounds, inputs)
}

// RequireCapabilityEpisodeProjection reopens a projection only after rebuilding
// it from canonical run-record authorities and checking its stored lineage.
func RequireCapabilityEpisodeProjection(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (dataset.CapabilityEpisodeProjection, error) {
	stored, err := dataset.LoadCapabilityEpisodeProjection(ctx, reader, id)
	if err != nil {
		return dataset.CapabilityEpisodeProjection{}, err
	}
	sources := make([]CapabilityEpisodeAuthoritySource, len(stored.Episodes))
	for index, episode := range stored.Episodes {
		sources[index] = CapabilityEpisodeAuthoritySource{Attempt: episode.Attempt, Trajectory: episode.Trajectory}
	}
	rebuilt, err := BuildCapabilityEpisodeProjection(ctx, reader, stored.Bounds, sources)
	if err != nil {
		return dataset.CapabilityEpisodeProjection{}, err
	}
	if rebuilt.ID != stored.ID {
		return dataset.CapabilityEpisodeProjection{}, errors.New("run record: capability episode projection differs from its authorities")
	}
	if err := requireStoredLineageExact(ctx, reader, stored.ID, rebuilt.Lineage()); err != nil {
		return dataset.CapabilityEpisodeProjection{}, err
	}
	return rebuilt, nil
}

// BuildInteractionArcProjection derives bounded, independently resolvable arc
// evidence without copying messages or results.
func BuildInteractionArcProjection(
	ctx context.Context,
	reader artifact.Reader,
	trajectory artifact.ID,
	bounds dataset.InteractionArcProjectionBounds,
) (dataset.InteractionArcProjection, []dataset.InteractionArc, error) {
	if ctx == nil || reader == nil {
		return dataset.InteractionArcProjection{}, nil, errors.New("run record: interaction arc source authority is absent")
	}
	trace, err := RequireInteractionTrace(ctx, reader, trajectory)
	if err != nil {
		return dataset.InteractionArcProjection{}, nil, err
	}
	if err := requireStoredLineageIncludes(ctx, reader, trace.ID, trace.Lineage()); err != nil {
		return dataset.InteractionArcProjection{}, nil, err
	}
	events, err := datasetInteractionProtocolEvents(trace)
	if err != nil {
		return dataset.InteractionArcProjection{}, nil, err
	}
	return dataset.CompileInteractionArcProjection(dataset.InteractionArcProjectionInput{
		Trace: trace.ID, Events: events, References: datasetCapabilityEpisodeReferences(trace),
	}, bounds)
}

// RequireInteractionArcProjection rebuilds the projection and every arc from
// its trajectory, then verifies the stored dependency edges exactly.
func RequireInteractionArcProjection(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (dataset.InteractionArcProjection, []dataset.InteractionArc, error) {
	stored, err := dataset.LoadInteractionArcProjection(ctx, reader, id)
	if err != nil {
		return dataset.InteractionArcProjection{}, nil, err
	}
	rebuilt, arcs, err := BuildInteractionArcProjection(ctx, reader, stored.Trace, stored.Bounds)
	if err != nil {
		return dataset.InteractionArcProjection{}, nil, err
	}
	if rebuilt.ID != stored.ID || len(rebuilt.Arcs) != len(stored.Arcs) {
		return dataset.InteractionArcProjection{}, nil, errors.New("run record: interaction arc projection differs from its trajectory")
	}
	for index, expected := range arcs {
		if rebuilt.Arcs[index] != stored.Arcs[index] {
			return dataset.InteractionArcProjection{}, nil, errors.New("run record: interaction arc index differs from its trajectory")
		}
		published, loadErr := dataset.LoadInteractionArc(ctx, reader, expected.ID)
		if loadErr != nil || published.ID != expected.ID {
			return dataset.InteractionArcProjection{}, nil, errors.Join(errors.New("run record: interaction arc evidence is absent"), loadErr)
		}
		if lineageErr := requireStoredLineageExact(ctx, reader, expected.ID, expected.Lineage()); lineageErr != nil {
			return dataset.InteractionArcProjection{}, nil, lineageErr
		}
	}
	if err := requireStoredLineageExact(ctx, reader, stored.ID, rebuilt.Lineage()); err != nil {
		return dataset.InteractionArcProjection{}, nil, err
	}
	return rebuilt, arcs, nil
}

// SelectInteractionArcs derives cost from exact immutable run-record content,
// makes one non-resumable admission against a stable repository head, and
// atomically publishes the measurements and selection that can replay it.
func SelectInteractionArcs(
	ctx context.Context,
	repository artifact.Repository,
	projectionID artifact.ID,
	branch string,
	bounds dataset.InteractionSelectionBounds,
	policyID artifact.ID,
) (dataset.InteractionArcSelection, error) {
	if ctx == nil || repository == nil || policyID.Kind() != artifact.KindProfile {
		return dataset.InteractionArcSelection{}, errors.New("run record: interaction arc selection authority is absent")
	}
	head, generation := repository.Head()
	if !head.Valid() {
		return dataset.InteractionArcSelection{}, errors.New("run record: interaction arc selection requires a committed head")
	}
	projection, arcs, err := RequireInteractionArcProjection(ctx, repository, projectionID)
	if err != nil {
		return dataset.InteractionArcSelection{}, err
	}
	policy, err := dataset.LoadInteractionArcMeasurementPolicy(ctx, repository, policyID)
	if err != nil {
		return dataset.InteractionArcSelection{}, err
	}
	if err := requireStoredLineageExact(ctx, repository, policy.ID, policy.Lineage()); err != nil {
		return dataset.InteractionArcSelection{}, err
	}
	measurements, err := deriveInteractionArcMeasurements(ctx, repository, projection, arcs, branch, policy)
	if err != nil {
		return dataset.InteractionArcSelection{}, err
	}
	endHead, endGeneration := repository.Head()
	if endHead != head || endGeneration != generation {
		return dataset.InteractionArcSelection{}, errors.New("run record: repository changed during interaction arc selection")
	}
	selection, err := dataset.SelectInteractionArcsOnce(bounds, head, projection, arcs, branch, measurements)
	if err != nil {
		return dataset.InteractionArcSelection{}, err
	}
	batch, err := selection.Batch("interaction-arc/selection/"+selection.ID.String(), measurements)
	if err != nil {
		return dataset.InteractionArcSelection{}, err
	}
	expected := head
	batch.ExpectedHead = &expected
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return dataset.InteractionArcSelection{}, err
	}
	return RequireInteractionArcSelection(ctx, repository, selection.ID)
}

// RequireInteractionArcSelection reopens a durable admission only after
// rederiving every cost from its exact transcript/context authorities and
// reproducing the selected page at the recorded source head.
func RequireInteractionArcSelection(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (dataset.InteractionArcSelection, error) {
	selection, err := dataset.LoadInteractionArcSelection(ctx, reader, id)
	if err != nil {
		return dataset.InteractionArcSelection{}, err
	}
	if err := requireStoredLineageExact(ctx, reader, selection.ID, selection.Lineage()); err != nil {
		return dataset.InteractionArcSelection{}, err
	}
	projection, arcs, err := RequireInteractionArcProjection(ctx, reader, selection.Projection)
	if err != nil {
		return dataset.InteractionArcSelection{}, err
	}
	policy, err := dataset.LoadInteractionArcMeasurementPolicy(ctx, reader, selection.Policy)
	if err != nil {
		return dataset.InteractionArcSelection{}, err
	}
	if err := requireStoredLineageExact(ctx, reader, policy.ID, policy.Lineage()); err != nil {
		return dataset.InteractionArcSelection{}, err
	}
	derived, err := deriveInteractionArcMeasurements(ctx, reader, projection, arcs, selection.Branch, policy)
	if err != nil || len(derived) != len(selection.Measurements) {
		return dataset.InteractionArcSelection{}, errors.Join(errors.New("run record: interaction arc selection measurement population differs"), err)
	}
	for index, expected := range derived {
		if expected.ID != selection.Measurements[index] {
			return dataset.InteractionArcSelection{}, errors.New("run record: interaction arc selection cost differs from immutable content")
		}
		stored, loadErr := dataset.LoadInteractionArcMeasurement(ctx, reader, expected.ID)
		if loadErr != nil || stored.ID != expected.ID {
			return dataset.InteractionArcSelection{}, errors.Join(errors.New("run record: interaction arc selection measurement is absent"), loadErr)
		}
		if lineageErr := requireStoredLineageExact(ctx, reader, stored.ID, expected.Lineage()); lineageErr != nil {
			return dataset.InteractionArcSelection{}, lineageErr
		}
	}
	rebuilt, err := dataset.SelectInteractionArcsOnce(
		selection.Bounds, selection.Head, projection, arcs, selection.Branch, derived,
	)
	if err != nil || rebuilt.ID != selection.ID {
		return dataset.InteractionArcSelection{}, errors.Join(errors.New("run record: interaction arc selection does not replay"), err)
	}
	return rebuilt, nil
}

func deriveInteractionArcMeasurements(
	ctx context.Context,
	reader artifact.Reader,
	projection dataset.InteractionArcProjection,
	arcs []dataset.InteractionArc,
	branch string,
	policy dataset.InteractionArcMeasurementPolicy,
) ([]dataset.InteractionArcMeasurement, error) {
	trace, err := RequireInteractionTrace(ctx, reader, projection.Trace)
	if err != nil {
		return nil, err
	}
	transcript, err := RequireInteractionTranscript(ctx, reader, trace.Request)
	if err != nil {
		return nil, err
	}
	if len(transcript.Messages) > len(trace.Events) {
		return nil, errors.New("run record: interaction transcript exceeds its immutable trace")
	}
	for index, message := range transcript.Messages {
		if !reflect.DeepEqual(message, trace.Events[index].Message) {
			return nil, errors.New("run record: interaction transcript differs from its immutable trace")
		}
	}
	transcriptContent, err := interactionTranscriptCodec.Content(transcript)
	if err != nil {
		return nil, err
	}
	authorityBytes := transcriptContent.Descriptor.Size
	contextIDs := slices.DeleteFunc(slices.Clone(trace.Context), func(id artifact.ID) bool { return id == trace.Request })
	err = artifact.ReadContents(ctx, reader, contextIDs, func(content artifact.Content) error {
		if authorityBytes > math.MaxUint64-content.Descriptor.Size {
			return errors.New("run record: interaction context cost overflows")
		}
		authorityBytes += content.Descriptor.Size
		return nil
	})
	if err != nil {
		return nil, errors.Join(errors.New("run record: interaction context content is absent"), err)
	}
	events := make(map[uint32]InteractionTraceEvent, len(trace.Events))
	for _, event := range trace.Events {
		events[event.Sequence] = event
	}
	arcByID := make(map[artifact.ID]dataset.InteractionArc, len(arcs))
	for _, arc := range arcs {
		arcByID[arc.ID] = arc
	}
	measurements := make([]dataset.InteractionArcMeasurement, 0)
	for _, entry := range projection.Arcs {
		if entry.Branch != branch {
			continue
		}
		arc, found := arcByID[entry.Arc]
		if !found {
			return nil, errors.New("run record: interaction arc measurement source is absent")
		}
		selectedEvents := make([]InteractionTraceEvent, len(arc.Sequences))
		for eventIndex, sequence := range arc.Sequences {
			event, eventFound := events[sequence]
			if !eventFound {
				return nil, errors.New("run record: interaction arc sequence is absent from its trace")
			}
			selectedEvents[eventIndex] = event
		}
		encoded, encodeErr := json.Marshal(selectedEvents)
		if encodeErr != nil {
			return nil, encodeErr
		}
		eventBytes := uint64(len(encoded))
		if authorityBytes > math.MaxUint64-eventBytes {
			return nil, errors.New("run record: interaction arc cost overflows")
		}
		cost := authorityBytes + eventBytes
		measurement, measureErr := dataset.NewInteractionArcMeasurement(
			arc.ID, policy.ID, trace.Request, trace.Context, cost, cost,
		)
		if measureErr != nil {
			return nil, measureErr
		}
		measurements = append(measurements, measurement)
	}
	if len(measurements) == 0 {
		return nil, errors.New("run record: interaction arc branch has no measurable evidence")
	}
	return measurements, nil
}

func datasetInteractionProtocolEvents(trace InteractionTrace) ([]dataset.InteractionProtocolEvent, error) {
	if _, _, err := trace.ToolExchanges(); err != nil {
		return nil, err
	}
	events := make([]dataset.InteractionProtocolEvent, len(trace.Events))
	for index, event := range trace.Events {
		mapped := dataset.InteractionProtocolEvent{
			Sequence: event.Sequence, Branch: event.Branch, ResultCallID: event.Message.ToolCallID,
			ResultFailed: event.Message.ToolResultError,
		}
		switch event.Kind {
		case InteractionEventRequest:
			mapped.Kind = dataset.InteractionProtocolRequest
		case InteractionEventOutput:
			mapped.Kind = dataset.InteractionProtocolOutput
		case InteractionEventToolCall:
			mapped.Kind = dataset.InteractionProtocolToolCall
			mapped.Calls = make([]dataset.InteractionCallReference, len(event.Message.ToolCalls))
			for callIndex, call := range event.Message.ToolCalls {
				reference, err := dataset.NewInteractionCallReference(call.ID, call.Type, call.Name, call.Manual, call.Arguments)
				if err != nil {
					return nil, err
				}
				mapped.Calls[callIndex] = reference
			}
		case InteractionEventToolResult:
			mapped.Kind = dataset.InteractionProtocolToolResult
		default:
			return nil, errors.New("run record: unknown interaction event kind")
		}
		events[index] = mapped
	}
	return events, nil
}

func datasetCapabilityEpisodeReferences(trace InteractionTrace) []dataset.CapabilityEpisodeReference {
	groups := []struct {
		kind dataset.CapabilityEpisodeReferenceKind
		ids  []artifact.ID
	}{
		{dataset.CapabilityEpisodeToolAction, trace.ToolActions},
		{dataset.CapabilityEpisodeDecision, trace.Decisions},
		{dataset.CapabilityEpisodeFinalArtifact, trace.FinalArtifacts},
		{dataset.CapabilityEpisodeToolManual, trace.ToolManuals},
		{dataset.CapabilityEpisodeInvocationEffect, trace.InvocationEffects},
		{dataset.CapabilityEpisodeObligation, trace.Obligations},
		{dataset.CapabilityEpisodeResolution, trace.Resolutions},
		{dataset.CapabilityEpisodeWorkspaceClaim, trace.WorkspaceClaims},
		{dataset.CapabilityEpisodePriorAttempt, trace.Attempts},
		{dataset.CapabilityEpisodeBudgetCharge, trace.BudgetCharges},
		{dataset.CapabilityEpisodeContext, trace.Context},
	}
	count := 0
	for _, group := range groups {
		count += len(group.ids)
	}
	references := make([]dataset.CapabilityEpisodeReference, 0, count)
	for _, group := range groups {
		for _, id := range group.ids {
			references = append(references, dataset.CapabilityEpisodeReference{Kind: group.kind, ID: id})
		}
	}
	return slices.Clone(references)
}

func requireStoredLineageIncludes(ctx context.Context, reader artifact.Reader, child artifact.ID, required []artifact.Lineage) error {
	actual, err := reader.Parents(ctx, child)
	if err != nil {
		return err
	}
	actual = canonicalLineage(actual)
	required = canonicalLineage(required)
	for _, edge := range required {
		if _, found := slices.BinarySearchFunc(actual, edge, compareLineage); !found {
			return errors.New("run record: stored authority lineage is incomplete")
		}
	}
	return nil
}

func requireStoredLineageExact(ctx context.Context, reader artifact.Reader, child artifact.ID, expected []artifact.Lineage) error {
	actual, err := reader.Parents(ctx, child)
	if err != nil {
		return err
	}
	actual = canonicalLineage(actual)
	expected = canonicalLineage(expected)
	if !slices.Equal(actual, expected) {
		return errors.New("run record: stored projection lineage differs from its immutable authorities")
	}
	return nil
}

func canonicalLineage(values []artifact.Lineage) []artifact.Lineage {
	values = slices.Clone(values)
	slices.SortFunc(values, compareLineage)
	return values
}

func compareLineage(left, right artifact.Lineage) int {
	if order := artifact.CompareID(left.Child, right.Child); order != 0 {
		return order
	}
	if order := artifact.CompareID(left.Parent, right.Parent); order != 0 {
		return order
	}
	return cmp.Compare(left.Relation, right.Relation)
}
