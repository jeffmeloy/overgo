package runrecord

import (
	"bytes"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

const projectionFixtureCommit = "0123456789abcdef0123456789abcdef01234567"

var projectionArcBounds = dataset.InteractionArcProjectionBounds{
	MaximumBranches: 2, MaximumArcs: 8, MaximumProjectedSequences: 32,
	MaximumToolPairs: 8, MaximumReferences: 64,
}

type projectionAuthorityFixture struct {
	trace   InteractionTrace
	gate    GateResult
	attempt AttemptRecord
}

func TestCapabilityEpisodeProjection(t *testing.T) {
	directory := t.TempDir()
	store, err := overgodb.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	first := newProjectionAuthorityFixture(t, "first", projectionFixtureEvents(t, "first", "candidate/a"), nil)
	second := newProjectionAuthorityFixture(t, "second", projectionFixtureEvents(t, "second", "candidate/a"), func(trace *InteractionTrace) {
		trace.Terminal = OutcomeFailed
	})
	commitProjectionAuthorities(t, store, "capability-episode/sources", []projectionAuthorityFixture{first, second})
	bounds := dataset.CapabilityEpisodeProjectionBounds{
		MaximumEpisodes: 2, MaximumEventsPerEpisode: 16, MaximumCallsPerEpisode: 8, MaximumReferencesPerEpisode: 32,
	}
	ordered := []CapabilityEpisodeAuthoritySource{
		{Attempt: first.attempt.ID, Trajectory: first.trace.ID},
		{Attempt: second.attempt.ID, Trajectory: second.trace.ID},
	}
	permuted := slices.Clone(ordered)
	slices.Reverse(permuted)
	one, err := BuildCapabilityEpisodeProjection(t.Context(), store, bounds, ordered)
	if err != nil {
		t.Fatal(err)
	}
	two, err := BuildCapabilityEpisodeProjection(t.Context(), store, bounds, permuted)
	if err != nil || one.ID != two.ID || one.SourceSet != two.SourceSet || len(one.Episodes) != 2 {
		t.Fatalf("permuted episode projections = (%s, %s, %v)", one.ID, two.ID, err)
	}
	terminalOutcomes := map[string]bool{}
	for _, episode := range one.Episodes {
		terminalOutcomes[episode.Terminal] = true
		if episode.GateRecipe == episode.ExecutionRecipe || episode.AttemptOutcome != string(OutcomeSucceeded) ||
			episode.ToolCalls != 2 || episode.RepeatedCalls != 1 || episode.ToolFailures != 1 ||
			episode.Events != 8 || !episode.Coverage.Strategy || !episode.Coverage.Environment || !episode.Coverage.CostUnits ||
			episode.Coverage.ToolManuals != 1 || episode.Coverage.InvocationEffects != 1 ||
			episode.Coverage.Obligations != 1 || episode.Coverage.Resolutions != 1 ||
			episode.Coverage.PriorAttempts != 1 || episode.Coverage.Context != 1 {
			t.Fatalf("derived episode facts = %+v", episode)
		}
	}
	if !terminalOutcomes[string(OutcomeSucceeded)] || !terminalOutcomes[string(OutcomeFailed)] {
		t.Fatalf("attempt and terminal outcomes were collapsed: %+v", terminalOutcomes)
	}
	content, err := one.Content()
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range [][]byte{[]byte("secret-request"), []byte("secret-result"), []byte("call-first"), []byte("call-second")} {
		if bytes.Contains(content.Data, forbidden) {
			t.Fatalf("projection copied source payload %q", forbidden)
		}
	}
	if err := one.ValidateIdentity(); err != nil || len(one.Lineage()) != 4 {
		t.Fatalf("episode projection identity = (%d, %v)", len(one.Lineage()), err)
	}
	batch, err := one.Batch("capability-episode/projection")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	stored, err := RequireCapabilityEpisodeProjection(t.Context(), store, one.ID)
	if err != nil || stored.ID != one.ID {
		t.Fatalf("verified episode projection = (%s, %v)", stored.ID, err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := overgodb.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	cold, err := RequireCapabilityEpisodeProjection(t.Context(), reopened, one.ID)
	if err != nil || cold.ID != one.ID {
		t.Fatalf("cold verified episode rebuild = (%s, %s, %v)", one.ID, cold.ID, err)
	}
}

func TestCapabilityEpisodeProjectionRejectsCrossLinkedAuthority(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*testing.T, *projectionAuthorityFixture) *InteractionTrace
	}{
		{name: "task", mutate: func(t *testing.T, fixture *projectionAuthorityFixture) *InteractionTrace {
			fixture.attempt = replaceProjectionAttempt(t, fixture.attempt, func(value *AttemptRecord) {
				value.TaskContract = testutil.ArtifactID(t, artifact.KindRecipe, "foreign-task")
			})
			return nil
		}},
		{name: "strategy", mutate: func(t *testing.T, fixture *projectionAuthorityFixture) *InteractionTrace {
			fixture.attempt = replaceProjectionAttempt(t, fixture.attempt, func(value *AttemptRecord) {
				value.StrategyID = testutil.ArtifactID(t, artifact.KindProfile, "foreign-strategy")
			})
			return nil
		}},
		{name: "gate-recipe", mutate: func(t *testing.T, fixture *projectionAuthorityFixture) *InteractionTrace {
			fixture.attempt = replaceProjectionAttempt(t, fixture.attempt, func(value *AttemptRecord) {
				value.Recipe = testutil.ArtifactID(t, artifact.KindRecipe, "foreign-gate-recipe")
			})
			return nil
		}},
		{name: "trajectory", mutate: func(t *testing.T, fixture *projectionAuthorityFixture) *InteractionTrace {
			return new(newProjectionTrace(t, "foreign-trajectory", projectionFixtureEvents(t, "foreign", "candidate/a"),
				fixture.trace.TaskContract, fixture.trace.Strategy, fixture.trace.Recipe))
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			store, err := overgodb.Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			fixture := newProjectionAuthorityFixture(t, test.name, projectionFixtureEvents(t, test.name, "candidate/a"), nil)
			foreign := test.mutate(t, &fixture)
			fixtures := []projectionAuthorityFixture{fixture}
			trajectory := fixture.trace.ID
			if foreign != nil {
				fixtures = append(fixtures, projectionAuthorityFixture{trace: *foreign})
				trajectory = foreign.ID
			}
			commitProjectionAuthorities(t, store, "capability-episode/cross-link-"+test.name, fixtures)
			bounds := dataset.CapabilityEpisodeProjectionBounds{
				MaximumEpisodes: 1, MaximumEventsPerEpisode: 16, MaximumCallsPerEpisode: 8, MaximumReferencesPerEpisode: 32,
			}
			_, err = BuildCapabilityEpisodeProjection(t.Context(), store, bounds, []CapabilityEpisodeAuthoritySource{{
				Attempt: fixture.attempt.ID, Trajectory: trajectory,
			}})
			if err == nil || !strings.Contains(err.Error(), "cross-linked") {
				t.Fatalf("cross-linked %s authority accepted: %v", test.name, err)
			}
		})
	}
}

func TestInteractionArcProjectionPreservesAtomicEvidence(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	fixture := newProjectionAuthorityFixture(t, "atomic", projectionFixtureEvents(t, "atomic", "candidate/a"), nil)
	commitProjectionAuthorities(t, store, "interaction-arc/atomic", []projectionAuthorityFixture{fixture})
	projection, arcs, err := BuildInteractionArcProjection(t.Context(), store, fixture.trace.ID, projectionArcBounds)
	if err != nil || len(projection.Branches) != 1 || projection.Branches[0] != "candidate/a" || len(arcs) != 2 {
		t.Fatalf("interaction arc projection = (%+v, %v)", projection, err)
	}
	for index, arc := range arcs {
		if projection.Arcs[index].Arc != arc.ID || len(arc.ToolPairs) != 1 || len(arc.References) != 11 ||
			arc.ToolPairs[0].CallSequence >= arc.ToolPairs[0].ResultSequence ||
			!slices.Contains(arc.Sequences, arc.ToolPairs[0].CallSequence) ||
			!slices.Contains(arc.Sequences, arc.ToolPairs[0].ResultSequence) {
			t.Fatalf("non-atomic interaction arc = %+v", arc)
		}
		content, contentErr := arc.Content()
		if contentErr != nil {
			t.Fatal(contentErr)
		}
		if bytes.Contains(content.Data, []byte("secret-request")) || bytes.Contains(content.Data, []byte("secret-result")) ||
			bytes.Contains(content.Data, []byte("call-atomic")) {
			t.Fatalf("interaction arc copied source payload: %s", content.Data)
		}
	}
	if len(projection.Lineage()) != 3 {
		t.Fatalf("projection lineage = %+v", projection.Lineage())
	}
	batch, err := projection.Batch("interaction-arc/projection", arcs)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	stored, storedArcs, err := RequireInteractionArcProjection(t.Context(), store, projection.ID)
	if err != nil || stored.ID != projection.ID || len(storedArcs) != 2 {
		t.Fatalf("verified interaction arc projection = (%s, %d, %v)", stored.ID, len(storedArcs), err)
	}
	for _, entry := range stored.Arcs {
		if content, found, readErr := artifact.ReadContent(t.Context(), store, entry.Arc); readErr != nil || !found || len(content.Data) == 0 {
			t.Fatalf("arc evidence %s is not readable: found=%v err=%v", entry.Arc, found, readErr)
		}
	}

	t.Run("missing result", func(t *testing.T) {
		events := projectionFixtureEvents(t, "missing", "candidate/a")
		events = slices.Delete(events, 2, 3)
		for index := range events {
			events[index].Sequence = uint32(index + 1)
		}
		trace := newProjectionTrace(t, "missing", events, fixture.trace.TaskContract, fixture.trace.Strategy, fixture.trace.Recipe)
		commitProjectionAuthorities(t, store, "interaction-arc/missing", []projectionAuthorityFixture{{trace: trace}})
		if _, _, err := BuildInteractionArcProjection(t.Context(), store, trace.ID, projectionArcBounds); err == nil {
			t.Fatal("arc projection fabricated a missing tool result")
		}
	})

	t.Run("cross branch result", func(t *testing.T) {
		events := projectionFixtureEvents(t, "cross-branch", "candidate/a")[:4]
		events[2].Branch = "candidate/b"
		trace := newProjectionTrace(t, "cross-branch", events, fixture.trace.TaskContract, fixture.trace.Strategy, fixture.trace.Recipe)
		commitProjectionAuthorities(t, store, "interaction-arc/cross-branch", []projectionAuthorityFixture{{trace: trace}})
		if _, _, err := BuildInteractionArcProjection(t.Context(), store, trace.ID, projectionArcBounds); err == nil {
			t.Fatal("arc projection joined a call and result across branches")
		}
	})

	for name, bounds := range map[string]dataset.InteractionArcProjectionBounds{
		"arc":       {MaximumBranches: 2, MaximumArcs: 1, MaximumProjectedSequences: 32, MaximumToolPairs: 8, MaximumReferences: 64},
		"sequence":  {MaximumBranches: 2, MaximumArcs: 8, MaximumProjectedSequences: 7, MaximumToolPairs: 8, MaximumReferences: 64},
		"pair":      {MaximumBranches: 2, MaximumArcs: 8, MaximumProjectedSequences: 32, MaximumToolPairs: 1, MaximumReferences: 64},
		"reference": {MaximumBranches: 2, MaximumArcs: 8, MaximumProjectedSequences: 32, MaximumToolPairs: 8, MaximumReferences: 21},
	} {
		t.Run(name+" bound", func(t *testing.T) {
			if _, _, err := BuildInteractionArcProjection(t.Context(), store, fixture.trace.ID, bounds); err == nil {
				t.Fatalf("%s expansion exceeded its identity-bound limit", name)
			}
		})
	}
}

func TestInteractionArcSelectionIsHeadBound(t *testing.T) {
	directory := t.TempDir()
	store, err := overgodb.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	fixture := newProjectionAuthorityFixture(t, "selection", projectionFixtureEvents(t, "selection", "candidate/a"), nil)
	commitProjectionAuthorities(t, store, "interaction-arc/selection", []projectionAuthorityFixture{fixture})
	projection, arcs, err := BuildInteractionArcProjection(t.Context(), store, fixture.trace.ID, projectionArcBounds)
	if err != nil {
		t.Fatal(err)
	}
	projectionBatch, err := projection.Batch("interaction-arc/selection-projection", arcs)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), projectionBatch); err != nil {
		t.Fatal(err)
	}
	tokenizer := testutil.ArtifactID(t, artifact.KindTokenizer, "selection-tokenizer")
	counter := testutil.ArtifactID(t, artifact.KindProfile, "selection-counter")
	policy, err := dataset.NewInteractionArcMeasurementPolicy(tokenizer, counter)
	if err != nil {
		t.Fatal(err)
	}
	policyBatch, err := policy.Batch("interaction-arc/selection-policy")
	if err != nil {
		t.Fatal(err)
	}
	policyBatch.Artifacts = append(policyBatch.Artifacts, artifact.Descriptor{ID: tokenizer}, artifact.Descriptor{ID: counter})
	if _, err := store.Commit(t.Context(), policyBatch); err != nil {
		t.Fatal(err)
	}
	bounds := dataset.InteractionSelectionBounds{
		MaxTokens: 1 << 20, MaxBytes: 1 << 20, MaxDocuments: 1, MaxDepth: 2, MaxResults: 1,
	}
	head, _ := store.Head()
	first, err := SelectInteractionArcs(t.Context(), store, projection.ID, "candidate/a", bounds, policy.ID)
	if err != nil || first.Head != head || len(first.Sources) != 1 || !first.Truncated || first.Population != 2 ||
		len(first.Measurements) != len(arcs) || first.Sources[0].Source != arcs[len(arcs)-1].ID {
		t.Fatalf("single-admission arc selection = (%+v, %v)", first, err)
	}
	for _, id := range first.Measurements {
		measurement, loadErr := dataset.LoadInteractionArcMeasurement(t.Context(), store, id)
		if loadErr != nil || measurement.Transcript != fixture.trace.Request ||
			!slices.Equal(measurement.Context, fixture.trace.Context) || measurement.Tokens != measurement.Bytes ||
			measurement.Bytes == 0 {
			t.Fatalf("owner-derived interaction cost = (%+v, %v)", measurement, loadErr)
		}
	}
	replayed, err := RequireInteractionArcSelection(t.Context(), store, first.ID)
	if err != nil || replayed.ID != first.ID || replayed.Head != head {
		t.Fatalf("durable interaction selection replay = (%s, %v)", replayed.ID, err)
	}
	missingContext := newProjectionAuthorityFixture(
		t, "selection-missing-context", projectionFixtureEvents(t, "selection-missing-context", "candidate/a"),
		func(trace *InteractionTrace) {
			trace.Context = []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "selection-unresolved-context")}
		},
	)
	commitProjectionAuthorities(t, store, "interaction-arc/selection-missing-context", []projectionAuthorityFixture{missingContext})
	missingProjection, missingArcs, err := BuildInteractionArcProjection(
		t.Context(), store, missingContext.trace.ID, projectionArcBounds,
	)
	if err != nil {
		t.Fatal(err)
	}
	missingBatch, err := missingProjection.Batch("interaction-arc/selection-missing-context-projection", missingArcs)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), missingBatch); err != nil {
		t.Fatal(err)
	}
	if _, err := SelectInteractionArcs(
		t.Context(), store, missingProjection.ID, "candidate/a", bounds, policy.ID,
	); err == nil || !strings.Contains(err.Error(), "context content is absent") {
		t.Fatalf("descriptor-only interaction context supplied cost authority: %v", err)
	}

	verifiedProjection, verifiedArcs, err := RequireInteractionArcProjection(t.Context(), store, projection.ID)
	if err != nil {
		t.Fatal(err)
	}
	derived, err := deriveInteractionArcMeasurements(
		t.Context(), store, verifiedProjection, verifiedArcs, "candidate/a", policy,
	)
	if err != nil {
		t.Fatal(err)
	}
	forged := slices.Clone(derived)
	forged[0], err = dataset.NewInteractionArcMeasurement(
		forged[0].Arc, forged[0].Policy, forged[0].Transcript, forged[0].Context,
		forged[0].Tokens+1, forged[0].Bytes+1,
	)
	if err != nil {
		t.Fatal(err)
	}
	forgedHead, _ := store.Head()
	forgedSelection, err := dataset.SelectInteractionArcsOnce(
		bounds, forgedHead, verifiedProjection, verifiedArcs, "candidate/a", forged,
	)
	if err != nil {
		t.Fatal(err)
	}
	forgedBatch, err := forgedSelection.Batch("interaction-arc/selection-forged-cost", forged)
	if err != nil {
		t.Fatal(err)
	}
	forgedBatch.ExpectedHead = &forgedHead
	if _, err := store.Commit(t.Context(), forgedBatch); err != nil {
		t.Fatal(err)
	}
	if _, err := RequireInteractionArcSelection(t.Context(), store, forgedSelection.ID); err == nil ||
		!strings.Contains(err.Error(), "cost differs") {
		t.Fatalf("caller-authored interaction cost was replayable: %v", err)
	}

	lineageHead, _ := store.Head()
	lineageSelection, err := dataset.SelectInteractionArcsOnce(
		bounds, lineageHead, verifiedProjection, verifiedArcs, "candidate/a", derived,
	)
	if err != nil {
		t.Fatal(err)
	}
	lineageBatch, err := lineageSelection.Batch("interaction-arc/selection-forged-lineage", derived)
	if err != nil {
		t.Fatal(err)
	}
	extraAuthority := testutil.ArtifactID(t, artifact.KindEvidence, "selection-extra-authority")
	lineageBatch.Artifacts = append(lineageBatch.Artifacts, artifact.Descriptor{ID: extraAuthority})
	lineageBatch.Lineage = append(lineageBatch.Lineage, artifact.DependencyLineage(lineageSelection.ID, extraAuthority)...)
	lineageBatch.ExpectedHead = &lineageHead
	if _, err := store.Commit(t.Context(), lineageBatch); err != nil {
		t.Fatal(err)
	}
	if _, err := RequireInteractionArcSelection(t.Context(), store, lineageSelection.ID); err == nil ||
		!strings.Contains(err.Error(), "lineage differs") {
		t.Fatalf("interaction selection with forged lineage was replayable: %v", err)
	}

	staleHead, _ := store.Head()
	staleSelection, err := dataset.SelectInteractionArcsOnce(
		bounds, staleHead, verifiedProjection, verifiedArcs, "candidate/a", derived,
	)
	if err != nil {
		t.Fatal(err)
	}
	staleBatch, err := staleSelection.Batch("interaction-arc/selection-stale", derived)
	if err != nil {
		t.Fatal(err)
	}
	staleBatch.ExpectedHead = &staleHead
	newHeadArtifact := testutil.ArtifactID(t, artifact.KindEvidence, "selection-new-head")
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "interaction-arc/selection-new-head", Artifacts: []artifact.Descriptor{{ID: newHeadArtifact}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), staleBatch); err == nil {
		t.Fatal("interaction selection publication accepted a stale source head")
	}
	second, err := SelectInteractionArcs(t.Context(), store, projection.ID, "candidate/a", bounds, policy.ID)
	if err != nil || second.Head == first.Head || second.ID == first.ID {
		t.Fatalf("selection did not bind the repository head: first=%s second=%s err=%v", first.ID, second.ID, err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := overgodb.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	cold, err := RequireInteractionArcSelection(t.Context(), reopened, second.ID)
	if err != nil || cold.ID != second.ID || cold.Head != second.Head {
		t.Fatalf("cold interaction selection replay = (%s, %v)", cold.ID, err)
	}
}

func newProjectionAuthorityFixture(
	t *testing.T,
	tag string,
	events []InteractionTraceEvent,
	mutate func(*InteractionTrace),
) projectionAuthorityFixture {
	t.Helper()
	task := testutil.ArtifactID(t, artifact.KindRecipe, tag+"-task")
	strategy := testutil.ArtifactID(t, artifact.KindProfile, tag+"-strategy")
	executionRecipe := testutil.ArtifactID(t, artifact.KindRecipe, tag+"-execution-recipe")
	gateRecipe := testutil.ArtifactID(t, artifact.KindRecipe, tag+"-gate-recipe")
	environment := testutil.ArtifactID(t, artifact.KindEvidence, tag+"-environment")
	trace := newProjectionTrace(t, tag, events, task, strategy, executionRecipe)
	if mutate != nil {
		mutate(&trace)
		var err error
		trace, err = NewAgentTrajectory(trace)
		if err != nil {
			t.Fatal(err)
		}
	}
	gateRecord, err := NewGateRecord(
		gateRecipe, environment, projectionFixtureCommit, OutcomeSucceeded, "", 101,
		[]GateStep{{Name: "verify", Phase: PhaseValidate, Outcome: StepSucceeded, DurationNS: 100}},
	)
	if err != nil {
		t.Fatal(err)
	}
	attempt, err := NewAttemptRecord(AttemptRecord{
		PlanItem: "experience-projection", PlanStep: "capability-episodes", Strategy: "strategy-" + tag,
		Result: gateRecord.Result.ID, Recipe: gateRecipe, CodeCommit: projectionFixtureCommit,
		Outcome: OutcomeSucceeded, WallNS: 100 + uint64(len(tag)), StrategyID: strategy, TaskContract: task,
		Environment: environment, Trajectory: trace.ID, CostUnits: 10 + uint64(len(tag)),
	})
	if err != nil {
		t.Fatal(err)
	}
	return projectionAuthorityFixture{trace: trace, gate: gateRecord.Result, attempt: attempt}
}

func newProjectionTrace(
	t *testing.T,
	tag string,
	events []InteractionTraceEvent,
	task, strategy, recipe artifact.ID,
) InteractionTrace {
	t.Helper()
	requestMessages := make([]InteractionMessage, len(events))
	for index, event := range events {
		requestMessages[index] = event.Message
	}
	if len(requestMessages) > 1 && requestMessages[len(requestMessages)-1].Role == "assistant" {
		requestMessages = requestMessages[:len(requestMessages)-1]
	}
	request, err := NewInteractionTranscript(requestMessages)
	if err != nil {
		t.Fatal(err)
	}
	trace, err := NewAgentTrajectory(InteractionTrace{
		Version: artifact.InitialDocumentVersion, Recipe: recipe,
		Model:     testutil.ArtifactID(t, artifact.KindModel, tag+"-model"),
		Operation: testutil.ArtifactID(t, artifact.KindEvidence, tag+"-operation"),
		Request:   request.ID, Events: events,
		TaskContract: task, Strategy: strategy,
		ToolActions:       []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, tag+"-tool-action")},
		Decisions:         []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, tag+"-decision")},
		FinalArtifacts:    []artifact.ID{testutil.ArtifactID(t, artifact.KindRun, tag+"-final")},
		ToolManuals:       []artifact.ID{testutil.ArtifactID(t, artifact.KindRecipe, tag+"-manual")},
		InvocationEffects: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, tag+"-effect")},
		Obligations:       []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, tag+"-obligation")},
		Resolutions:       []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, tag+"-resolution")},
		WorkspaceClaims:   []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, tag+"-claim")},
		Attempts:          []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, tag+"-prior-attempt")},
		BudgetCharges:     []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, tag+"-charge")},
		Context:           []artifact.ID{request.ID},
		Terminal:          OutcomeSucceeded,
	})
	if err != nil {
		t.Fatal(err)
	}
	return trace
}

func projectionFixtureEvents(t *testing.T, tag, branch string) []InteractionTraceEvent {
	t.Helper()
	manual := testutil.ArtifactID(t, artifact.KindRecipe, tag+"-manual")
	call := func(id string) InteractionMessage {
		return InteractionMessage{Role: "assistant", ToolCalls: []InteractionToolCall{{
			ID: id, Type: "function", Name: "inspect", Manual: manual, Arguments: `{"target":"same-secret"}`,
		}}}
	}
	return []InteractionTraceEvent{
		{Sequence: 1, Kind: InteractionEventRequest, Message: InteractionMessage{Role: "user", Content: "secret-request-one-" + tag}},
		{Sequence: 2, Kind: InteractionEventToolCall, Branch: branch, Message: call("call-" + tag + "-one")},
		{Sequence: 3, Kind: InteractionEventToolResult, Branch: branch, Message: InteractionMessage{Role: "tool", Content: "secret-result-one-" + tag, ToolCallID: "call-" + tag + "-one"}},
		{Sequence: 4, Kind: InteractionEventOutput, Branch: branch, Message: InteractionMessage{Role: "assistant", Content: "first answer"}},
		{Sequence: 5, Kind: InteractionEventRequest, Message: InteractionMessage{Role: "user", Content: "secret-request-two-" + tag}},
		{Sequence: 6, Kind: InteractionEventToolCall, Branch: branch, Message: call("call-" + tag + "-two")},
		{Sequence: 7, Kind: InteractionEventToolResult, Branch: branch, Message: InteractionMessage{Role: "tool", Content: "secret-result-two-" + tag, ToolCallID: "call-" + tag + "-two", ToolResultError: true}},
		{Sequence: 8, Kind: InteractionEventOutput, Branch: branch, Message: InteractionMessage{Role: "assistant", Content: "second answer"}},
	}
}

func replaceProjectionAttempt(t *testing.T, source AttemptRecord, mutate func(*AttemptRecord)) AttemptRecord {
	t.Helper()
	mutate(&source)
	source.ID = artifact.ID{}
	replacement, err := NewAttemptRecord(source)
	if err != nil {
		t.Fatal(err)
	}
	return replacement
}

func commitProjectionAuthorities(t *testing.T, store *overgodb.Store, key string, fixtures []projectionAuthorityFixture) {
	t.Helper()
	contents := make([]artifact.Content, 0, len(fixtures)*3)
	lineage := make([]artifact.Lineage, 0)
	contentIDs := map[artifact.ID]struct{}{}
	appendDocument := func(id artifact.ID, content artifact.Content, edges []artifact.Lineage) {
		if _, duplicate := contentIDs[id]; duplicate {
			return
		}
		contents = append(contents, content)
		contentIDs[id] = struct{}{}
		lineage = append(lineage, edges...)
	}
	for _, fixture := range fixtures {
		if fixture.trace.ID.Valid() {
			messages := make([]InteractionMessage, len(fixture.trace.Events))
			for index, event := range fixture.trace.Events {
				messages[index] = event.Message
			}
			if len(messages) > 1 && messages[len(messages)-1].Role == "assistant" {
				messages = messages[:len(messages)-1]
			}
			transcript, err := NewInteractionTranscript(messages)
			if err != nil || transcript.ID != fixture.trace.Request {
				t.Fatalf("projection request transcript = (%s, %s, %v)", transcript.ID, fixture.trace.Request, err)
			}
			transcriptContent, err := interactionTranscriptCodec.Content(transcript)
			if err != nil {
				t.Fatal(err)
			}
			appendDocument(transcript.ID, transcriptContent, nil)
			content, err := fixture.trace.Content()
			if err != nil {
				t.Fatal(err)
			}
			appendDocument(fixture.trace.ID, content, fixture.trace.Lineage())
		}
		if fixture.gate.ID.Valid() {
			content, err := fixture.gate.Content()
			if err != nil {
				t.Fatal(err)
			}
			appendDocument(fixture.gate.ID, content, fixture.gate.Lineage())
		}
		if fixture.attempt.ID.Valid() {
			content, err := fixture.attempt.Content()
			if err != nil {
				t.Fatal(err)
			}
			appendDocument(fixture.attempt.ID, content, fixture.attempt.Lineage())
		}
	}
	parents := map[artifact.ID]struct{}{}
	for _, edge := range lineage {
		if _, isContent := contentIDs[edge.Parent]; !isContent {
			parents[edge.Parent] = struct{}{}
		}
	}
	descriptors := make([]artifact.Descriptor, 0, len(parents))
	for id := range parents {
		descriptors = append(descriptors, artifact.Descriptor{ID: id})
	}
	slices.SortFunc(descriptors, func(left, right artifact.Descriptor) int { return artifact.CompareID(left.ID, right.ID) })
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: key, Artifacts: descriptors, Contents: contents, Lineage: lineage,
	}); err != nil {
		t.Fatal(err)
	}
}
