package dataset

import (
	"cmp"
	"context"
	"errors"
	"math"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
)

const (
	// CapabilityEpisodeProjectionMediaType identifies a rebuildable episode view.
	CapabilityEpisodeProjectionMediaType = "application/vnd.overgo.capability-episode-projection+json"
	// CapabilityEpisodeProjectionSchema identifies the exact episode projection contract.
	CapabilityEpisodeProjectionSchema = "overgo/capability-episode-projection/v1"
)

// CapabilityEpisodeProjectionBounds makes every materialized fact population
// explicit. It is part of the projection identity rather than a runtime knob.
type CapabilityEpisodeProjectionBounds struct {
	MaximumEpisodes             uint32 `json:"maximum_episodes"`
	MaximumEventsPerEpisode     uint32 `json:"maximum_events_per_episode"`
	MaximumCallsPerEpisode      uint32 `json:"maximum_calls_per_episode"`
	MaximumReferencesPerEpisode uint32 `json:"maximum_references_per_episode"`
}

// Identify validates and identifies one exact projection budget.
func (bounds CapabilityEpisodeProjectionBounds) Identify() (artifact.ID, error) {
	if bounds.MaximumEpisodes == 0 || bounds.MaximumEventsPerEpisode == 0 ||
		bounds.MaximumCallsPerEpisode == 0 || bounds.MaximumReferencesPerEpisode == 0 {
		return artifact.ID{}, errors.New("dataset: capability episode projection requires every bound")
	}
	return artifact.JSONID(artifact.KindProfile, bounds)
}

// CapabilityEpisodeReferenceKind classifies an exact source identity without
// copying the payload owned by that source.
type CapabilityEpisodeReferenceKind string

const (
	// CapabilityEpisodeToolAction cites one attempted tool action.
	CapabilityEpisodeToolAction CapabilityEpisodeReferenceKind = "tool-action"
	// CapabilityEpisodeDecision cites one stored decision.
	CapabilityEpisodeDecision CapabilityEpisodeReferenceKind = "decision"
	// CapabilityEpisodeFinalArtifact cites one terminal output artifact.
	CapabilityEpisodeFinalArtifact CapabilityEpisodeReferenceKind = "final-artifact"
	// CapabilityEpisodeInvocationEffect cites one typed invocation effect.
	CapabilityEpisodeInvocationEffect CapabilityEpisodeReferenceKind = "invocation-effect"
	// CapabilityEpisodeObligation cites one unresolved or resolved obligation.
	CapabilityEpisodeObligation CapabilityEpisodeReferenceKind = "obligation"
	// CapabilityEpisodeResolution cites evidence that closes an obligation.
	CapabilityEpisodeResolution CapabilityEpisodeReferenceKind = "resolution"
	// CapabilityEpisodeWorkspaceClaim cites one workspace authority claim.
	CapabilityEpisodeWorkspaceClaim CapabilityEpisodeReferenceKind = "workspace-claim"
	// CapabilityEpisodeBudgetCharge cites one durable budget charge.
	CapabilityEpisodeBudgetCharge CapabilityEpisodeReferenceKind = "budget-charge"
	// CapabilityEpisodeToolManual cites one exact tool contract.
	CapabilityEpisodeToolManual CapabilityEpisodeReferenceKind = "tool-manual"
	// CapabilityEpisodePriorAttempt cites one predecessor attempt.
	CapabilityEpisodePriorAttempt CapabilityEpisodeReferenceKind = "prior-attempt"
	// CapabilityEpisodeContext cites one exact context artifact.
	CapabilityEpisodeContext CapabilityEpisodeReferenceKind = "context"
)

func (kind CapabilityEpisodeReferenceKind) valid() bool {
	switch kind {
	case CapabilityEpisodeToolAction, CapabilityEpisodeDecision, CapabilityEpisodeFinalArtifact,
		CapabilityEpisodeInvocationEffect, CapabilityEpisodeObligation, CapabilityEpisodeResolution,
		CapabilityEpisodeWorkspaceClaim, CapabilityEpisodeBudgetCharge, CapabilityEpisodeToolManual,
		CapabilityEpisodePriorAttempt, CapabilityEpisodeContext:
		return true
	default:
		return false
	}
}

// CapabilityEpisodeReference is one immutable identity cited by a trajectory.
type CapabilityEpisodeReference struct {
	Kind CapabilityEpisodeReferenceKind `json:"kind"`
	ID   artifact.ID                    `json:"id"`
}

// CapabilityEpisodeCoverage reports optional source coverage explicitly.
// Counts are checked against References when the projection is constructed or
// reopened.
type CapabilityEpisodeCoverage struct {
	Strategy          bool   `json:"strategy"`
	Environment       bool   `json:"environment"`
	Operation         bool   `json:"operation"`
	ToolActions       uint32 `json:"tool_actions"`
	Decisions         uint32 `json:"decisions"`
	FinalArtifacts    uint32 `json:"final_artifacts"`
	InvocationEffects uint32 `json:"invocation_effects"`
	Obligations       uint32 `json:"obligations"`
	Resolutions       uint32 `json:"resolutions"`
	WorkspaceClaims   uint32 `json:"workspace_claims"`
	BudgetCharges     uint32 `json:"budget_charges"`
	ToolManuals       uint32 `json:"tool_manuals"`
	PriorAttempts     uint32 `json:"prior_attempts"`
	Context           uint32 `json:"context"`
	CostUnits         bool   `json:"cost_units"`
}

// CapabilityEpisodeInput is the narrow immutable boundary supplied by the
// run-record owner after it has loaded and validated the exact attempt and
// trajectory. Attempt-side and trace-side authorities are intentionally kept
// separate here so this package can refuse cross-linking before projection.
type CapabilityEpisodeInput struct {
	Attempt             artifact.ID
	Trajectory          artifact.ID
	AttemptTrajectory   artifact.ID
	AttemptTaskContract artifact.ID
	TraceTaskContract   artifact.ID
	AttemptStrategy     artifact.ID
	TraceStrategy       artifact.ID
	AttemptRecipe       artifact.ID
	TraceRecipe         artifact.ID
	Model               artifact.ID
	Environment         artifact.ID
	Result              artifact.ID
	Request             artifact.ID
	Operation           artifact.ID
	AttemptOutcome      string
	TerminalOutcome     string
	WallNS              uint64
	CostUnits           uint64
	References          []CapabilityEpisodeReference
	Events              []InteractionProtocolEvent
}

// CapabilityEpisode is a bounded derived fact row. It contains source
// identities and aggregate facts only; message and tool-result payloads remain
// exclusively in the run-record trajectory.
type CapabilityEpisode struct {
	Attempt         artifact.ID                  `json:"attempt"`
	Trajectory      artifact.ID                  `json:"trajectory"`
	TaskContract    artifact.ID                  `json:"task_contract"`
	Strategy        artifact.ID                  `json:"strategy,omitzero"`
	GateRecipe      artifact.ID                  `json:"gate_recipe"`
	ExecutionRecipe artifact.ID                  `json:"execution_recipe"`
	Model           artifact.ID                  `json:"model"`
	Environment     artifact.ID                  `json:"environment,omitzero"`
	Result          artifact.ID                  `json:"result"`
	Request         artifact.ID                  `json:"request"`
	Operation       artifact.ID                  `json:"operation,omitzero"`
	AttemptOutcome  string                       `json:"attempt_outcome"`
	Terminal        string                       `json:"terminal"`
	WallNS          uint64                       `json:"wall_ns"`
	CostUnits       uint64                       `json:"cost_units"`
	Events          uint32                       `json:"events"`
	ToolCalls       uint32                       `json:"tool_calls"`
	ToolFailures    uint32                       `json:"tool_failures"`
	RepeatedCalls   uint32                       `json:"repeated_calls"`
	References      []CapabilityEpisodeReference `json:"references,omitempty"`
	Coverage        CapabilityEpisodeCoverage    `json:"coverage"`
}

// CapabilityEpisodeProjection is one content-addressed, rebuildable view over
// exact run-record sources.
type CapabilityEpisodeProjection struct {
	Version   uint16                            `json:"version"`
	Bounds    CapabilityEpisodeProjectionBounds `json:"bounds"`
	SourceSet artifact.ID                       `json:"source_set"`
	Episodes  []CapabilityEpisode               `json:"episodes"`
	ID        artifact.ID                       `json:"-"`
}

type capabilityEpisodeSourceIdentity struct {
	Attempt    artifact.ID `json:"attempt"`
	Trajectory artifact.ID `json:"trajectory"`
}

var capabilityEpisodeProjectionCodec = artifact.JSONDocumentCodec(
	"capability episode projection", artifact.KindProfile,
	CapabilityEpisodeProjectionMediaType, CapabilityEpisodeProjectionSchema,
	canonicalizeCapabilityEpisodeProjection,
	func(value CapabilityEpisodeProjection) artifact.ID { return value.ID },
	func(value *CapabilityEpisodeProjection, id artifact.ID) { value.ID = id },
	cloneCapabilityEpisodeProjection,
)

// CompileCapabilityEpisodeProjection compiles owner-validated source facts into a
// deterministic projection. Input order cannot affect its identity.
// Production callers must enter through runrecord.BuildCapabilityEpisodeProjection.
func CompileCapabilityEpisodeProjection(
	bounds CapabilityEpisodeProjectionBounds,
	inputs []CapabilityEpisodeInput,
) (CapabilityEpisodeProjection, error) {
	if _, err := bounds.Identify(); err != nil || len(inputs) == 0 || uint64(len(inputs)) > uint64(bounds.MaximumEpisodes) {
		return CapabilityEpisodeProjection{}, errors.Join(errors.New("dataset: invalid capability episode projection input"), err)
	}
	episodes := make([]CapabilityEpisode, len(inputs))
	for index, input := range inputs {
		episode, err := newCapabilityEpisode(bounds, input)
		if err != nil {
			return CapabilityEpisodeProjection{}, err
		}
		episodes[index] = episode
	}
	slices.SortFunc(episodes, compareCapabilityEpisodes)
	sources := make([]capabilityEpisodeSourceIdentity, len(episodes))
	seenAttempts := make(map[artifact.ID]struct{}, len(episodes))
	seenTrajectories := make(map[artifact.ID]struct{}, len(episodes))
	for index, episode := range episodes {
		if _, duplicate := seenAttempts[episode.Attempt]; duplicate {
			return CapabilityEpisodeProjection{}, errors.New("dataset: duplicate capability episode authority")
		}
		if _, duplicate := seenTrajectories[episode.Trajectory]; duplicate {
			return CapabilityEpisodeProjection{}, errors.New("dataset: duplicate capability episode authority")
		}
		seenAttempts[episode.Attempt] = struct{}{}
		seenTrajectories[episode.Trajectory] = struct{}{}
		sources[index] = capabilityEpisodeSourceIdentity{Attempt: episode.Attempt, Trajectory: episode.Trajectory}
	}
	sourceSet, err := artifact.JSONID(artifact.KindEvidence, sources)
	if err != nil {
		return CapabilityEpisodeProjection{}, err
	}
	return capabilityEpisodeProjectionCodec.New(CapabilityEpisodeProjection{
		Version: artifact.InitialDocumentVersion, Bounds: bounds, SourceSet: sourceSet, Episodes: episodes,
	})
}

func compareCapabilityEpisodes(left, right CapabilityEpisode) int {
	if order := artifact.CompareID(left.Attempt, right.Attempt); order != 0 {
		return order
	}
	return artifact.CompareID(left.Trajectory, right.Trajectory)
}

func compareCapabilityEpisodeReferences(left, right CapabilityEpisodeReference) int {
	if order := cmp.Compare(left.Kind, right.Kind); order != 0 {
		return order
	}
	return artifact.CompareID(left.ID, right.ID)
}

func newCapabilityEpisode(bounds CapabilityEpisodeProjectionBounds, input CapabilityEpisodeInput) (CapabilityEpisode, error) {
	if input.Attempt.Kind() != artifact.KindEvidence || input.Trajectory.Kind() != artifact.KindEvidence ||
		input.AttemptTrajectory != input.Trajectory || input.AttemptTaskContract != input.TraceTaskContract ||
		input.AttemptTaskContract.Kind() != artifact.KindRecipe || input.AttemptStrategy != input.TraceStrategy ||
		(input.AttemptStrategy.Valid() && input.AttemptStrategy.Kind() != artifact.KindProfile) ||
		input.AttemptRecipe.Kind() != artifact.KindRecipe || input.TraceRecipe.Kind() != artifact.KindRecipe ||
		input.Model.Kind() != artifact.KindModel || (input.Environment.Valid() && input.Environment.Kind() != artifact.KindEvidence) ||
		input.Result.Kind() != artifact.KindEvidence || input.Request.Kind() != artifact.KindEvidence ||
		(input.Operation.Valid() && input.Operation.Kind() != artifact.KindEvidence) || input.WallNS == 0 ||
		!validEpisodeOutcome(input.AttemptOutcome) || !validEpisodeOutcome(input.TerminalOutcome) {
		return CapabilityEpisode{}, errors.New("dataset: cross-linked or invalid capability episode authority")
	}
	if uint64(len(input.Events)) > uint64(bounds.MaximumEventsPerEpisode) ||
		uint64(len(input.References)) > uint64(bounds.MaximumReferencesPerEpisode) {
		return CapabilityEpisode{}, errors.New("dataset: capability episode source exceeds its declared bound")
	}
	facts, err := deriveInteractionProtocolFacts(input.Events)
	if err != nil || facts.Calls > bounds.MaximumCallsPerEpisode {
		return CapabilityEpisode{}, errors.Join(errors.New("dataset: invalid bounded capability episode interaction facts"), err)
	}
	references := slices.Clone(input.References)
	slices.SortFunc(references, compareCapabilityEpisodeReferences)
	coverage := CapabilityEpisodeCoverage{
		Strategy: input.AttemptStrategy.Valid(), Environment: input.Environment.Valid(), Operation: input.Operation.Valid(),
		CostUnits: input.CostUnits != 0,
	}
	for index, reference := range references {
		if !reference.Kind.valid() || !reference.ID.Valid() ||
			index > 0 && reference == references[index-1] {
			return CapabilityEpisode{}, errors.New("dataset: invalid capability episode reference")
		}
		counter, err := episodeCoverageCounter(&coverage, reference.Kind)
		if err != nil || *counter == math.MaxUint32 {
			return CapabilityEpisode{}, errors.Join(errors.New("dataset: capability episode reference coverage overflow"), err)
		}
		*counter++
	}
	return CapabilityEpisode{
		Attempt: input.Attempt, Trajectory: input.Trajectory, TaskContract: input.AttemptTaskContract,
		Strategy: input.AttemptStrategy, GateRecipe: input.AttemptRecipe, ExecutionRecipe: input.TraceRecipe, Model: input.Model,
		Environment: input.Environment, Result: input.Result, Request: input.Request, Operation: input.Operation,
		AttemptOutcome: input.AttemptOutcome, Terminal: input.TerminalOutcome, WallNS: input.WallNS,
		CostUnits: input.CostUnits, Events: uint32(len(input.Events)), ToolCalls: facts.Calls,
		ToolFailures: facts.Failures, RepeatedCalls: facts.Repeated, References: references, Coverage: coverage,
	}, nil
}

func episodeCoverageCounter(coverage *CapabilityEpisodeCoverage, kind CapabilityEpisodeReferenceKind) (*uint32, error) {
	switch kind {
	case CapabilityEpisodeToolAction:
		return &coverage.ToolActions, nil
	case CapabilityEpisodeDecision:
		return &coverage.Decisions, nil
	case CapabilityEpisodeFinalArtifact:
		return &coverage.FinalArtifacts, nil
	case CapabilityEpisodeInvocationEffect:
		return &coverage.InvocationEffects, nil
	case CapabilityEpisodeObligation:
		return &coverage.Obligations, nil
	case CapabilityEpisodeResolution:
		return &coverage.Resolutions, nil
	case CapabilityEpisodeWorkspaceClaim:
		return &coverage.WorkspaceClaims, nil
	case CapabilityEpisodeBudgetCharge:
		return &coverage.BudgetCharges, nil
	case CapabilityEpisodeToolManual:
		return &coverage.ToolManuals, nil
	case CapabilityEpisodePriorAttempt:
		return &coverage.PriorAttempts, nil
	case CapabilityEpisodeContext:
		return &coverage.Context, nil
	default:
		return nil, errors.New("dataset: unknown capability episode reference kind")
	}
}

func validEpisodeOutcome(value string) bool {
	return value != "" && strings.TrimSpace(value) == value && textcheck.Bounded(value, artifact.MaxContentBytes, " /\x00\r\n\t")
}

func canonicalizeCapabilityEpisodeProjection(value *CapabilityEpisodeProjection) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.SourceSet.Kind() != artifact.KindEvidence ||
		len(value.Episodes) == 0 || uint64(len(value.Episodes)) > uint64(value.Bounds.MaximumEpisodes) {
		return errors.New("dataset: invalid capability episode projection")
	}
	if _, err := value.Bounds.Identify(); err != nil {
		return err
	}
	slices.SortFunc(value.Episodes, compareCapabilityEpisodes)
	sources := make([]capabilityEpisodeSourceIdentity, len(value.Episodes))
	seenAttempts := make(map[artifact.ID]struct{}, len(value.Episodes))
	seenTrajectories := make(map[artifact.ID]struct{}, len(value.Episodes))
	for index := range value.Episodes {
		episode := &value.Episodes[index]
		if err := validateCapabilityEpisode(value.Bounds, episode); err != nil {
			return err
		}
		if _, duplicate := seenAttempts[episode.Attempt]; duplicate {
			return errors.New("dataset: duplicate capability episode authority")
		}
		if _, duplicate := seenTrajectories[episode.Trajectory]; duplicate {
			return errors.New("dataset: duplicate capability episode authority")
		}
		seenAttempts[episode.Attempt] = struct{}{}
		seenTrajectories[episode.Trajectory] = struct{}{}
		sources[index] = capabilityEpisodeSourceIdentity{Attempt: episode.Attempt, Trajectory: episode.Trajectory}
	}
	sourceSet, err := artifact.JSONID(artifact.KindEvidence, sources)
	if err != nil || sourceSet != value.SourceSet {
		return errors.Join(errors.New("dataset: capability episode source set differs"), err)
	}
	return nil
}

func validateCapabilityEpisode(bounds CapabilityEpisodeProjectionBounds, episode *CapabilityEpisode) error {
	if episode.Attempt.Kind() != artifact.KindEvidence || episode.Trajectory.Kind() != artifact.KindEvidence ||
		episode.TaskContract.Kind() != artifact.KindRecipe || (episode.Strategy.Valid() && episode.Strategy.Kind() != artifact.KindProfile) ||
		episode.GateRecipe.Kind() != artifact.KindRecipe || episode.ExecutionRecipe.Kind() != artifact.KindRecipe ||
		episode.Model.Kind() != artifact.KindModel ||
		(episode.Environment.Valid() && episode.Environment.Kind() != artifact.KindEvidence) || episode.Result.Kind() != artifact.KindEvidence ||
		episode.Request.Kind() != artifact.KindEvidence || (episode.Operation.Valid() && episode.Operation.Kind() != artifact.KindEvidence) ||
		episode.WallNS == 0 || episode.Events == 0 || episode.Events > bounds.MaximumEventsPerEpisode ||
		episode.ToolCalls > bounds.MaximumCallsPerEpisode || episode.ToolFailures > episode.ToolCalls ||
		episode.RepeatedCalls > episode.ToolCalls || uint64(len(episode.References)) > uint64(bounds.MaximumReferencesPerEpisode) ||
		!validEpisodeOutcome(episode.AttemptOutcome) || !validEpisodeOutcome(episode.Terminal) ||
		episode.Coverage.Strategy != episode.Strategy.Valid() || episode.Coverage.Environment != episode.Environment.Valid() ||
		episode.Coverage.Operation != episode.Operation.Valid() || episode.Coverage.CostUnits != (episode.CostUnits != 0) {
		return errors.New("dataset: invalid capability episode fact")
	}
	slices.SortFunc(episode.References, compareCapabilityEpisodeReferences)
	var coverage CapabilityEpisodeCoverage
	coverage.Strategy, coverage.Environment, coverage.Operation = episode.Strategy.Valid(), episode.Environment.Valid(), episode.Operation.Valid()
	coverage.CostUnits = episode.CostUnits != 0
	for index, reference := range episode.References {
		if !reference.Kind.valid() || !reference.ID.Valid() || index > 0 && reference == episode.References[index-1] {
			return errors.New("dataset: invalid capability episode reference")
		}
		counter, err := episodeCoverageCounter(&coverage, reference.Kind)
		if err != nil || *counter == math.MaxUint32 {
			return errors.Join(errors.New("dataset: capability episode coverage overflow"), err)
		}
		*counter++
	}
	if coverage != episode.Coverage {
		return errors.New("dataset: capability episode coverage differs from references")
	}
	return nil
}

// LoadCapabilityEpisodeProjection loads a canonical projection without
// re-deriving its run-record authorities. Production callers must use the
// verified runrecord loader.
func LoadCapabilityEpisodeProjection(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (CapabilityEpisodeProjection, error) {
	return capabilityEpisodeProjectionCodec.Require(ctx, reader, id)
}

// Content returns the canonical projection document.
func (value CapabilityEpisodeProjection) Content() (artifact.Content, error) {
	return capabilityEpisodeProjectionCodec.Content(value)
}

// ValidateIdentity proves the projection still matches its immutable facts.
func (value CapabilityEpisodeProjection) ValidateIdentity() error {
	return capabilityEpisodeProjectionCodec.ValidateIdentity(value)
}

// Lineage cites only authoritative attempt and trajectory documents.
func (value CapabilityEpisodeProjection) Lineage() []artifact.Lineage {
	parents := make([]artifact.ID, 0, len(value.Episodes)*2)
	for _, episode := range value.Episodes {
		parents = append(parents, episode.Attempt, episode.Trajectory)
	}
	return artifact.DependencyLineage(value.ID, parents...)
}

// Batch prepares one atomic projection publication without an active alias.
// Rebuildability comes from exact source identities, not mutable projection
// state.
func (value CapabilityEpisodeProjection) Batch(key string) (artifact.Batch, error) {
	content, err := value.Content()
	if err != nil {
		return artifact.Batch{}, err
	}
	return artifact.NewDocumentBatch(key, []artifact.Content{content}, value.Lineage(), nil)
}

func cloneCapabilityEpisodeProjection(value CapabilityEpisodeProjection) CapabilityEpisodeProjection {
	value.Episodes = slices.Clone(value.Episodes)
	for index := range value.Episodes {
		value.Episodes[index].References = slices.Clone(value.Episodes[index].References)
	}
	return value
}
