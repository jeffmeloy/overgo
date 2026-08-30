package dataset

import (
	"cmp"
	"context"
	"errors"
	"maps"
	"math"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
)

const (
	// InteractionArcMediaType identifies an immutable interaction-arc document.
	InteractionArcMediaType = "application/vnd.overgo.interaction-arc+json"
	// InteractionArcSchema identifies the interaction-arc wire contract.
	InteractionArcSchema = "overgo/interaction-arc/v1"
	// InteractionArcProjectionMediaType identifies an interaction-arc projection.
	InteractionArcProjectionMediaType = "application/vnd.overgo.interaction-arc-projection+json"
	// InteractionArcProjectionSchema identifies the interaction-arc projection contract.
	InteractionArcProjectionSchema = "overgo/interaction-arc-projection/v1"
	// InteractionArcMeasurementPolicyMediaType identifies an arc cost policy.
	InteractionArcMeasurementPolicyMediaType = "application/vnd.overgo.interaction-arc-measurement-policy+json"
	// InteractionArcMeasurementPolicySchema identifies the arc cost policy contract.
	InteractionArcMeasurementPolicySchema = "overgo/interaction-arc-measurement-policy/v1"
	// InteractionArcMeasurementMediaType identifies immutable arc cost evidence.
	InteractionArcMeasurementMediaType = "application/vnd.overgo.interaction-arc-measurement+json"
	// InteractionArcMeasurementSchema identifies the arc cost evidence contract.
	InteractionArcMeasurementSchema = "overgo/interaction-arc-measurement/v1"
	// InteractionArcSelectionMediaType identifies one bounded arc admission.
	InteractionArcSelectionMediaType = "application/vnd.overgo.interaction-arc-selection+json"
	// InteractionArcSelectionSchema identifies the bounded arc admission contract.
	InteractionArcSelectionSchema = "overgo/interaction-arc-selection/v1"
)

// InteractionProtocolEventKind is the structural vocabulary supplied by the
// authoritative run-record adapter.
type InteractionProtocolEventKind string

const (
	// InteractionProtocolRequest marks an input request event.
	InteractionProtocolRequest InteractionProtocolEventKind = "request"
	// InteractionProtocolOutput marks a model output event.
	InteractionProtocolOutput InteractionProtocolEventKind = "output"
	// InteractionProtocolToolCall marks a tool invocation event.
	InteractionProtocolToolCall InteractionProtocolEventKind = "tool-call"
	// InteractionProtocolToolResult marks the matching tool result event.
	InteractionProtocolToolResult InteractionProtocolEventKind = "tool-result"
)

func (kind InteractionProtocolEventKind) valid() bool {
	return kind == InteractionProtocolRequest || kind == InteractionProtocolOutput ||
		kind == InteractionProtocolToolCall || kind == InteractionProtocolToolResult
}

// InteractionCallReference keeps a transient call identity for pairing and an
// opaque fingerprint for exact repetition. Raw arguments are never projected.
type InteractionCallReference struct {
	ID          string
	Fingerprint artifact.ID
}

type interactionCallFingerprint struct {
	Type      string      `json:"type"`
	Name      string      `json:"name"`
	Manual    artifact.ID `json:"manual,omitzero"`
	Arguments string      `json:"arguments"`
}

// NewInteractionCallReference validates a call and derives its opaque fingerprint.
func NewInteractionCallReference(id, callType, name string, manual artifact.ID, arguments string) (InteractionCallReference, error) {
	if id == "" || len(id) > artifact.MaxContentBytes || callType != "function" ||
		len(name) > artifact.MaxContentBytes || len(arguments) > artifact.MaxContentBytes ||
		(manual.Valid() && manual.Kind() != artifact.KindRecipe) {
		return InteractionCallReference{}, errors.New("dataset: invalid interaction call reference")
	}
	fingerprint, err := artifact.JSONID(artifact.KindEvidence, interactionCallFingerprint{
		Type: callType, Name: name, Manual: manual, Arguments: arguments,
	})
	return InteractionCallReference{ID: id, Fingerprint: fingerprint}, err
}

// InteractionProtocolEvent carries protocol structure only.
type InteractionProtocolEvent struct {
	Sequence     uint32
	Kind         InteractionProtocolEventKind
	Branch       string
	Calls        []InteractionCallReference
	ResultCallID string
	ResultFailed bool
}

// InteractionArcProjectionInput is the narrow structural output of runrecord.
type InteractionArcProjectionInput struct {
	Trace      artifact.ID
	Events     []InteractionProtocolEvent
	References []CapabilityEpisodeReference
}

// InteractionArcProjectionBounds bounds derived expansion, including shared
// events repeated into several branch-visible arcs.
type InteractionArcProjectionBounds struct {
	MaximumBranches           uint32 `json:"maximum_branches"`
	MaximumArcs               uint32 `json:"maximum_arcs"`
	MaximumProjectedSequences uint32 `json:"maximum_projected_sequences"`
	MaximumToolPairs          uint32 `json:"maximum_tool_pairs"`
	MaximumReferences         uint32 `json:"maximum_references"`
}

// Identify validates and identifies one exact interaction-arc projection bound.
func (bounds InteractionArcProjectionBounds) Identify() (artifact.ID, error) {
	if bounds.MaximumBranches == 0 || bounds.MaximumArcs == 0 || bounds.MaximumProjectedSequences == 0 ||
		bounds.MaximumToolPairs == 0 || bounds.MaximumReferences == 0 {
		return artifact.ID{}, errors.New("dataset: interaction arc projection requires every bound")
	}
	return artifact.JSONID(artifact.KindProfile, bounds)
}

// InteractionToolPairReference cites one complete call/result pair by source
// sequence and call ordinal.
type InteractionToolPairReference struct {
	CallSequence   uint32      `json:"call_sequence"`
	CallOrdinal    uint32      `json:"call_ordinal"`
	ResultSequence uint32      `json:"result_sequence"`
	Fingerprint    artifact.ID `json:"fingerprint"`
	Failed         bool        `json:"failed"`
}

// InteractionArc is a resolvable immutable evidence document. It contains no
// message or result payload.
type InteractionArc struct {
	Version         uint16                         `json:"version"`
	Trace           artifact.ID                    `json:"trace"`
	Branch          string                         `json:"branch,omitzero"`
	RequestSequence uint32                         `json:"request_sequence"`
	LastSequence    uint32                         `json:"last_sequence"`
	Sequences       []uint32                       `json:"sequences"`
	ToolPairs       []InteractionToolPairReference `json:"tool_pairs,omitempty"`
	References      []CapabilityEpisodeReference   `json:"references,omitempty"`
	ID              artifact.ID                    `json:"-"`
}

// InteractionArcEntry keeps deterministic branch ordering while arc payloads
// remain independently resolvable.
type InteractionArcEntry struct {
	Arc             artifact.ID `json:"arc"`
	Branch          string      `json:"branch,omitzero"`
	RequestSequence uint32      `json:"request_sequence"`
	LastSequence    uint32      `json:"last_sequence"`
}

// InteractionArcProjection is the rebuildable index over exact arc evidence.
type InteractionArcProjection struct {
	Version  uint16                         `json:"version"`
	Trace    artifact.ID                    `json:"trace"`
	Bounds   InteractionArcProjectionBounds `json:"bounds"`
	Branches []string                       `json:"branches"`
	Arcs     []InteractionArcEntry          `json:"arcs"`
	ID       artifact.ID                    `json:"-"`
}

var interactionArcCodec = artifact.JSONDocumentCodec(
	"interaction arc", artifact.KindEvidence, InteractionArcMediaType, InteractionArcSchema,
	canonicalizeInteractionArc,
	func(value InteractionArc) artifact.ID { return value.ID },
	func(value *InteractionArc, id artifact.ID) { value.ID = id },
	cloneInteractionArc,
)

var interactionArcProjectionCodec = artifact.JSONDocumentCodec(
	"interaction arc projection", artifact.KindProfile,
	InteractionArcProjectionMediaType, InteractionArcProjectionSchema,
	canonicalizeInteractionArcProjection,
	func(value InteractionArcProjection) artifact.ID { return value.ID },
	func(value *InteractionArcProjection, id artifact.ID) { value.ID = id },
	func(value InteractionArcProjection) InteractionArcProjection {
		value.Branches = slices.Clone(value.Branches)
		value.Arcs = slices.Clone(value.Arcs)
		return value
	},
)

type interactionProtocolFacts struct {
	Calls    uint32
	Failures uint32
	Repeated uint32
}

type pendingInteractionCall struct {
	sequence    uint32
	ordinal     uint32
	fingerprint artifact.ID
}

type interactionArcBudget struct {
	arcs       uint32
	sequences  uint32
	pairs      uint32
	references uint32
}

// CompileInteractionArcProjection compiles owner-validated structural input.
// Production callers must enter through runrecord.BuildInteractionArcProjection.
func CompileInteractionArcProjection(
	input InteractionArcProjectionInput,
	bounds InteractionArcProjectionBounds,
) (InteractionArcProjection, []InteractionArc, error) {
	if input.Trace.Kind() != artifact.KindEvidence || len(input.Events) == 0 {
		return InteractionArcProjection{}, nil, errors.New("dataset: invalid interaction arc projection input")
	}
	if _, err := bounds.Identify(); err != nil {
		return InteractionArcProjection{}, nil, err
	}
	if _, err := deriveInteractionProtocolFacts(input.Events); err != nil {
		return InteractionArcProjection{}, nil, err
	}
	references, err := canonicalInteractionReferences(input.References)
	if err != nil {
		return InteractionArcProjection{}, nil, err
	}
	branches, err := interactionBranches(input.Events, bounds.MaximumBranches)
	if err != nil {
		return InteractionArcProjection{}, nil, err
	}
	budget := interactionArcBudget{}
	arcs := make([]InteractionArc, 0)
	for _, branch := range branches {
		branchArcs, deriveErr := deriveBranchInteractionArcs(input.Trace, branch, input.Events, references, bounds, &budget)
		if deriveErr != nil {
			return InteractionArcProjection{}, nil, deriveErr
		}
		arcs = append(arcs, branchArcs...)
	}
	entries := make([]InteractionArcEntry, len(arcs))
	for index, arc := range arcs {
		entries[index] = InteractionArcEntry{
			Arc: arc.ID, Branch: arc.Branch, RequestSequence: arc.RequestSequence, LastSequence: arc.LastSequence,
		}
	}
	projection, err := interactionArcProjectionCodec.New(InteractionArcProjection{
		Version: artifact.InitialDocumentVersion, Trace: input.Trace, Bounds: bounds, Branches: branches, Arcs: entries,
	})
	return projection, arcs, err
}

func deriveInteractionProtocolFacts(events []InteractionProtocolEvent) (interactionProtocolFacts, error) {
	if len(events) == 0 || uint64(len(events)) > uint64(math.MaxUint32) {
		return interactionProtocolFacts{}, errors.New("dataset: invalid interaction protocol event population")
	}
	issued := make(map[string]InteractionCallReference)
	resolved := make(map[string]struct{})
	fingerprints := make(map[artifact.ID]uint32)
	var facts interactionProtocolFacts
	for index, event := range events {
		if event.Sequence != uint32(index+1) || !event.Kind.valid() ||
			(event.Branch != "" && (strings.TrimSpace(event.Branch) != event.Branch ||
				!textcheck.Bounded(event.Branch, artifact.MaxContentBytes, "\x00\r\n\t"))) {
			return interactionProtocolFacts{}, errors.New("dataset: invalid interaction protocol event")
		}
		switch event.Kind {
		case InteractionProtocolToolCall:
			if len(event.Calls) == 0 || event.ResultCallID != "" || event.ResultFailed {
				return interactionProtocolFacts{}, errors.New("dataset: invalid interaction tool-call event")
			}
			for _, call := range event.Calls {
				if call.ID == "" || call.Fingerprint.Kind() != artifact.KindEvidence {
					return interactionProtocolFacts{}, errors.New("dataset: invalid interaction call identity")
				}
				if _, duplicate := issued[call.ID]; duplicate {
					return interactionProtocolFacts{}, errors.New("dataset: duplicate interaction call identity")
				}
				issued[call.ID] = call
				if fingerprints[call.Fingerprint] > 0 {
					facts.Repeated++
				}
				fingerprints[call.Fingerprint]++
				facts.Calls++
			}
		case InteractionProtocolToolResult:
			if len(event.Calls) != 0 || event.ResultCallID == "" {
				return interactionProtocolFacts{}, errors.New("dataset: invalid interaction tool-result event")
			}
			if _, found := issued[event.ResultCallID]; !found {
				return interactionProtocolFacts{}, errors.New("dataset: interaction result precedes or lacks its call")
			}
			if _, duplicate := resolved[event.ResultCallID]; duplicate {
				return interactionProtocolFacts{}, errors.New("dataset: duplicate interaction tool result")
			}
			resolved[event.ResultCallID] = struct{}{}
			if event.ResultFailed {
				facts.Failures++
			}
		default:
			if len(event.Calls) != 0 || event.ResultCallID != "" || event.ResultFailed {
				return interactionProtocolFacts{}, errors.New("dataset: non-tool event contains tool protocol facts")
			}
		}
	}
	if len(issued) != len(resolved) {
		return interactionProtocolFacts{}, errors.New("dataset: interaction call lacks exactly one result")
	}
	return facts, nil
}

func interactionBranches(events []InteractionProtocolEvent, maximum uint32) ([]string, error) {
	seen := map[string]struct{}{}
	for _, event := range events {
		if event.Branch == "" {
			continue
		}
		if _, found := seen[event.Branch]; found {
			continue
		}
		if uint64(len(seen)) == uint64(maximum) {
			return nil, errors.New("dataset: interaction branch population exceeds its bound")
		}
		seen[event.Branch] = struct{}{}
	}
	if len(seen) == 0 {
		return []string{""}, nil
	}
	return slices.Sorted(maps.Keys(seen)), nil
}

func deriveBranchInteractionArcs(
	trace artifact.ID,
	branch string,
	events []InteractionProtocolEvent,
	references []CapabilityEpisodeReference,
	bounds InteractionArcProjectionBounds,
	budget *interactionArcBudget,
) ([]InteractionArc, error) {
	var result []InteractionArc
	var current *InteractionArc
	pending := map[string]pendingInteractionCall{}
	appendSequence := func(sequence uint32) error {
		if budget.sequences == bounds.MaximumProjectedSequences {
			return errors.New("dataset: projected interaction sequences exceed their bound")
		}
		budget.sequences++
		current.Sequences = append(current.Sequences, sequence)
		current.LastSequence = sequence
		return nil
	}
	finish := func() error {
		if current == nil {
			return nil
		}
		if len(pending) != 0 {
			return errors.New("dataset: interaction arc would split a call/result pair")
		}
		if len(current.Sequences) < 2 {
			return errors.New("dataset: interaction request lacks a visible response")
		}
		if budget.arcs == bounds.MaximumArcs || uint64(len(references)) > uint64(bounds.MaximumReferences-budget.references) {
			return errors.New("dataset: projected interaction arcs or references exceed their bound")
		}
		budget.arcs++
		budget.references += uint32(len(references))
		identified, err := interactionArcCodec.New(*current)
		if err != nil {
			return err
		}
		result = append(result, identified)
		current = nil
		return nil
	}
	for _, event := range events {
		if event.Branch != "" && event.Branch != branch {
			continue
		}
		if event.Kind == InteractionProtocolRequest {
			if err := finish(); err != nil {
				return nil, err
			}
			pending = map[string]pendingInteractionCall{}
			current = &InteractionArc{
				Version: artifact.InitialDocumentVersion, Trace: trace, Branch: branch,
				RequestSequence: event.Sequence, References: slices.Clone(references),
			}
			if err := appendSequence(event.Sequence); err != nil {
				return nil, err
			}
			continue
		}
		if current == nil {
			return nil, errors.New("dataset: interaction response lacks a visible request")
		}
		if err := appendSequence(event.Sequence); err != nil {
			return nil, err
		}
		switch event.Kind {
		case InteractionProtocolToolCall:
			for ordinal, call := range event.Calls {
				pending[call.ID] = pendingInteractionCall{
					sequence: event.Sequence, ordinal: uint32(ordinal), fingerprint: call.Fingerprint,
				}
			}
		case InteractionProtocolToolResult:
			call, found := pending[event.ResultCallID]
			if !found {
				return nil, errors.New("dataset: interaction result crosses an arc or branch boundary")
			}
			if budget.pairs == bounds.MaximumToolPairs {
				return nil, errors.New("dataset: projected tool pairs exceed their bound")
			}
			budget.pairs++
			current.ToolPairs = append(current.ToolPairs, InteractionToolPairReference{
				CallSequence: call.sequence, CallOrdinal: call.ordinal, ResultSequence: event.Sequence,
				Fingerprint: call.fingerprint, Failed: event.ResultFailed,
			})
			delete(pending, event.ResultCallID)
		}
	}
	if err := finish(); err != nil {
		return nil, err
	}
	if len(result) == 0 {
		return nil, errors.New("dataset: branch contains no complete interaction arc")
	}
	return result, nil
}

func compareInteractionToolPairs(left, right InteractionToolPairReference) int {
	if order := cmp.Compare(left.CallSequence, right.CallSequence); order != 0 {
		return order
	}
	return cmp.Compare(left.CallOrdinal, right.CallOrdinal)
}

func compareInteractionArcEntries(left, right InteractionArcEntry) int {
	if order := cmp.Compare(left.Branch, right.Branch); order != 0 {
		return order
	}
	return cmp.Compare(left.RequestSequence, right.RequestSequence)
}

func canonicalInteractionReferences(values []CapabilityEpisodeReference) ([]CapabilityEpisodeReference, error) {
	values = slices.Clone(values)
	slices.SortFunc(values, compareCapabilityEpisodeReferences)
	for index, value := range values {
		if !value.Kind.valid() || !value.ID.Valid() || index > 0 && value == values[index-1] {
			return nil, errors.New("dataset: invalid interaction arc reference closure")
		}
	}
	return values, nil
}

func canonicalizeInteractionArc(value *InteractionArc) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Trace.Kind() != artifact.KindEvidence ||
		value.RequestSequence == 0 || len(value.Sequences) < 2 || value.Sequences[0] != value.RequestSequence ||
		value.LastSequence != value.Sequences[len(value.Sequences)-1] ||
		(value.Branch != "" && !textcheck.Bounded(value.Branch, artifact.MaxContentBytes, "\x00\r\n\t")) {
		return errors.New("dataset: invalid interaction arc")
	}
	for index, sequence := range value.Sequences {
		if sequence == 0 || index > 0 && sequence <= value.Sequences[index-1] {
			return errors.New("dataset: interaction arc sequences are not ordered")
		}
	}
	slices.SortFunc(value.ToolPairs, compareInteractionToolPairs)
	callSet := map[[2]uint32]struct{}{}
	resultSet := map[uint32]struct{}{}
	for _, pair := range value.ToolPairs {
		call := [2]uint32{pair.CallSequence, pair.CallOrdinal}
		if pair.Fingerprint.Kind() != artifact.KindEvidence || pair.CallSequence >= pair.ResultSequence ||
			!slices.Contains(value.Sequences, pair.CallSequence) || !slices.Contains(value.Sequences, pair.ResultSequence) {
			return errors.New("dataset: interaction arc contains a split pair")
		}
		if _, duplicate := callSet[call]; duplicate {
			return errors.New("dataset: duplicate interaction arc call")
		}
		if _, duplicate := resultSet[pair.ResultSequence]; duplicate {
			return errors.New("dataset: duplicate interaction arc result")
		}
		callSet[call], resultSet[pair.ResultSequence] = struct{}{}, struct{}{}
	}
	references, err := canonicalInteractionReferences(value.References)
	if err != nil {
		return err
	}
	value.References = references
	return nil
}

func canonicalizeInteractionArcProjection(value *InteractionArcProjection) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Trace.Kind() != artifact.KindEvidence ||
		len(value.Branches) == 0 || len(value.Arcs) == 0 || uint64(len(value.Branches)) > uint64(value.Bounds.MaximumBranches) ||
		uint64(len(value.Arcs)) > uint64(value.Bounds.MaximumArcs) {
		return errors.New("dataset: invalid interaction arc projection")
	}
	if _, err := value.Bounds.Identify(); err != nil {
		return err
	}
	slices.Sort(value.Branches)
	if len(slices.Compact(slices.Clone(value.Branches))) != len(value.Branches) || len(value.Branches) > 1 && value.Branches[0] == "" {
		return errors.New("dataset: invalid interaction arc branches")
	}
	slices.SortFunc(value.Arcs, compareInteractionArcEntries)
	seenArcs := map[artifact.ID]struct{}{}
	branchCounts := map[string]uint32{}
	branchLast := map[string]uint32{}
	for _, entry := range value.Arcs {
		if entry.Arc.Kind() != artifact.KindEvidence || !slices.Contains(value.Branches, entry.Branch) ||
			entry.RequestSequence == 0 || entry.LastSequence <= entry.RequestSequence ||
			branchLast[entry.Branch] != 0 && entry.RequestSequence <= branchLast[entry.Branch] {
			return errors.New("dataset: invalid interaction arc projection entry")
		}
		if _, duplicate := seenArcs[entry.Arc]; duplicate {
			return errors.New("dataset: duplicate interaction arc projection entry")
		}
		seenArcs[entry.Arc] = struct{}{}
		branchCounts[entry.Branch]++
		branchLast[entry.Branch] = entry.LastSequence
	}
	for _, branch := range value.Branches {
		if branchCounts[branch] == 0 {
			return errors.New("dataset: interaction arc branch has no arcs")
		}
	}
	return nil
}

func cloneInteractionArc(value InteractionArc) InteractionArc {
	value.Sequences = slices.Clone(value.Sequences)
	value.ToolPairs = slices.Clone(value.ToolPairs)
	value.References = slices.Clone(value.References)
	return value
}

// Content returns the canonical interaction-arc document.
func (value InteractionArc) Content() (artifact.Content, error) {
	return interactionArcCodec.Content(value)
}

// Lineage binds an arc to its trace and exact referenced evidence.
func (value InteractionArc) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.Trace}
	for _, reference := range value.References {
		parents = append(parents, reference.ID)
	}
	slices.SortFunc(parents, artifact.CompareID)
	parents = slices.Compact(parents)
	return artifact.DependencyLineage(value.ID, parents...)
}

// Lineage binds a projection to its trace and every materialized arc.
func (value InteractionArcProjection) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.Trace}
	for _, entry := range value.Arcs {
		parents = append(parents, entry.Arc)
	}
	slices.SortFunc(parents, artifact.CompareID)
	parents = slices.Compact(parents)
	return artifact.DependencyLineage(value.ID, parents...)
}

// Batch atomically stores every resolvable arc and its projection.
func (value InteractionArcProjection) Batch(key string, arcs []InteractionArc) (artifact.Batch, error) {
	if err := interactionArcProjectionCodec.ValidateIdentity(value); err != nil || len(arcs) != len(value.Arcs) {
		return artifact.Batch{}, errors.Join(errors.New("dataset: invalid interaction arc publication"), err)
	}
	byID := make(map[artifact.ID]InteractionArc, len(arcs))
	for _, arc := range arcs {
		if err := interactionArcCodec.ValidateIdentity(arc); err != nil || arc.Trace != value.Trace {
			return artifact.Batch{}, errors.Join(errors.New("dataset: invalid interaction arc publication member"), err)
		}
		if _, duplicate := byID[arc.ID]; duplicate {
			return artifact.Batch{}, errors.New("dataset: duplicate interaction arc publication member")
		}
		byID[arc.ID] = arc
	}
	contents := make([]artifact.Content, 0, len(arcs)+1)
	lineage := make([]artifact.Lineage, 0)
	var sequences, pairs, references uint64
	for _, entry := range value.Arcs {
		arc, found := byID[entry.Arc]
		if !found || arc.Branch != entry.Branch || arc.RequestSequence != entry.RequestSequence || arc.LastSequence != entry.LastSequence {
			return artifact.Batch{}, errors.New("dataset: interaction arc projection entry differs from its evidence")
		}
		content, err := arc.Content()
		if err != nil {
			return artifact.Batch{}, err
		}
		contents = append(contents, content)
		lineage = append(lineage, arc.Lineage()...)
		sequences += uint64(len(arc.Sequences))
		pairs += uint64(len(arc.ToolPairs))
		references += uint64(len(arc.References))
	}
	if sequences > uint64(value.Bounds.MaximumProjectedSequences) || pairs > uint64(value.Bounds.MaximumToolPairs) ||
		references > uint64(value.Bounds.MaximumReferences) {
		return artifact.Batch{}, errors.New("dataset: interaction arc publication exceeds projection bounds")
	}
	projectionContent, err := interactionArcProjectionCodec.Content(value)
	if err != nil {
		return artifact.Batch{}, err
	}
	contents = append(contents, projectionContent)
	lineage = append(lineage, value.Lineage()...)
	return artifact.NewDocumentBatch(key, contents, lineage, nil)
}

// LoadInteractionArc loads one arc for verification by the runrecord owner.
func LoadInteractionArc(ctx context.Context, reader artifact.Reader, id artifact.ID) (InteractionArc, error) {
	return interactionArcCodec.Require(ctx, reader, id)
}

// LoadInteractionArcProjection loads one projection with exact stored lineage.
func LoadInteractionArcProjection(ctx context.Context, reader artifact.Reader, id artifact.ID) (InteractionArcProjection, error) {
	if ctx == nil {
		return InteractionArcProjection{}, errors.New("dataset: interaction arc projection context is absent")
	}
	if id.Kind() != artifact.KindProfile {
		return InteractionArcProjection{}, errors.New("dataset: invalid interaction arc projection identity")
	}
	return interactionArcProjectionCodec.RequireExactLineage(ctx, reader, id, InteractionArcProjection.Lineage)
}

// InteractionArcMeasurementPolicy binds costs to exact tokenizer and counter identities.
type InteractionArcMeasurementPolicy struct {
	Version   uint16      `json:"version"`
	Tokenizer artifact.ID `json:"tokenizer"`
	Counter   artifact.ID `json:"counter"`
	ID        artifact.ID `json:"-"`
}

var interactionArcMeasurementPolicyCodec = artifact.JSONDocumentCodec(
	"interaction arc measurement policy", artifact.KindProfile,
	InteractionArcMeasurementPolicyMediaType, InteractionArcMeasurementPolicySchema,
	func(value *InteractionArcMeasurementPolicy) error {
		if value == nil || value.Version != artifact.InitialDocumentVersion || value.Tokenizer.Kind() != artifact.KindTokenizer ||
			value.Counter.Kind() != artifact.KindProfile {
			return errors.New("dataset: invalid interaction arc measurement policy")
		}
		return nil
	},
	func(value InteractionArcMeasurementPolicy) artifact.ID { return value.ID },
	func(value *InteractionArcMeasurementPolicy, id artifact.ID) { value.ID = id }, nil,
)

// NewInteractionArcMeasurementPolicy identifies the exact tokenizer and counter policy.
func NewInteractionArcMeasurementPolicy(tokenizer, counter artifact.ID) (InteractionArcMeasurementPolicy, error) {
	return interactionArcMeasurementPolicyCodec.NewPrepared(
		InteractionArcMeasurementPolicy{Tokenizer: tokenizer, Counter: counter},
		func(value *InteractionArcMeasurementPolicy) { value.Version = artifact.InitialDocumentVersion },
	)
}

// LoadInteractionArcMeasurementPolicy loads one policy with exact stored lineage.
func LoadInteractionArcMeasurementPolicy(ctx context.Context, reader artifact.Reader, id artifact.ID) (InteractionArcMeasurementPolicy, error) {
	if id.Kind() != artifact.KindProfile {
		return InteractionArcMeasurementPolicy{}, errors.New("dataset: invalid interaction arc measurement policy identity")
	}
	return interactionArcMeasurementPolicyCodec.RequireExactLineage(
		ctx, reader, id, InteractionArcMeasurementPolicy.Lineage,
	)
}

// Lineage binds a measurement policy to its tokenizer and counter authorities.
func (value InteractionArcMeasurementPolicy) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, value.Tokenizer, value.Counter)
}

// Batch publishes one identified measurement policy and its lineage.
func (value InteractionArcMeasurementPolicy) Batch(key string) (artifact.Batch, error) {
	if value.ID.Kind() != artifact.KindProfile {
		return artifact.Batch{}, errors.New("dataset: unidentified interaction arc measurement policy")
	}
	return interactionArcMeasurementPolicyCodec.Batch(key, value, value.Lineage(), nil)
}

// InteractionArcMeasurement is immutable cost evidence for one exact arc.
type InteractionArcMeasurement struct {
	Version    uint16        `json:"version"`
	Arc        artifact.ID   `json:"arc"`
	Policy     artifact.ID   `json:"policy"`
	Transcript artifact.ID   `json:"transcript"`
	Context    []artifact.ID `json:"context,omitempty"`
	Tokens     uint64        `json:"tokens"`
	Bytes      uint64        `json:"bytes"`
	ID         artifact.ID   `json:"-"`
}

var interactionArcMeasurementCodec = artifact.JSONDocumentCodec(
	"interaction arc measurement", artifact.KindEvidence,
	InteractionArcMeasurementMediaType, InteractionArcMeasurementSchema,
	func(value *InteractionArcMeasurement) error {
		if value == nil || value.Version != artifact.InitialDocumentVersion || value.Arc.Kind() != artifact.KindEvidence ||
			value.Policy.Kind() != artifact.KindProfile || value.Transcript.Kind() != artifact.KindEvidence ||
			value.Tokens == 0 || value.Bytes == 0 {
			return errors.New("dataset: invalid interaction arc measurement")
		}
		slices.SortFunc(value.Context, artifact.CompareID)
		if slices.ContainsFunc(value.Context, func(id artifact.ID) bool { return !id.Valid() }) ||
			len(slices.Compact(slices.Clone(value.Context))) != len(value.Context) {
			return errors.New("dataset: invalid interaction arc measurement context")
		}
		return nil
	},
	func(value InteractionArcMeasurement) artifact.ID { return value.ID },
	func(value *InteractionArcMeasurement, id artifact.ID) { value.ID = id },
	func(value InteractionArcMeasurement) InteractionArcMeasurement {
		value.Context = slices.Clone(value.Context)
		return value
	},
)

// NewInteractionArcMeasurement identifies one owner-derived cost result. It is
// a low-level constructor: production callers enter through runrecord, which
// reloads the immutable transcript and context before supplying the counts.
func NewInteractionArcMeasurement(
	arc, policy, transcript artifact.ID,
	context []artifact.ID,
	tokens, bytes uint64,
) (InteractionArcMeasurement, error) {
	return interactionArcMeasurementCodec.NewPrepared(
		InteractionArcMeasurement{
			Arc: arc, Policy: policy, Transcript: transcript, Context: slices.Clone(context), Tokens: tokens, Bytes: bytes,
		},
		func(value *InteractionArcMeasurement) { value.Version = artifact.InitialDocumentVersion },
	)
}

// LoadInteractionArcMeasurement loads one immutable arc cost record.
func LoadInteractionArcMeasurement(ctx context.Context, reader artifact.Reader, id artifact.ID) (InteractionArcMeasurement, error) {
	if ctx == nil || reader == nil || id.Kind() != artifact.KindEvidence {
		return InteractionArcMeasurement{}, errors.New("dataset: invalid interaction arc measurement load")
	}
	return interactionArcMeasurementCodec.Require(ctx, reader, id)
}

// Lineage binds an arc measurement to its arc, policy, transcript, and context.
func (value InteractionArcMeasurement) Lineage() []artifact.Lineage {
	parents := slices.Concat([]artifact.ID{value.Arc, value.Policy, value.Transcript}, value.Context)
	slices.SortFunc(parents, artifact.CompareID)
	return artifact.DependencyLineage(value.ID, slices.Compact(parents)...)
}

// InteractionArcSelection is a single bounded admission. It deliberately has
// no continuation cursor, so omitted arcs cannot be concatenated under the same budget.
type InteractionArcSelection struct {
	Version      uint16                       `json:"version"`
	Head         artifact.CommitID            `json:"head"`
	Projection   artifact.ID                  `json:"projection"`
	Branch       string                       `json:"branch,omitzero"`
	Bounds       InteractionSelectionBounds   `json:"bounds"`
	Policy       artifact.ID                  `json:"policy"`
	Measurements []artifact.ID                `json:"measurements"`
	SourceSet    artifact.ID                  `json:"source_set"`
	Sources      []InteractionSelectionSource `json:"sources"`
	Usage        InteractionSelectionUsage    `json:"usage"`
	Population   uint32                       `json:"population"`
	Truncated    bool                         `json:"truncated"`
	ID           artifact.ID                  `json:"-"`
}

var interactionArcSelectionCodec = artifact.JSONDocumentCodec(
	"interaction arc selection", artifact.KindEvidence,
	InteractionArcSelectionMediaType, InteractionArcSelectionSchema,
	canonicalizeInteractionArcSelection,
	func(value InteractionArcSelection) artifact.ID { return value.ID },
	func(value *InteractionArcSelection, id artifact.ID) { value.ID = id },
	func(value InteractionArcSelection) InteractionArcSelection {
		value.Measurements = slices.Clone(value.Measurements)
		value.Sources = slices.Clone(value.Sources)
		return value
	},
)

// SelectInteractionArcsOnce is the low-level, non-resumable selector used by
// the stable-head runrecord wrapper.
func SelectInteractionArcsOnce(
	bounds InteractionSelectionBounds,
	head artifact.CommitID,
	projection InteractionArcProjection,
	arcs []InteractionArc,
	branch string,
	measurements []InteractionArcMeasurement,
) (InteractionArcSelection, error) {
	if err := interactionArcProjectionCodec.ValidateIdentity(projection); err != nil ||
		!slices.Contains(projection.Branches, branch) || len(arcs) != len(projection.Arcs) {
		return InteractionArcSelection{}, errors.Join(errors.New("dataset: invalid interaction arc selection input"), err)
	}
	arcByID := make(map[artifact.ID]InteractionArc, len(arcs))
	for _, arc := range arcs {
		if err := interactionArcCodec.ValidateIdentity(arc); err != nil {
			return InteractionArcSelection{}, err
		}
		arcByID[arc.ID] = arc
	}
	measurementByArc := make(map[artifact.ID]InteractionArcMeasurement, len(measurements))
	var policy artifact.ID
	for _, measurement := range measurements {
		if err := interactionArcMeasurementCodec.ValidateIdentity(measurement); err != nil || policy.Valid() && policy != measurement.Policy {
			return InteractionArcSelection{}, errors.Join(errors.New("dataset: invalid interaction arc measurement population"), err)
		}
		policy = measurement.Policy
		if _, duplicate := measurementByArc[measurement.Arc]; duplicate {
			return InteractionArcSelection{}, errors.New("dataset: duplicate interaction arc measurement")
		}
		measurementByArc[measurement.Arc] = measurement
	}
	var branchEntries []InteractionArcEntry
	for _, entry := range projection.Arcs {
		arc, found := arcByID[entry.Arc]
		if !found || arc.Branch != entry.Branch || arc.RequestSequence != entry.RequestSequence || arc.LastSequence != entry.LastSequence {
			return InteractionArcSelection{}, errors.New("dataset: interaction arc differs from projection entry")
		}
		if entry.Branch == branch {
			branchEntries = append(branchEntries, entry)
		}
	}
	if len(branchEntries) == 0 || len(measurementByArc) != len(branchEntries) {
		return InteractionArcSelection{}, errors.New("dataset: measurements do not exactly cover selected branch")
	}
	sources := make([]InteractionSelectionSource, len(branchEntries))
	measurementIDs := make([]artifact.ID, len(branchEntries))
	for index, entry := range branchEntries {
		measurement, found := measurementByArc[entry.Arc]
		if !found {
			return InteractionArcSelection{}, errors.New("dataset: missing interaction arc measurement")
		}
		delete(measurementByArc, entry.Arc)
		measurementIDs[index] = measurement.ID
		sources[index] = InteractionSelectionSource{
			Source: entry.Arc, CausalRoot: projection.Trace, Tokens: measurement.Tokens,
			Bytes: measurement.Bytes, Documents: 1, Depth: uint32(len(branchEntries) - index),
		}
	}
	if len(measurementByArc) != 0 {
		return InteractionArcSelection{}, errors.New("dataset: measurement belongs to another branch")
	}
	page, err := SelectInteractions(bounds, head, sources, nil)
	if err != nil {
		return InteractionArcSelection{}, err
	}
	selection := InteractionArcSelection{
		Version: artifact.InitialDocumentVersion, Head: head, Projection: projection.ID, Branch: branch,
		Bounds: bounds, Policy: policy, Measurements: measurementIDs,
		SourceSet: page.SourceSet, Sources: slices.Clone(page.Sources),
		Usage: page.Usage, Population: uint32(len(branchEntries)), Truncated: page.Truncated,
	}
	return interactionArcSelectionCodec.New(selection)
}

func canonicalizeInteractionArcSelection(value *InteractionArcSelection) error {
	if value == nil {
		return errors.New("dataset: invalid interaction arc selection")
	}
	if value.Version != artifact.InitialDocumentVersion || !value.Head.Valid() || value.Projection.Kind() != artifact.KindProfile ||
		value.Policy.Kind() != artifact.KindProfile || value.SourceSet.Kind() != artifact.KindEvidence || len(value.Sources) == 0 ||
		value.Population != uint32(len(value.Measurements)) || value.Population < uint32(len(value.Sources)) ||
		value.Truncated != (value.Population > uint32(len(value.Sources))) ||
		(value.Branch != "" && !textcheck.Bounded(value.Branch, artifact.MaxContentBytes, "\x00\r\n\t")) {
		return errors.New("dataset: invalid interaction arc selection")
	}
	seen := make(map[artifact.ID]struct{}, len(value.Measurements))
	for _, id := range value.Measurements {
		if id.Kind() != artifact.KindEvidence {
			return errors.New("dataset: invalid interaction arc selection measurement")
		}
		if _, duplicate := seen[id]; duplicate {
			return errors.New("dataset: duplicate interaction arc selection measurement")
		}
		seen[id] = struct{}{}
	}
	if _, err := value.Bounds.Identify(); err != nil {
		return err
	}
	boundary := InteractionSelection{Bounds: value.Bounds, Sources: slices.Clone(value.Sources), Usage: value.Usage}
	if err := validateInteractionSelectionPage(boundary); err != nil {
		return err
	}
	return nil
}

// Lineage binds an arc selection to its projection, policy, and measurements.
func (value InteractionArcSelection) Lineage() []artifact.Lineage {
	parents := slices.Concat([]artifact.ID{value.Projection, value.Policy}, value.Measurements)
	slices.SortFunc(parents, artifact.CompareID)
	return artifact.DependencyLineage(value.ID, slices.Compact(parents)...)
}

// Batch atomically publishes every owner-derived measurement and the exact
// head-bound admission that consumed it.
func (value InteractionArcSelection) Batch(key string, measurements []InteractionArcMeasurement) (artifact.Batch, error) {
	if err := interactionArcSelectionCodec.ValidateIdentity(value); err != nil || len(measurements) != len(value.Measurements) {
		return artifact.Batch{}, errors.Join(errors.New("dataset: invalid interaction arc selection publication"), err)
	}
	contents := make([]artifact.Content, 0, len(measurements)+1)
	lineage := make([]artifact.Lineage, 0)
	for index, measurement := range measurements {
		if err := interactionArcMeasurementCodec.ValidateIdentity(measurement); err != nil || measurement.ID != value.Measurements[index] ||
			measurement.Policy != value.Policy {
			return artifact.Batch{}, errors.Join(errors.New("dataset: invalid interaction arc selection measurement publication"), err)
		}
		content, err := interactionArcMeasurementCodec.Content(measurement)
		if err != nil {
			return artifact.Batch{}, err
		}
		contents = append(contents, content)
		lineage = append(lineage, measurement.Lineage()...)
	}
	content, err := interactionArcSelectionCodec.Content(value)
	if err != nil {
		return artifact.Batch{}, err
	}
	contents = append(contents, content)
	lineage = append(lineage, value.Lineage()...)
	return artifact.NewDocumentBatch(key, contents, lineage, nil)
}

// LoadInteractionArcSelection is the low-level loader verified by runrecord.
func LoadInteractionArcSelection(ctx context.Context, reader artifact.Reader, id artifact.ID) (InteractionArcSelection, error) {
	switch {
	case ctx == nil:
		return InteractionArcSelection{}, errors.New("dataset: interaction arc selection context is absent")
	case reader == nil:
		return InteractionArcSelection{}, errors.New("dataset: interaction arc selection reader is absent")
	case id.Kind() != artifact.KindEvidence:
		return InteractionArcSelection{}, errors.New("dataset: invalid interaction arc selection identity")
	}
	return interactionArcSelectionCodec.Require(ctx, reader, id)
}
