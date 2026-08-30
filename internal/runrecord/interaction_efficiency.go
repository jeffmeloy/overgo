package runrecord

import (
	"errors"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
)

const (
	// EfficiencyTraceMediaType is the wire media type of the document.
	EfficiencyTraceMediaType = "application/vnd.overgo.interaction-efficiency-trace+json"
	// EfficiencyTraceSchema is the versioned document schema name.
	EfficiencyTraceSchema = "overgo/interaction-efficiency-trace/v1"
)

// InteractionSurface names the system surface a trace measured.
type InteractionSurface string

// The surfaces the interaction-efficiency and resource-fitness baselines cover; a
// trace on any other surface is refused so surface totals stay
// comparable across campaigns.
const (
	// SurfaceStorage traces OvergoDB session work.
	SurfaceStorage InteractionSurface = "storage"
	// SurfaceAgent traces agent attempt work.
	SurfaceAgent InteractionSurface = "agent"
	// SurfaceTool traces tool invocation work.
	SurfaceTool InteractionSurface = "tool"
	// SurfaceWorkflow traces workflow operation work.
	SurfaceWorkflow InteractionSurface = "workflow"
	// SurfacePeer traces peer coordination work.
	SurfacePeer InteractionSurface = "peer"
	// SurfaceEvaluation traces evaluation run work.
	SurfaceEvaluation InteractionSurface = "evaluation"
	// SurfaceUI traces operator interface work.
	SurfaceUI InteractionSurface = "ui"
	// SurfaceTraining traces model training work.
	SurfaceTraining InteractionSurface = "training"
	// SurfaceServing traces model serving work.
	SurfaceServing InteractionSurface = "serving"
)

var interactionSurfaces = []InteractionSurface{
	SurfaceStorage, SurfaceAgent, SurfaceTool, SurfaceWorkflow,
	SurfacePeer, SurfaceEvaluation, SurfaceUI,
	SurfaceTraining, SurfaceServing,
}

// InteractionWork counts the exact work one traced task performed.
// Counters are exact observations, never estimates: a baseline whose
// counts cannot be reproduced from its task's construction is not
// admissible evidence for an efficiency comparison.
type InteractionWork struct {
	SemanticTransitions uint64 `json:"semantic_transitions,omitempty"`
	Commits             uint64 `json:"commits,omitempty"`
	ArtifactReads       uint64 `json:"artifact_reads,omitempty"`
	BlobReads           uint64 `json:"blob_reads,omitempty"`
	ScannedFacts        uint64 `json:"scanned_facts,omitempty"`
	ReturnedFacts       uint64 `json:"returned_facts,omitempty"`
	Bytes               uint64 `json:"bytes,omitempty"`
	Wakeups             uint64 `json:"wakeups,omitempty"`
	Failures            uint64 `json:"failures,omitempty"`
	Retries             uint64 `json:"retries,omitempty"`
	ModelTurns          uint64 `json:"model_turns,omitempty"`
	ContextBytes        uint64 `json:"context_bytes,omitempty"`
	RepeatedContextIDs  uint64 `json:"repeated_context_ids,omitempty"`
	ToolCalls           uint64 `json:"tool_calls,omitempty"`
	Waits               uint64 `json:"waits,omitempty"`
	CopiedBytes         uint64 `json:"copied_bytes,omitempty"`
	Allocations         uint64 `json:"allocations,omitempty"`
	WallNS              uint64 `json:"wall_ns,omitempty"`
	ResourcePeakBytes   uint64 `json:"resource_peak_bytes,omitempty"`
}

// counters names every comparison dimension with its exact observation.
func (work InteractionWork) counters() map[string]uint64 {
	return map[string]uint64{
		"semantic_transitions": work.SemanticTransitions,
		"commits":              work.Commits,
		"artifact_reads":       work.ArtifactReads,
		"blob_reads":           work.BlobReads,
		"scanned_facts":        work.ScannedFacts,
		"returned_facts":       work.ReturnedFacts,
		"bytes":                work.Bytes,
		"wakeups":              work.Wakeups,
		"failures":             work.Failures,
		"retries":              work.Retries,
		"model_turns":          work.ModelTurns,
		"context_bytes":        work.ContextBytes,
		"repeated_context_ids": work.RepeatedContextIDs,
		"tool_calls":           work.ToolCalls,
		"waits":                work.Waits,
		"copied_bytes":         work.CopiedBytes,
		"allocations":          work.Allocations,
		"wall_ns":              work.WallNS,
		"resource_peak_bytes":  work.ResourcePeakBytes,
	}
}

// EfficiencyCounterNames enumerates the closed comparison surface in stable
// order. A claim must cover every name: a counter nobody measured blocks the
// claim instead of flattering it.
func EfficiencyCounterNames() []string {
	counters := (InteractionWork{}).counters()
	names := make([]string, 0, len(counters))
	for name := range counters {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// EfficiencyTradeoff is the explicit reviewed decision that accepts named
// worsened counters for a non-Pareto win. Without it, any regression on any
// covered counter leaves the claim without a win.
type EfficiencyTradeoff struct {
	Decider   string   `json:"decider"`
	Rationale string   `json:"rationale"`
	Accepted  []string `json:"accepted"`
}

// EfficiencyComparison is the typed verdict over one candidate trace.
type EfficiencyComparison struct {
	Win      bool     `json:"win"`
	Improved []string `json:"improved,omitempty"`
	Worsened []string `json:"worsened,omitempty"`
	Tradeoff bool     `json:"tradeoff,omitempty"`
}

// CompareEfficiencyTraces judges one candidate trace against its baseline.
// Both traces must be canonical and complete the same task on the same
// surface with result and evidence bound — an optimization that loses
// capability or evidence never reaches the counters. The claim must declare
// coverage of the complete closed counter surface; a missing counter blocks
// it. A win requires a Pareto improvement, or an explicit tradeoff decision
// naming every worsened counter — work merely shifted onto an unnamed
// counter refuses.
func CompareEfficiencyTraces(
	candidate, baseline EfficiencyTrace,
	covered []string,
	tradeoff *EfficiencyTradeoff,
) (EfficiencyComparison, error) {
	candidate.Version, baseline.Version = artifact.InitialDocumentVersion, artifact.InitialDocumentVersion
	if err := errors.Join(canonicalizeEfficiencyTrace(&candidate), canonicalizeEfficiencyTrace(&baseline)); err != nil {
		return EfficiencyComparison{}, err
	}
	if candidate.Surface != baseline.Surface || candidate.Task != baseline.Task {
		return EfficiencyComparison{}, errors.New("run record: traces measure different tasks; they are not comparable")
	}
	for _, name := range EfficiencyCounterNames() {
		if !slices.Contains(covered, name) {
			return EfficiencyComparison{}, errors.New(
				"run record: counter " + name + " is not covered; an unmeasured claim is blocked",
			)
		}
	}
	candidateCounters, baselineCounters := candidate.Work.counters(), baseline.Work.counters()
	comparison := EfficiencyComparison{}
	for _, name := range EfficiencyCounterNames() {
		switch {
		case candidateCounters[name] < baselineCounters[name]:
			comparison.Improved = append(comparison.Improved, name)
		case candidateCounters[name] > baselineCounters[name]:
			comparison.Worsened = append(comparison.Worsened, name)
		}
	}
	if len(comparison.Worsened) == 0 {
		comparison.Win = len(comparison.Improved) != 0
		return comparison, nil
	}
	if tradeoff == nil {
		return comparison, nil
	}
	if tradeoff.Decider == "" || tradeoff.Rationale == "" {
		return EfficiencyComparison{}, errors.New("run record: tradeoff decision requires a decider and a rationale")
	}
	for _, worsened := range comparison.Worsened {
		if !slices.Contains(tradeoff.Accepted, worsened) {
			return EfficiencyComparison{}, errors.New(
				"run record: counter " + worsened + " worsened without an accepting tradeoff decision; work was shifted, not saved",
			)
		}
	}
	comparison.Win = len(comparison.Improved) != 0
	comparison.Tradeoff = true
	return comparison, nil
}

// EfficiencyTrace records one representative task's exact work and
// round-trip counts on one surface, bound to the completed result and
// its evidence -- so an interaction reduction can never take credit
// for simply doing less of the task.
type EfficiencyTrace struct {
	Version  uint16             `json:"version"`
	Surface  InteractionSurface `json:"surface"`
	Task     string             `json:"task"`
	Work     InteractionWork    `json:"work"`
	Result   artifact.ID        `json:"result"`
	Evidence artifact.ID        `json:"evidence"`
	ID       artifact.ID        `json:"-"`
}

var efficiencyTraceCodec = artifact.JSONDocumentCodec(
	"interaction efficiency trace", artifact.KindEvidence, EfficiencyTraceMediaType, EfficiencyTraceSchema,
	canonicalizeEfficiencyTrace,
	func(value EfficiencyTrace) artifact.ID { return value.ID },
	func(value *EfficiencyTrace, id artifact.ID) { value.ID = id }, nil,
)

func canonicalizeEfficiencyTrace(value *EfficiencyTrace) error {
	if value.Version != artifact.InitialDocumentVersion {
		return errors.New("run record: invalid interaction trace version")
	}
	if !slices.Contains(interactionSurfaces, value.Surface) {
		return errors.New("run record: unknown interaction surface")
	}
	if value.Task == "" || strings.TrimSpace(value.Task) != value.Task {
		return errors.New("run record: invalid interaction trace task")
	}
	if value.Work == (InteractionWork{}) {
		return errors.New("run record: interaction trace records no work")
	}
	if !value.Result.Valid() || value.Evidence.Kind() != artifact.KindEvidence {
		return errors.New("run record: interaction trace must bind result and evidence")
	}
	return nil
}

// NewEfficiencyTrace canonicalizes and identifies one trace.
func NewEfficiencyTrace(value EfficiencyTrace) (EfficiencyTrace, error) {
	value.Version = artifact.InitialDocumentVersion
	return efficiencyTraceCodec.New(value)
}

// ParseEfficiencyTrace decodes one canonical trace document.
func ParseEfficiencyTrace(data []byte) (EfficiencyTrace, error) {
	return efficiencyTraceCodec.Parse(data)
}

// Content returns the canonical committed bytes of the trace.
func (value EfficiencyTrace) Content() (artifact.Content, error) {
	return efficiencyTraceCodec.Content(value)
}

// Lineage binds the trace to its result and evidence.
func (value EfficiencyTrace) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, value.Result, value.Evidence)
}

// Batch wraps the trace as one committable store batch.
func (value EfficiencyTrace) Batch(key string) (artifact.Batch, error) {
	return efficiencyTraceCodec.Batch(key, value, value.Lineage(), nil)
}
