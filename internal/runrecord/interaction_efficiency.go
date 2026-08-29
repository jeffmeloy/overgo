package runrecord

import (
	"errors"
	"slices"
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
	SemanticTransitions uint64 `json:"semantic_transitions,omitzero"`
	Commits             uint64 `json:"commits,omitzero"`
	ArtifactReads       uint64 `json:"artifact_reads,omitzero"`
	BlobReads           uint64 `json:"blob_reads,omitzero"`
	ScannedFacts        uint64 `json:"scanned_facts,omitzero"`
	ReturnedFacts       uint64 `json:"returned_facts,omitzero"`
	Bytes               uint64 `json:"bytes,omitzero"`
	Wakeups             uint64 `json:"wakeups,omitzero"`
	Failures            uint64 `json:"failures,omitzero"`
	Retries             uint64 `json:"retries,omitzero"`
	ModelTurns          uint64 `json:"model_turns,omitzero"`
	ContextBytes        uint64 `json:"context_bytes,omitzero"`
	RepeatedContextIDs  uint64 `json:"repeated_context_ids,omitzero"`
	ToolCalls           uint64 `json:"tool_calls,omitzero"`
	Waits               uint64 `json:"waits,omitzero"`
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
