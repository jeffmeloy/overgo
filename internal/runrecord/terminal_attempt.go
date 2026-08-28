package runrecord

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/executionfailure"
	"overgo/internal/processcontrol"
)

const (
	// TerminalAttemptReceiptMediaType identifies terminal attempt receipts.
	TerminalAttemptReceiptMediaType = "application/vnd.overgo.terminal-attempt-receipt+json"
	// TerminalAttemptReceiptSchema is the receipt document version.
	TerminalAttemptReceiptSchema = "overgo/terminal-attempt-receipt/v1"
	// TerminalAttemptAliasRoot scopes receipts per operation and attempt.
	TerminalAttemptAliasRoot = "attempt/terminal/"
)

// Coverage gap labels: a receipt without an optional measurement names
// the gap explicitly, so absence is a recorded fact instead of a hole
// a reader infers.
const (
	// GapProcessTermination marks an attempt with no supervised process
	// termination evidence.
	GapProcessTermination = "process-termination"
	// GapResources marks an attempt with no resource measurements.
	GapResources = "resources"
	// GapTranscript marks an attempt whose transcript was not captured.
	GapTranscript = "transcript"
	// GapToolOutput marks an attempt whose tool output was not captured.
	GapToolOutput = "tool-output"
)

// ProcessTermination summarizes one supervised process tree's terminal
// state inside a receipt.
type ProcessTermination struct {
	ExitCode       int32 `json:"exit_code"`
	Interrupted    bool  `json:"interrupted,omitempty"`
	TreeTerminated bool  `json:"tree_terminated,omitempty"`
	StdoutBytes    int64 `json:"stdout_bytes,omitempty"`
	StderrBytes    int64 `json:"stderr_bytes,omitempty"`
	WallNS         int64 `json:"wall_ns"`
}

// NewProcessTermination adapts a supervisor receipt into receipt
// evidence.
func NewProcessTermination(receipt processcontrol.Receipt) ProcessTermination {
	return ProcessTermination{
		ExitCode: int32(receipt.ExitCode), Interrupted: receipt.Interrupted,
		TreeTerminated: receipt.TreeTerminated, StdoutBytes: receipt.StdoutBytes,
		StderrBytes: receipt.StderrBytes, WallNS: receipt.WallNS,
	}
}

// TerminalAttemptReceipt is the one self-contained terminal record of
// an execution attempt: canonical outcome, typed failure evidence and
// the derived disposition when it failed, process termination and
// resource measurements when they exist, referenced blobs for large
// transcript and tool output, and recovery lineage back to the prior
// attempt. Whatever optional evidence is absent is named in Gaps.
type TerminalAttemptReceipt struct {
	Version   uint16      `json:"version"`
	Operation artifact.ID `json:"operation"`
	Attempt   uint32      `json:"attempt,omitempty"`
	Outcome   Outcome     `json:"outcome"`

	FailureObservation   artifact.ID                   `json:"failure_observation,omitzero"`
	FailureNormalization artifact.ID                   `json:"failure_normalization,omitzero"`
	Disposition          *executionfailure.Disposition `json:"disposition,omitempty"`

	Process   *ProcessTermination `json:"process,omitempty"`
	Resources ServingResources    `json:"resources"`
	Gaps      []string            `json:"gaps,omitempty"`

	Transcript artifact.ID `json:"transcript,omitzero"`
	ToolOutput artifact.ID `json:"tool_output,omitzero"`

	Previous           artifact.ID `json:"previous,omitzero"`
	RestoredCheckpoint artifact.ID `json:"restored_checkpoint,omitzero"`
	ObservedUnixNS     int64       `json:"observed_unix_ns"`
	ID                 artifact.ID `json:"-"`
}

var terminalAttemptCodec = artifact.JSONDocumentCodec(
	"terminal attempt receipt", artifact.KindEvidence, TerminalAttemptReceiptMediaType, TerminalAttemptReceiptSchema,
	canonicalizeTerminalAttempt,
	func(value TerminalAttemptReceipt) artifact.ID { return value.ID },
	func(value *TerminalAttemptReceipt, id artifact.ID) { value.ID = id },
	func(value TerminalAttemptReceipt) TerminalAttemptReceipt {
		value.Gaps = slices.Clone(value.Gaps)
		return value
	},
)

// RequireTerminalAttemptReceipt returns a validated stored receipt.
func RequireTerminalAttemptReceipt(ctx context.Context, reader artifact.Reader, id artifact.ID) (TerminalAttemptReceipt, error) {
	return terminalAttemptCodec.Require(ctx, reader, id)
}

// ResolveTerminalAttemptReceipt returns one operation's receipt for one
// attempt ordinal, when that attempt closed.
func ResolveTerminalAttemptReceipt(ctx context.Context, reader artifact.Reader, operation artifact.ID, attempt uint32) (TerminalAttemptReceipt, bool, error) {
	return terminalAttemptCodec.Resolve(ctx, reader, terminalAttemptAlias(operation, attempt))
}

// PublishTerminalAttemptReceipt commits one terminal receipt. Failure
// evidence must already exist in the store, and the receipt links it,
// the referenced blobs, and the recovery lineage as parents, so the
// receipt closes over everything evaluation and recovery read.
func PublishTerminalAttemptReceipt(ctx context.Context, repository artifact.Repository, value TerminalAttemptReceipt) (TerminalAttemptReceipt, error) {
	if ctx == nil || repository == nil {
		return TerminalAttemptReceipt{}, errors.New("run record: terminal attempt repository is absent")
	}
	if value.FailureObservation.Valid() {
		if _, err := RequireFailureObservation(ctx, repository, value.FailureObservation); err != nil {
			return TerminalAttemptReceipt{}, err
		}
	}
	if value.FailureNormalization.Valid() {
		if _, err := RequireFailureNormalization(ctx, repository, value.FailureNormalization); err != nil {
			return TerminalAttemptReceipt{}, err
		}
	}
	value.Version, value.ID = artifact.InitialDocumentVersion, artifact.ID{}
	identified, err := terminalAttemptCodec.New(value)
	if err != nil {
		return TerminalAttemptReceipt{}, err
	}
	parents := []artifact.ID{identified.Operation}
	for _, parent := range []artifact.ID{
		identified.FailureObservation, identified.FailureNormalization,
		identified.Transcript, identified.ToolOutput,
		identified.Previous, identified.RestoredCheckpoint,
	} {
		if parent.Valid() {
			parents = append(parents, parent)
		}
	}
	alias := terminalAttemptAlias(identified.Operation, identified.Attempt)
	return commitFailureDocument(ctx, repository, terminalAttemptCodec,
		"attempt/terminal/"+identified.ID.String(), identified,
		artifact.DependencyLineage(identified.ID, parents...),
		[]artifact.AliasBinding{{Name: alias, Target: identified.ID}})
}

func terminalAttemptAlias(operation artifact.ID, attempt uint32) string {
	return TerminalAttemptAliasRoot + operation.String() + "/" + fmt.Sprint(attempt)
}

// terminalGapFields pairs each coverage gap label with whether the
// receipt actually carries that measurement, so canonicalization can
// force the gap list to state exactly what is absent.
func terminalGapFields(value *TerminalAttemptReceipt) map[string]bool {
	return map[string]bool{
		GapProcessTermination: value.Process != nil,
		GapResources:          value.Resources != (ServingResources{}),
		GapTranscript:         value.Transcript.Valid(),
		GapToolOutput:         value.ToolOutput.Valid(),
	}
}

func canonicalizeTerminalAttempt(value *TerminalAttemptReceipt) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Operation.Kind() != artifact.KindEvidence || value.ObservedUnixNS <= 0 ||
		!ValidOutcome(value.Outcome) {
		return errors.New("run record: invalid terminal attempt receipt")
	}
	failed := value.Outcome == OutcomeFailed
	if failed != value.FailureObservation.Valid() || failed != value.FailureNormalization.Valid() ||
		failed != (value.Disposition != nil) {
		return errors.New("run record: terminal failure evidence must match the outcome")
	}
	if value.Disposition != nil && (!executionfailure.ValidDecision(value.Disposition.Decision) ||
		!executionfailure.ValidCause(value.Disposition.Situation.Cause) || value.Disposition.Rule == "") {
		return errors.New("run record: invalid terminal disposition")
	}
	present := terminalGapFields(value)
	if !slices.IsSorted(value.Gaps) {
		return errors.New("run record: terminal coverage gaps are unsorted")
	}
	for gap, carried := range present {
		named := slices.Contains(value.Gaps, gap)
		if named == carried {
			return fmt.Errorf("run record: coverage gap %q must name exactly the absent measurement", gap)
		}
	}
	if len(value.Gaps) != countAbsent(present) {
		return errors.New("run record: foreign terminal coverage gap")
	}
	for _, blob := range []artifact.ID{value.Transcript, value.ToolOutput} {
		if blob.Valid() && blob.Kind() != artifact.KindFile {
			return errors.New("run record: terminal blob references must be files")
		}
	}
	if value.Previous.Valid() != (value.Attempt > 0) || value.Previous == value.Operation ||
		value.Previous.Valid() && value.Previous.Kind() != artifact.KindEvidence {
		return errors.New("run record: invalid terminal recovery lineage")
	}
	if value.RestoredCheckpoint.Valid() && value.RestoredCheckpoint.Kind() != artifact.KindEvidence {
		return errors.New("run record: invalid terminal restored checkpoint")
	}
	return nil
}

func countAbsent(present map[string]bool) int {
	absent := 0
	for _, carried := range present {
		if !carried {
			absent++
		}
	}
	return absent
}
