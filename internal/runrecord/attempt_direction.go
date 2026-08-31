package runrecord

import (
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
)

const (
	// AttemptDirectionMediaType identifies attempt directions.
	AttemptDirectionMediaType = "application/vnd.overgo.attempt-direction+json"
	// AttemptDirectionSchema identifies the attempt direction schema.
	AttemptDirectionSchema = "overgo/attempt-direction/v1"
)

// AttemptDirectionInput is the structural surface fingerprinted to detect a
// repeated direction.
type AttemptDirectionInput struct {
	Strategy          artifact.ID `json:"strategy"`
	BaseManifest      artifact.ID `json:"base_manifest"`
	CandidateManifest artifact.ID `json:"candidate_manifest"`
	Symbols           []string    `json:"symbols"`
	Checks            []string    `json:"checks"`
	Objective         artifact.ID `json:"objective"`
}

// AttemptDirection is the accumulated history of one exact direction
// fingerprint, recommending a pivot when it repeats without success.
type AttemptDirection struct {
	Version     uint16                `json:"version"`
	ID          artifact.ID           `json:"-"`
	Fingerprint artifact.ID           `json:"fingerprint"`
	Input       AttemptDirectionInput `json:"input"`
	First       artifact.ID           `json:"first"`
	Last        artifact.ID           `json:"last"`
	Count       uint64                `json:"count"`
	LastOutcome Outcome               `json:"last_outcome"`
	Pivot       bool                  `json:"pivot,omitzero"`
	PivotReason string                `json:"pivot_reason,omitzero"`
}

var attemptDirectionCodec = artifact.JSONDocumentCodec(
	"attempt direction", artifact.KindEvidence, AttemptDirectionMediaType, AttemptDirectionSchema,
	canonicalizeAttemptDirection, func(v AttemptDirection) artifact.ID { return v.ID }, func(v *AttemptDirection, id artifact.ID) { v.ID = id },
	func(v AttemptDirection) AttemptDirection {
		v.Input.Symbols = slices.Clone(v.Input.Symbols)
		v.Input.Checks = slices.Clone(v.Input.Checks)
		return v
	},
)

// ObserveAttemptDirection derives a structural fingerprint and extends only
// exact matching history. A repeated failure recommends a pivot.
func ObserveAttemptDirection(input AttemptDirectionInput, attempt AttemptRecord, history []AttemptDirection) (AttemptDirection, error) {
	canonicalizeDirectionInput(&input)
	fingerprint, err := artifact.JSONID(artifact.KindEvidence, input)
	if err != nil {
		return AttemptDirection{}, err
	}
	value := AttemptDirection{Version: artifact.InitialDocumentVersion, Fingerprint: fingerprint, Input: input,
		First: attempt.ID, Last: attempt.ID, Count: uint64(artifact.InitialDocumentVersion), LastOutcome: attempt.Outcome}
	for _, prior := range history {
		if prior.Fingerprint != fingerprint {
			continue
		}
		value.First, value.Count = prior.First, prior.Count+uint64(artifact.InitialDocumentVersion)
		if attempt.Outcome == OutcomeFailed {
			value.Pivot, value.PivotReason = true, "exact direction repeated without success"
		}
		break
	}
	return attemptDirectionCodec.New(value)
}

// Content returns the canonical committed bytes of the direction.
func (v AttemptDirection) Content() (artifact.Content, error) {
	return attemptDirectionCodec.Content(v)
}

// Lineage links the direction to its input authorities and boundary attempts.
func (v AttemptDirection) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(v.ID, v.Input.Strategy, v.Input.BaseManifest, v.Input.CandidateManifest, v.Input.Objective, v.First, v.Last)
}

func canonicalizeAttemptDirection(v *AttemptDirection) error {
	if v == nil || v.Version != artifact.InitialDocumentVersion || v.Fingerprint.Kind() != artifact.KindEvidence || v.First.Kind() != artifact.KindEvidence ||
		v.Last.Kind() != artifact.KindEvidence || v.Count == 0 || v.LastOutcome != OutcomeSucceeded && v.LastOutcome != OutcomeFailed || v.Pivot != (v.PivotReason != "") ||
		v.Input.Strategy.Kind() != artifact.KindProfile || v.Input.BaseManifest.Kind() != artifact.KindProfile || v.Input.CandidateManifest.Kind() != artifact.KindProfile ||
		v.Input.Objective.Kind() != artifact.KindRecipe || len(v.Input.Symbols) == 0 || len(v.Input.Checks) == 0 {
		return errors.New("run record: invalid attempt direction")
	}
	for _, item := range append(slices.Clone(v.Input.Symbols), v.Input.Checks...) {
		if !textcheck.Bounded(item, len(item), "\x00\r\n") || item == "" {
			return errors.New("run record: invalid direction surface")
		}
	}
	canonicalizeDirectionInput(&v.Input)
	want, err := artifact.JSONID(artifact.KindEvidence, v.Input)
	if err != nil || want != v.Fingerprint {
		return errors.Join(err, errors.New("run record: direction fingerprint differs"))
	}
	if v.PivotReason != "" && !textcheck.Bounded(v.PivotReason, len(v.PivotReason), "\x00\r\n") {
		return errors.New("run record: invalid pivot reason")
	}
	return nil
}

func canonicalizeDirectionInput(input *AttemptDirectionInput) {
	if input == nil {
		return
	}
	slices.Sort(input.Symbols)
	input.Symbols = slices.Compact(input.Symbols)
	slices.Sort(input.Checks)
	input.Checks = slices.Compact(input.Checks)
}
