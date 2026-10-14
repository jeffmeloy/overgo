package evaluation

import (
	"context"
	"errors"

	"overgo/internal/artifact"
)

// ReplayReceiptSchema identifies replay scope, including a recorded full fallback.
const ReplayReceiptSchema = "overgo/evaluation-replay/v1"

var replayReceiptContract = artifact.DocumentContract{
	Kind: artifact.KindEvaluation, MediaType: "application/vnd.overgo.evaluation-replay+json", Schema: ReplayReceiptSchema,
}

// ReplayReceipt is the report of a run that carried Carried's result to
// Plan's commit: Replay, a replay of Cases (suite indices), reproduced their
// observations in Report, the full report that result stands on. Runs wrote
// one until 2026-10-10, when a result came to be given again only under the
// identity of the code that gave it (Plan.observed); nothing writes one now,
// and the store's are read where a record names one.
type ReplayReceipt struct {
	Version uint16      `json:"version"`
	Plan    artifact.ID `json:"plan"`
	Carried artifact.ID `json:"carried,omitzero"`
	Report  artifact.ID `json:"report"`
	Replay  artifact.ID `json:"replay,omitzero"`
	Cases   []int       `json:"cases"`
	// Fallback says Report is freshly evaluated in full after the replay.
	// Missing Replay/Cases means the failed replay's case count is unknown.
	Fallback bool        `json:"fallback,omitzero"`
	ID       artifact.ID `json:"-"`
}

var replayReceiptCodec = artifact.JSONDocumentCodec(
	"evaluation replay receipt", replayReceiptContract.Kind, replayReceiptContract.MediaType, replayReceiptContract.Schema,
	func(value *ReplayReceipt) error {
		if value == nil || value.Version != artifact.InitialDocumentVersion || value.Plan.Kind() != artifact.KindProfile ||
			value.Report.Kind() != artifact.KindEvaluation || value.Report == value.Replay ||
			value.Fallback == value.Carried.Valid() ||
			value.Carried.Valid() && value.Carried.Kind() != artifact.KindEvaluation ||
			value.Replay.Valid() && (value.Replay.Kind() != artifact.KindEvaluation || len(value.Cases) == 0) ||
			!value.Replay.Valid() && (!value.Fallback || len(value.Cases) != 0) {
			return errors.New("evaluation: incomplete replay receipt")
		}
		for index, selected := range value.Cases {
			if selected < 0 || index > 0 && selected <= value.Cases[index-1] {
				return errors.New("evaluation: replay case indices are not distinct and ordered")
			}
		}
		return nil
	},
	func(value ReplayReceipt) artifact.ID { return value.ID },
	func(value *ReplayReceipt, id artifact.ID) { value.ID = id },
)

// RequireReplayReceipt reads the existing immutable receipt through its typed
// owner. It names selected cases and prior quality, not a fresh full-suite run.
func RequireReplayReceipt(ctx context.Context, reader artifact.Reader, id artifact.ID) (ReplayReceipt, error) {
	return replayReceiptCodec.Require(ctx, reader, id)
}
