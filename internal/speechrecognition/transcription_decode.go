package speechrecognition

import (
	"context"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/scratch"
)

var segmentedTranscriptionContract = artifact.JSONContract(artifact.KindEvidence, "overgo/segmented-transcription/v1")

// Each piece retains original sample coordinates and the exact borrowed PCM
// identity. These spans locate recognition inputs; they are not word timestamps.
type transcriptionPiece struct {
	Span          recipecontract.SampleSpan `json:"span"`
	SamplesSHA256 string                    `json:"samples_sha256"`
	Text          string                    `json:"text"`
}

type segmentedTranscription struct {
	Source   recipecontract.AudioReference `json:"source"`
	Activity artifact.ID                   `json:"activity"`
	Pieces   []transcriptionPiece          `json:"pieces"`
}

// decodeText is the single clip inference path for both whole-record and
// activity-selected transcription. It borrows samples and reuses all numeric
// workspaces; each clip starts a fresh offline frontend/encoder computation.
func (transcriber *Transcriber) decodeText(ctx context.Context, samples []float32, rate int, workspace *TranscriptionWorkspace) (string, []runrecord.PhaseMetric, string, error) {
	phases := completedTranscriptionPhases(0, 0, 0, 0)
	prepareStart := time.Now()
	features, frames, _, err := transcriber.frontend.ProcessGrouped(ctx, samples, rate, &workspace.Frontend, transcriber.profile.Grouping)
	phases[1].DurationNS = elapsedNanoseconds(prepareStart)
	if err != nil {
		return "", phases, "transcription-prepare-failed", err
	}
	prefillStart := time.Now()
	hidden, outputFrames, err := transcriber.encoder.Encode(ctx, features, frames, &workspace.Encoder, nil)
	var logits []float32
	if err == nil {
		logits, err = transcriber.encoder.Project(ctx, hidden, outputFrames, &workspace.Encoder)
	}
	phases[2].DurationNS = elapsedNanoseconds(prefillStart)
	if err != nil {
		return "", phases, "transcription-inference-failed", err
	}
	postStart := time.Now()
	workspace.frameIDs = scratch.Resize(workspace.frameIDs, outputFrames)
	workspace.tokens = scratch.Resize(workspace.tokens, outputFrames)
	tokens, err := GreedyCTC(ctx, workspace.frameIDs, workspace.tokens, logits, outputFrames, transcriber.encoder.VocabularySize(), transcriber.profile.BlankToken)
	var text string
	if err == nil {
		text, err = transcriber.tokenizer.DecodeStrict(tokens)
	}
	phases[3].DurationNS = elapsedNanoseconds(postStart)
	return text, phases, "transcription-postprocess-failed", err
}
