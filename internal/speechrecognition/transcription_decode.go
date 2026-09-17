package speechrecognition

import (
	"context"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/processmeasure"
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
func (transcriber *transcriptionModel) decodeText(ctx context.Context, samples []float32, rate int, workspace *transcriptionWorkspace, walls *processmeasure.Walls) (string, []runrecord.PhaseMetric, string, error) {
	phases := completedTranscriptionPhases(0, 0, 0, 0)
	prepareStart := processmeasure.NewStopwatch()
	var features []float32
	var frames int
	var err error
	if transcriber.transducer == nil {
		features, frames, _, err = transcriber.frontend.ProcessGrouped(ctx, samples, rate, &workspace.Frontend, transcriber.profile.Grouping)
	} else {
		features, frames, err = transcriber.frontend.Process(ctx, [][]float32{samples}, rate, &workspace.Frontend, audiodsp.ProcessOptions{})
	}
	phases[1].DurationNS = walls.Elapsed(prepareStart)
	if err != nil {
		return "", phases, "transcription-prepare-failed", err
	}
	prefillStart := processmeasure.NewStopwatch()
	var logits []float32
	var outputFrames int
	var result TransducerResult
	if transcriber.transducer == nil {
		var hidden []float32
		hidden, outputFrames, err = transcriber.encoder.Encode(ctx, features, frames, &workspace.Encoder, nil)
		if err == nil {
			logits, err = transcriber.encoder.Project(ctx, hidden, outputFrames, &workspace.Encoder)
		}
	} else {
		result, err = transcriber.transducer.Recognize(ctx, features, frames, &workspace.Transducer, nil)
	}
	phases[2].DurationNS = walls.Elapsed(prefillStart)
	if err != nil {
		return "", phases, "transcription-inference-failed", err
	}
	postStart := processmeasure.NewStopwatch()
	var text string
	if transcriber.transducer == nil {
		workspace.frameIDs = scratch.Resize(workspace.frameIDs, outputFrames)
		workspace.tokens = scratch.Resize(workspace.tokens, outputFrames)
		var tokens []int
		tokens, err = GreedyCTC(ctx, workspace.frameIDs, workspace.tokens, logits, outputFrames, transcriber.encoder.VocabularySize(), transcriber.profile.BlankToken)
		if err == nil {
			text, err = transcriber.tokenizer.DecodeStrict(tokens)
		}
	} else {
		text, err = transcriber.tokenizer.DecodeText(result.Tokens)
	}
	phases[3].DurationNS = walls.Elapsed(postStart)
	return text, phases, "transcription-postprocess-failed", err
}
