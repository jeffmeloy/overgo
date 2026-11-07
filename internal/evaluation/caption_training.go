package evaluation

import (
	"os"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

const (
	// CaptureSamplesSuffix ends the name of the samples file a seam capture writes beside its representations, which
	// names its samples for a bridge's training.
	CaptureSamplesSuffix = ".samples.json"
	// CaptureTarget names a capture's tensor of each sample's caption mean embedding, a row of the text model's width.
	CaptureTarget = "target"
)

// HeldBack divides a capture's samples' items into those a run trains on and those it holds back: every sample of the
// first held source videos in the capture's order, which its seed shuffled.
func HeldBack[T any](samples []CapturedCaption, items []T, held int) (trained, scored []T) {
	sources := map[string]bool{}
	for index, sample := range samples {
		if sources[sample.Source] || len(sources) < held {
			sources[sample.Source] = true
			scored = append(scored, items[index])
		} else {
			trained = append(trained, items[index])
		}
	}
	return trained, scored
}

// CaptionTrainingClip is a captioned clip a model may train on beside a caption-matching suite, with the source video
// it groups under.
type CaptionTrainingClip struct {
	Vid     string `json:"vid"`
	Source  string `json:"source"`
	Caption string `json:"caption"`
}

// CaptionTrainingCapture is the training side a seam capture read, as a bridge's training reads it back: the encoder
// and the text model it paired, the acceptance it serves, the directory holding each sample's frame tokens, the
// encoder's token width and the text model's, and the samples in the capture's order.
type CaptionTrainingCapture struct {
	Encoder      string            `json:"encoder"`
	TextModel    string            `json:"text_model"`
	Acceptance   string            `json:"acceptance"`
	Tokens       string            `json:"tokens"`
	EncoderWidth int               `json:"encoder_width"`
	TextWidth    int               `json:"text_width"`
	Samples      []CapturedCaption `json:"samples"`
}

// CapturedCaption is a captured sample: its clip, and the identity of the bytes of its frame's tokens as the capture
// cached them.
type CapturedCaption struct {
	CaptionTrainingClip
	Frame artifact.ID `json:"frame"`
}

// CaptionTrainingClips are the captioned clips of spec's directory a model may train on beside the suite spec builds:
// every clip but the suite's own, every other clip of their source videos, and every clip whose caption the suite offers
// as a wrong answer, in the order of their identities.
func CaptionTrainingClips(spec CaptionMatchingSpec) ([]CaptionTrainingClip, error) {
	rounds, captions, err := captionRounds(spec)
	if err != nil {
		return nil, err
	}
	heldClips, heldSources := map[string]bool{}, map[string]bool{}
	for _, round := range rounds {
		source, _, _ := strings.Cut(round.vid, sceneMarker)
		heldSources[source] = true
		for _, option := range round.options {
			heldClips[option] = true
		}
	}
	entries, err := os.ReadDir(spec.Clips)
	if err != nil {
		return nil, err
	}
	var clips []CaptionTrainingClip
	for _, entry := range entries {
		vid, isClip := strings.CutSuffix(entry.Name(), clipExtension)
		caption, captioned := captions[vid]
		source, _, _ := strings.Cut(vid, sceneMarker)
		if isClip && captioned && !heldClips[vid] && !heldSources[source] {
			clips = append(clips, CaptionTrainingClip{Vid: vid, Source: source, Caption: caption})
		}
	}
	slices.SortFunc(clips, func(left, right CaptionTrainingClip) int { return strings.Compare(left.Vid, right.Vid) })
	return clips, nil
}
