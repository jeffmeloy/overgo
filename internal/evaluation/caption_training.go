package evaluation

import (
	"os"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

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
