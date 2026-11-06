package evaluation

import (
	"os"
	"slices"
	"strings"
)

// CaptionTrainingClip is a captioned clip a model may train on beside a caption-matching suite, with the source video
// it groups under.
type CaptionTrainingClip struct {
	Vid, Source, Caption string
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
