package evaluation

import (
	"slices"
	"strings"
	"testing"
)

// TestCaptionMatchingOfFramesAsksOfStillsAndReadsTheLeadingLetter holds the
// image form of the suite to the clips and options of the video form: the same
// clips with the same captions, each named by a frame, answered by the letter a
// sentence opens with.
func TestCaptionMatchingOfFramesAsksOfStillsAndReadsTheLeadingLetter(t *testing.T) {
	t.Parallel()
	spec := captionFixture(t, 4, 6)
	video, err := BuildCaptionMatching(spec)
	if err != nil {
		t.Fatal(err)
	}
	spec.Kind = MediaImage
	frames, err := BuildCaptionMatching(spec)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := CompileGeneratedAnswer(frames); err != nil {
		t.Fatalf("the frame suite does not compile: %v", err)
	}
	if !slices.Contains(frames.Transforms, transformLeadingLetter) || slices.Contains(video.Transforms, transformLeadingLetter) {
		t.Fatalf("transforms: frames %v, clips %v", frames.Transforms, video.Transforms)
	}
	if frames.Source == video.Source || frames.Schema == video.Schema || len(frames.Cases) != len(video.Cases) {
		t.Fatalf("the frame suite must be another suite over the same clips: %s/%d against %s/%d", frames.Source, len(frames.Cases), video.Source, len(video.Cases))
	}
	for index, testCase := range frames.Cases {
		kind, name, question, ok := ParseMediaPrompt(testCase.Prompt)
		_, clip, clipQuestion, _ := ParseMediaPrompt(video.Cases[index].Prompt)
		if !ok || kind != MediaImage || !strings.HasSuffix(name, frameExtension) ||
			strings.TrimSuffix(name, frameExtension) != strings.TrimSuffix(clip, clipExtension) ||
			strings.Replace(question, "image", "video", 1) != clipQuestion || testCase.Answers[0] != video.Cases[index].Answers[0] {
			t.Fatalf("case %d: %q does not ask the video's question of the clip's frame", index, testCase.Prompt)
		}
	}
}
