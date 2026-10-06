package evaluation

import "testing"

// TestMediaPromptReadsBackWhatItWrote holds a media prompt to its parts: the
// kind, the name, and the question after the line that names them, whole.
func TestMediaPromptReadsBackWhatItWrote(t *testing.T) {
	t.Parallel()
	const question = "Which description matches the video?\nA. one\nB. two\nAnswer with the letter."
	kind, name, got, ok := ParseMediaPrompt(mediaPrompt(MediaVideo, "Eep9uvenxAo-Scene-0030.mp4", question))
	if !ok || kind != MediaVideo || name != "Eep9uvenxAo-Scene-0030.mp4" || got != question {
		t.Fatalf("parsed (%q, %q, %q, %t)", kind, name, got, ok)
	}
}

// TestMediaPromptRefusesWhatNamesNoMedia holds the reader to the one line it
// writes: a text prompt, a name with whitespace, an unknown kind and a
// missing name or question are not media prompts.
func TestMediaPromptRefusesWhatNamesNoMedia(t *testing.T) {
	t.Parallel()
	for name, prompt := range map[string]string{
		"a text prompt":      "Which description matches the video?",
		"no question":        "<overgo-media kind=video name=a.mp4>",
		"a spaced name":      "<overgo-media kind=video name=a b.mp4>\nquestion",
		"an unknown kind":    "<overgo-media kind=sound name=a.wav>\nquestion",
		"no name":            "<overgo-media kind=video name=>\nquestion",
		"no kind":            "<overgo-media name=a.mp4>\nquestion",
		"an unclosed line":   "<overgo-media kind=video name=a.mp4\nquestion",
		"text before it":     "intro <overgo-media kind=video name=a.mp4>\nquestion",
		"a third field":      "<overgo-media kind=video name=a.mp4 start=1>\nquestion",
		"the fields reverse": "<overgo-media name=a.mp4 kind=video>\nquestion",
	} {
		if _, _, _, ok := ParseMediaPrompt(prompt); ok {
			t.Errorf("%s was read as a media prompt", name)
		}
	}
}
