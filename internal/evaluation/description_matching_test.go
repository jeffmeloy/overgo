package evaluation

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// TestBestMatchChoosesTheDescriptionThatSharesTheMostWords holds the scorer to
// the candidate a description shares the largest share of words with, and to
// declining a tie and a description that shares no word.
func TestBestMatchChoosesTheDescriptionThatSharesTheMostWords(t *testing.T) {
	t.Parallel()
	candidates := []string{
		"The video shows a polar bear walking on a snowy landscape.",
		"The images depict a female superhero in a comic book style, with bold colors.",
		"Three women are sitting on a couch smiling.",
	}
	for name, test := range map[string]struct {
		description string
		want        int
	}{
		"the closest":       {"A stylized comic-book illustration of superheroes with bold colors", 1},
		"case and marks":    {"POLAR BEAR!! walking, on snow", 0},
		"no word shared":    {"xylophone", -1},
		"an empty answer":   {"", -1},
		"a shared sentence": {"Three women sitting on a couch", 2},
	} {
		if got := bestMatch(test.description, candidates); got != test.want {
			t.Errorf("%s: matched %d, want %d", name, got, test.want)
		}
	}
	if got := bestMatch("red", []string{"red car", "red bus"}); got != -1 {
		t.Errorf("a description sharing as much with two candidates matched %d, want none", got)
	}
}

// TestDescriptionMatchingScoresTheShareOfDescriptionsThatMatchTheirClip runs a
// suite through a generator that always describes a bear, and holds the
// compiler to the cases it refuses.
func TestDescriptionMatchingScoresTheShareOfDescriptionsThatMatchTheirClip(t *testing.T) {
	t.Parallel()
	candidates := []string{"A polar bear walks on snow.", "A man repairs a motorcycle in a garage."}
	suite := DescriptionMatchingSuite{Kind: descriptionMatchingKind, Schema: "fixture/v1", Source: "fixture", Cases: []DescriptionMatchingCase{
		{Name: "bear", Prompt: "p", MaxTokens: 8, Candidates: candidates, Answer: 0},
		{Name: "motorcycle", Prompt: "p", MaxTokens: 8, Candidates: candidates, Answer: 1},
	}}
	compiled, err := compileDescriptionMatching(suite)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := bindDescriptionMatching(compiled, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"), RuntimeRecipe: planID(t, artifact.KindRecipe, "recipe"),
		CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "environment"),
		Execution: ExecutionPolicy{Lifecycle: LifecycleResident},
	})
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	publishPlanFixtureAuthorities(t, store, plan)
	report, err := evaluateDescriptionMatching(t.Context(), store, textGenerator("A white polar bear walking on the snow"), compiled, plan)
	if err != nil {
		t.Fatal(err)
	}
	if report.Accuracy != 0.5 || !report.Observations[0].Accepted || report.Observations[1].Accepted || report.Observations[1].Chosen != 0 {
		t.Fatalf("report = %+v, want the bear case right and the motorcycle case matched to the bear", report)
	}
	for name, change := range map[string]func(*DescriptionMatchingSuite){
		"one candidate":      func(s *DescriptionMatchingSuite) { s.Cases[0].Candidates = s.Cases[0].Candidates[:1] },
		"an answer outside":  func(s *DescriptionMatchingSuite) { s.Cases[0].Answer = 2 },
		"an empty candidate": func(s *DescriptionMatchingSuite) { s.Cases[0].Candidates = []string{"", "b"} },
		"a repeated case":    func(s *DescriptionMatchingSuite) { s.Cases[1].Name = s.Cases[0].Name },
		"no cases":           func(s *DescriptionMatchingSuite) { s.Cases = nil },
	} {
		broken := DescriptionMatchingSuite{Kind: suite.Kind, Schema: suite.Schema, Source: suite.Source, Cases: slices.Clone(suite.Cases)}
		change(&broken)
		if _, err := compileDescriptionMatching(broken); err == nil {
			t.Errorf("a suite with %s compiled", name)
		}
	}
}

// TestFrameDescriptionMatchingAsksOfTheVideoSuitesClips holds the frame suite
// to the video suite's clips and options: the same rounds of the same spec,
// each asking for a description of the clip's frame, the answer the index of the
// clip's own caption among the four.
func TestFrameDescriptionMatchingAsksOfTheVideoSuitesClips(t *testing.T) {
	t.Parallel()
	spec := captionFixture(t, 4, 6)
	video, err := BuildCaptionMatching(spec)
	if err != nil {
		t.Fatal(err)
	}
	frames, err := BuildFrameDescriptionMatching(spec)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := compileDescriptionMatching(frames); err != nil {
		t.Fatalf("the frame suite does not compile: %v", err)
	}
	if len(frames.Cases) != len(video.Cases) || frames.Source == video.Source {
		t.Fatalf("frame suite %s/%d, video suite %s/%d", frames.Source, len(frames.Cases), video.Source, len(video.Cases))
	}
	for index, testCase := range frames.Cases {
		kind, name, question, ok := ParseMediaPrompt(testCase.Prompt)
		_, clip, _, _ := ParseMediaPrompt(video.Cases[index].Prompt)
		vid := strings.TrimSuffix(clip, clipExtension)
		if !ok || kind != MediaImage || name != vid+frameExtension || question != frameQuestion ||
			len(testCase.Candidates) != captionChoices || testCase.Candidates[testCase.Answer] != "caption of "+vid {
			t.Fatalf("case %d (%s): %q does not ask for the frame of %s with its caption among %v", index, testCase.Name, testCase.Prompt, vid, testCase.Candidates)
		}
		if video.Cases[index].Answers[0] != string(captionLetters[testCase.Answer]) {
			t.Fatalf("case %d: the video suite answers %v, the frame suite index %d", index, video.Cases[index].Answers, testCase.Answer)
		}
	}
}
