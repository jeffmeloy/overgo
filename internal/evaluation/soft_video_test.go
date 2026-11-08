package evaluation

import (
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestTheVideoSuiteScoresProbabilitiesOnHarderCases holds the generated-video suite to what quality-instruments gave
// the picture suites: on 2026-11-08 Wan's four-step student scored 0.898 beside its base's 0.914 with 59 of 64 clips
// tied. Every case of the suite is a choice among letters grouped by its clip, so it compiles as a suite of choices and
// scores the judge's probability beside its accuracy; it holds, beside the picked captions' cases, one case for each
// question about a clip drawn from a composed caption, asked of the clip as a video; and two clips of one caption that
// the binary count ties, both answered right, score apart by how sure the judge is of each.
func TestTheVideoSuiteScoresProbabilitiesOnHarderCases(t *testing.T) {
	t.Parallel()
	spec := captionFixture(t, 6, 6)
	items := testcheck.Must(PickVideoCaptions(spec.Captions, 5, 17))(t)
	composed := testcheck.Must(ComposeTIFA(2, 17))(t)
	videos := map[string]string{}
	questions := 0
	for _, item := range items {
		videos[item.ID] = item.ID + ".gif"
	}
	for _, item := range composed {
		videos[item.ID] = item.ID + ".gif"
		questions += len(item.Questions)
	}
	suite := testcheck.Must(BuildGeneratedVideoMatching(items, composed, videos, "judge-a"))(t)
	if _, err := CompileGeneratedAnswer(suite); err != nil || !suite.choices() || len(suite.Cases) != len(items)*generatedVideoRounds+questions {
		t.Fatalf("the suite holds %d cases, choices %t, compiling with %v; want %d cases of choices that compile", len(suite.Cases), suite.choices(), err, len(items)*generatedVideoRounds+questions)
	}
	for _, testCase := range suite.Cases {
		kind, name, _, ok := ParseMediaPrompt(testCase.Prompt)
		if !ok || kind != MediaVideo || testCase.Group == "" || name != testCase.Group+".gif" {
			t.Fatalf("case %s asks of %s %q in group %q, want the clip of its group, asked as a video", testCase.Name, kind, name, testCase.Group)
		}
	}
	if composedCases := suite.Cases[len(items)*generatedVideoRounds:]; !strings.HasPrefix(composedCases[0].Group, "composed_") {
		t.Fatalf("the cases after the picked captions' are of %q, want a composed caption's", composedCases[0].Group)
	}
	clip := suite.Cases[:generatedVideoRounds]
	sure, unsure := softShare(clip, []float64{0.9, 0.9}), softShare(clip, []float64{0.4, 0.4})
	if sure <= unsure {
		t.Fatalf("a clip the judge is sure of scores %g and one it is not %g, both answered right; want them apart", sure, unsure)
	}
}
