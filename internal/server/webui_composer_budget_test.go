package server

import (
	"io/fs"
	"strings"
	"testing"

	"overgo/internal/webuilane"
)

// webuiReviewCeiling: the review criteria webuilane.ReviewMeasures counts, each at its
// measured value when the census landed (webui-quality-census); they only tighten.
var webuiReviewCeiling = webuilane.Review{
	SilentFallbacks: 0, WindowDialogs: 0, UnnamedControls: 0, UnnamedButtons: 0,
	InlineStyles: 0, NestedTernaries: 0, TimerLiterals: 0, DebtMarkers: 0,
}

func webuiJavaScript(t *testing.T) map[string]string {
	t.Helper()
	sources := map[string]string{}
	if err := fs.WalkDir(webuiFS, ".", func(name string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !entry.IsDir() && strings.HasSuffix(name, ".js") {
			data, err := fs.ReadFile(webuiFS, name)
			if err != nil {
				return err
			}
			sources[name] = string(data)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	return sources
}

// TestWebUIComposerBudget holds the client's one stream renderer to the
// event vocabulary this package declares (stream_events.go): every event
// the server can send has a case in composer.js.
func TestWebUIComposerBudget(t *testing.T) {
	t.Parallel()
	composer, ok := webuiJavaScript(t)["composer.js"]
	if !ok {
		t.Fatal("composer.js is absent")
	}
	for _, event := range streamEventTypes {
		if !strings.Contains(composer, `case "`+event+`":`) {
			t.Errorf("composer.js does not render stream event %q", event)
		}
	}
}

// TestWebUIReviewRatchet holds the client to the review criteria at their
// measured values: a source that adds a silent fallback, a dialog, an
// unnamed control, inline styling, a nested ternary, a timer literal or a
// debt marker fails here, and each ceiling only tightens as rows pay it.
func TestWebUIReviewRatchet(t *testing.T) {
	t.Parallel()
	var review webuilane.Review
	largest := 0
	for name, source := range webuiJavaScript(t) {
		measured := webuilane.ReviewMeasures(source)
		if measured.SilentFallbacks+measured.WindowDialogs+measured.UnnamedControls+measured.UnnamedButtons > 0 {
			t.Logf("%s: %+v", name, measured)
		}
		review.Add(measured)
		largest = max(largest, strings.Count(source, "\n"))
	}
	for _, check := range []struct {
		name           string
		value, ceiling int
	}{
		{"silent fallbacks", review.SilentFallbacks, webuiReviewCeiling.SilentFallbacks},
		{"window dialogs", review.WindowDialogs, webuiReviewCeiling.WindowDialogs},
		{"unnamed controls", review.UnnamedControls, webuiReviewCeiling.UnnamedControls},
		{"unnamed buttons", review.UnnamedButtons, webuiReviewCeiling.UnnamedButtons},
		{"inline styles", review.InlineStyles, webuiReviewCeiling.InlineStyles},
		{"nested ternaries", review.NestedTernaries, webuiReviewCeiling.NestedTernaries},
		{"timer literals", review.TimerLiterals, webuiReviewCeiling.TimerLiterals},
		{"debt markers", review.DebtMarkers, webuiReviewCeiling.DebtMarkers},
	} {
		if check.value > check.ceiling {
			t.Errorf("%s = %d, ceiling %d", check.name, check.value, check.ceiling)
		}
	}
	t.Logf("review ratchet: %+v largest=%d", review, largest)
}
