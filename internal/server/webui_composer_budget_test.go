package server

import (
	"io/fs"
	"regexp"
	"strings"
	"testing"

	"overgo/internal/webuilane"
)

// Composer ratchet (professional GUI campaign, gui-simplify/one-composer):
// every surface that asks the served model for something goes through the
// one composer and the one thread renderer in composer.js, over the one
// event vocabulary this package declares. The ceilings below are the
// measured values when the row landed; they only tighten.
const (
	webuiStreamReaderCeiling = 1    // response.body.getReader(): boot.js sseEvents, the one stream reader
	webuiRawFetchCeiling     = 4    // fetch(: boot.js api client only
	webuiAPIStreamCeiling    = 1    // api.stream(: chat
	webuiJavaScriptCeiling   = 4610 // total lines under webui/
)

// webuiReviewCeiling: the review criteria webuilane.ReviewMeasures counts, each at its
// measured value when the census landed (webui-quality-census); they only tighten.
var webuiReviewCeiling = webuilane.Review{
	SilentFallbacks: 0, WindowDialogs: 0, UnnamedControls: 0, UnnamedButtons: 0,
	InlineStyles: 67, NestedTernaries: 3, TimerLiterals: 6, DebtMarkers: 0,
}

// webuiLargestFileCeiling bounds one file's lines (the review's soft file ceiling is 800).
const webuiLargestFileCeiling = 677

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

func TestWebUIComposerBudget(t *testing.T) {
	sources := webuiJavaScript(t)
	composer, ok := sources["composer.js"]
	if !ok {
		t.Fatal("composer.js is absent")
	}
	for _, needle := range []string{"overgo.thread = thread", "overgo.composer = composer", "overgo.streams = {", "function toolCard", "function mediaCard", "function errorRow", "function thinking"} {
		if !strings.Contains(composer, needle) {
			t.Errorf("composer.js lacks %s", needle)
		}
	}
	// The vocabulary the renderer documents is the one this package declares.
	for _, event := range streamEventTypes {
		if !strings.Contains(composer, `case "`+event+`":`) {
			t.Errorf("composer.js does not render stream event %q", event)
		}
	}
	// Every surface that talks to the served model uses the shared pieces.
	for _, module := range []string{"mod/chat.js", "mod/agent.js"} {
		source := sources[module]
		if !strings.Contains(source, "overgo.composer(") || !strings.Contains(source, "overgo.thread(") {
			t.Errorf("%s does not use the shared composer and thread", module)
		}
		if strings.Contains(source, "getReader()") || strings.Contains(source, "FileReader") {
			t.Errorf("%s parses a wire format or files itself instead of through composer.js", module)
		}
	}
	// The loader lists composer.js after md.js: the thread renders markdown.
	boot := sources["boot.js"]
	if strings.Index(boot, `"/md.js"`) < 0 || strings.Index(boot, `"/composer.js"`) < strings.Index(boot, `"/md.js"`) {
		t.Error("boot.js must load composer.js after md.js")
	}

	readers, fetches, streams, lines := 0, 0, 0, 0
	fetchPattern := regexp.MustCompile(`\bfetch\(`)
	for name, source := range sources {
		readers += strings.Count(source, ".getReader()")
		fetches += len(fetchPattern.FindAllString(source, -1))
		streams += strings.Count(source, "api.stream(")
		lines += strings.Count(source, "\n")
		if name != "boot.js" && fetchPattern.MatchString(source) {
			t.Errorf("%s calls fetch directly; the API client in boot.js owns transport", name)
		}
	}
	if readers > webuiStreamReaderCeiling {
		t.Errorf("stream reader sites = %d, ceiling %d", readers, webuiStreamReaderCeiling)
	}
	if fetches > webuiRawFetchCeiling {
		t.Errorf("raw fetch sites = %d, ceiling %d", fetches, webuiRawFetchCeiling)
	}
	if streams > webuiAPIStreamCeiling {
		t.Errorf("api.stream sites = %d, ceiling %d", streams, webuiAPIStreamCeiling)
	}
	if lines > webuiJavaScriptCeiling {
		t.Errorf("webui JavaScript lines = %d, ceiling %d", lines, webuiJavaScriptCeiling)
	}
	t.Logf("composer budget: readers=%d fetch=%d api.stream=%d lines=%d", readers, fetches, streams, lines)
}

// TestWebUIReviewRatchet holds the client to the review criteria at their
// measured values: a source that adds a silent fallback, a dialog, an
// unnamed control, inline styling, a nested ternary, a timer literal or a
// debt marker fails here, and each ceiling only tightens as rows pay it.
func TestWebUIReviewRatchet(t *testing.T) {
	var review webuilane.Review
	largest := 0
	for name, source := range webuiJavaScript(t) {
		measured := webuilane.ReviewMeasures(source)
		if measured.SilentFallbacks+measured.WindowDialogs+measured.UnnamedControls+measured.UnnamedButtons > 0 {
			t.Logf("%s: %+v", name, measured)
		}
		review = webuilane.Review{
			SilentFallbacks: review.SilentFallbacks + measured.SilentFallbacks, WindowDialogs: review.WindowDialogs + measured.WindowDialogs,
			UnnamedControls: review.UnnamedControls + measured.UnnamedControls, UnnamedButtons: review.UnnamedButtons + measured.UnnamedButtons,
			InlineStyles: review.InlineStyles + measured.InlineStyles, NestedTernaries: review.NestedTernaries + measured.NestedTernaries,
			TimerLiterals: review.TimerLiterals + measured.TimerLiterals, DebtMarkers: review.DebtMarkers + measured.DebtMarkers,
		}
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
		{"largest file lines", largest, webuiLargestFileCeiling},
	} {
		if check.value > check.ceiling {
			t.Errorf("%s = %d, ceiling %d", check.name, check.value, check.ceiling)
		}
	}
	t.Logf("review ratchet: %+v largest=%d", review, largest)
}
