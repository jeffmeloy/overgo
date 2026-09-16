package webuilane

import (
	"strings"
	"testing"
)

func TestLaneObservationsSeparateProvenFromUnobserved(t *testing.T) {
	output := strings.Join([]string{
		"=== RUN   TestWebUIBrowserFirstRun",
		"    webui_browser_firstrun_test.go:41: first-run journey: model a.gguf at C:/a.gguf",
		"    webui_browser_firstrun_test.go:110: vision leg: the served model accepts no images, so the refusal contract was proven instead",
		"--- PASS: TestWebUIBrowserFirstRun (1.00s)",
		"webui lane: UNAVAILABLE first-run journey: no store",
		"",
	}, "\n")
	if err := LaneVerdict(output, []string{"first-run journey: model a.gguf at C:/a.gguf", "vision leg: the served model accepts no images", "TestWebUIBrowserFirstRun (1.00s)"}); err != nil {
		t.Fatalf("proven observation refused: %v", err)
	}
	if err := LaneVerdict(output, []string{"first-run journey: no store"}); err == nil {
		t.Fatal("unavailable observation accepted")
	}
}
