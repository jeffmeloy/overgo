package webuilane

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestMeasureTreeReadsThisRepository measures the checked-out client: the
// front page is one shell, every tab module counts, and the routes the
// client names are a subset of the manifest's routes.
func TestMeasureTreeReadsThisRepository(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := os.ReadFile(filepath.Join(root, "docs", "api_manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	census, err := MeasureTree(root, "head", manifest)
	if err != nil {
		t.Fatal(err)
	}
	if census.Shells != 1 || census.Modules == 0 || census.JavaScriptLines == 0 {
		t.Fatalf("census does not describe the client: %+v", census)
	}
	if census.ClientRoutes == 0 || census.ClientRoutes > census.Routes || census.BearerRoutes > census.Routes {
		t.Fatalf("route counts are inconsistent: %+v", census)
	}
	if census.StreamReaderSites != 1 {
		t.Fatalf("stream reader sites = %d, the ratchet holds one", census.StreamReaderSites)
	}
}

func TestLaneObservationsSeparateProvenFromUnobserved(t *testing.T) {
	output := strings.Join([]string{
		"=== RUN   TestWebUIBrowserFirstRun",
		"    webui_browser_firstrun_test.go:41: first-run journey: model a.gguf at C:/a.gguf",
		"    webui_browser_firstrun_test.go:110: vision leg: the served model accepts no images, so the refusal contract was proven instead",
		"--- PASS: TestWebUIBrowserFirstRun (1.00s)",
		"webui lane: UNAVAILABLE first-run journey: no store",
		"",
	}, "\n")
	proven, unobserved := LaneObservations(output)
	if len(proven) != 3 || proven[0] != "first-run journey: model a.gguf at C:/a.gguf" || proven[2] != "TestWebUIBrowserFirstRun (1.00s)" {
		t.Fatalf("proven = %q", proven)
	}
	if len(unobserved) != 1 || unobserved[0] != "first-run journey: no store" {
		t.Fatalf("unobserved = %q", unobserved)
	}
}

func TestSimplificationReportIsAFunctionOfItsInputs(t *testing.T) {
	fork := Census{Tree: "aaaa", Shells: 2, ListedScripts: 30, Modules: 24, JavaScriptLines: 5000, Routes: 100, ClientRoutes: 40}
	head := Census{Tree: "bbbb", Shells: 1, ListedScripts: 0, Modules: 24, JavaScriptLines: 4747, Routes: 102, ClientRoutes: 50}
	report := SimplificationReport(fork, head, []string{"TestWebUIBrowserFrontPage (0.6s)"}, nil)
	for _, needle := range []string{"aaaa to bbbb", "| Shells (HTML documents) | 2 | 1 | -1 |", "| JavaScript lines | 5000 | 4747 | -253 |", "- TestWebUIBrowserFrontPage (0.6s)"} {
		if !strings.Contains(report, needle) {
			t.Errorf("report lacks %q:\n%s", needle, report)
		}
	}
	if strings.Contains(report, "could not observe") {
		t.Error("report lists an unobserved section without unobserved behaviours")
	}
	if report != SimplificationReport(fork, head, []string{"TestWebUIBrowserFrontPage (0.6s)"}, nil) {
		t.Error("report differs between two renders of the same inputs")
	}
}
