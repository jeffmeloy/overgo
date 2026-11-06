package landing

import (
	"slices"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/webuilane"
)

// TestAMediaRuntimeChangeReachesOnlyTheBrowserTestsThatCallIt holds the browser lane to the tests a change calls: the
// server imports every runtime, and a path that enters the lane only through those imports runs just the browser tests
// whose files' own imports take it in, by name, a journey in the same file left out; one no browser test file takes in
// reaches nothing; and a change to the server's own code, or any path that reaches the lane whole, runs every test. The
// webui lane names its tests as the browser lane runs them.
func TestAMediaRuntimeChangeReachesOnlyTheBrowserTestsThatCallIt(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	writeModule(t, repo, map[string]string{
		"go.mod":                 "module fixture\n\ngo 1.26\n",
		"media/media.go":         "package media\n\nimport \"fixture/pixels\"\n\nvar M = pixels.P\n",
		"pixels/pixels.go":       "package pixels\n\nvar P = 1\n",
		"text/text.go":           "package text\n\nvar T = 1\n",
		"server/server.go":       "package server\n\nimport (\n\t\"fixture/media\"\n\t\"fixture/text\"\n)\n\nvar S = media.M + text.T\n",
		"server/page_test.go":    "package server\n\nimport \"testing\"\n\nfunc TestWebUIBrowserPage(t *testing.T) {}\n",
		"server/picture_test.go": "package server\n\nimport (\n\t\"testing\"\n\n\t\"fixture/media\"\n)\n\nfunc TestWebUIBrowserPicture(t *testing.T) { _ = media.M }\n\nfunc TestModelJourneyPicture(t *testing.T) {}\n",
	})
	candidate := lane{name: "webui", packages: []string{"./server"}, tests: "server", testPrefix: webuilane.BrowserTestPrefix}
	reaches := testcheck.Must(discover(repo, candidate))(t)
	for _, change := range []struct {
		paths []string
		by    string
		tests []string
	}{
		{[]string{"media/media.go"}, "media/media.go through server", []string{"TestWebUIBrowserPicture"}},
		{[]string{"pixels/pixels.go"}, "pixels/pixels.go through server", []string{"TestWebUIBrowserPicture"}},
		{[]string{"text/text.go"}, "", nil},
		{[]string{"text/text.go", "media/media.go"}, "media/media.go through server", []string{"TestWebUIBrowserPicture"}},
		{[]string{"server/server.go"}, "server/server.go", nil},
		{[]string{"media/media.go", "server/page_test.go"}, "server/page_test.go", nil},
	} {
		got := reaches(candidate, change.paths)
		if got.by != change.by || !slices.Equal(got.tests, change.tests) {
			t.Errorf("a change to %v reached the lane by %q for %v, want %q for %v", change.paths, got.by, got.tests, change.by, change.tests)
		}
	}
	webui := lanes[slices.IndexFunc(lanes, func(candidate lane) bool { return candidate.name == "webui" })]
	if webui.tests != "internal/server" || webui.testPrefix != webuilane.BrowserTestPrefix {
		t.Errorf("the webui lane names its tests in %q by %q, want internal/server by %q", webui.tests, webui.testPrefix, webuilane.BrowserTestPrefix)
	}
}
