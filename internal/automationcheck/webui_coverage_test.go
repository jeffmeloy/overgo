package automationcheck

import (
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestWebUIPathsCoverWorkspaceDeclarations derives the lane's path obligations
// from the source rather than a list: every document the server embeds
// (the workbench's assets and the workspace declarations it is built from)
// and every test file that opens a browser must select a browser lane. The
// manifest and schema documents, and two browser legs named outside the
// webui_browser_ convention, once changed without running a browser test.
func TestWebUIPathsCoverWorkspaceDeclarations(t *testing.T) {
	t.Parallel()
	root := filepath.Join("..", "..")
	embed := regexp.MustCompile(`(?m)^//go:embed (.+)$`)
	opensBrowser := regexp.MustCompile(`webuilane\.(?:Open|CaptureStates)\(`)
	var embedded, browserLegs []string
	server := filepath.Join(root, "internal", "server")
	entries, err := os.ReadDir(server)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".go") || strings.HasSuffix(entry.Name(), "_test.go") {
			continue
		}
		source, err := os.ReadFile(filepath.Join(server, entry.Name()))
		if err != nil {
			t.Fatal(err)
		}
		for _, match := range embed.FindAllStringSubmatch(string(source), -1) {
			for pattern := range strings.FieldsSeq(match[1]) {
				embedded = append(embedded, path.Join("internal/server", pattern))
			}
		}
	}
	for _, tree := range []string{"internal", "cmd"} {
		if err := filepath.WalkDir(filepath.Join(root, tree), func(file string, entry fs.DirEntry, err error) error {
			if err != nil || entry.IsDir() || !strings.HasSuffix(file, "_test.go") {
				return err
			}
			source, err := os.ReadFile(file)
			if err != nil {
				return err
			}
			if opensBrowser.Match(source) {
				relative, err := filepath.Rel(root, file)
				if err != nil {
					return err
				}
				browserLegs = append(browserLegs, filepath.ToSlash(relative))
			}
			return nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	if len(embedded) < 3 || len(browserLegs) == 0 {
		t.Fatalf("found %d embedded documents and %d browser legs; the scan lost its sources", len(embedded), len(browserLegs))
	}
	for _, document := range embedded {
		// A directory embeds everything beneath it; one file under it stands for the rest.
		if !WebUIPaths([]string{document}) && !WebUIPaths([]string{document + "/index.html"}) {
			t.Errorf("changing embedded %s selects no browser lane", document)
		}
	}
	for _, leg := range browserLegs {
		if !WebUIPaths([]string{leg}) && !ModelJourneyPaths([]string{leg}) {
			t.Errorf("changing browser leg %s selects no browser lane; name it webui_browser_*", leg)
		}
	}
}
