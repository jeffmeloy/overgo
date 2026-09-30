package main

import (
	"bytes"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/gitauthority"
)

// TestMediaReportIsRenderedOnDemand pins that the media report left the tree:
// neither docs/MEDIA_REPORT.md nor its typed projection docs/media_report.json
// is tracked or present, since go run ./cmd/compatibility -media markdown|json
// renders them from the store and the verification specifications. The render
// is held where its code lives, in cmd/compatibility (the rendered-report
// contract and TestAcceptedProtocolCoverageProjection over the real store);
// a main package cannot be imported here.
func TestMediaReportIsRenderedOnDemand(t *testing.T) {
	t.Parallel()
	root := filepath.Join("..", "..")
	reports := []string{"docs/MEDIA_REPORT.md", "docs/media_report.json"}
	tracked, err := gitauthority.Query(t.Context(), root, append([]string{"ls-files", "--"}, reports...)...)
	if err != nil {
		t.Fatal(err)
	}
	if len(tracked) != 0 {
		t.Fatalf("committed: %s; the media report renders with go run ./cmd/compatibility -media markdown|json", bytes.Fields(tracked))
	}
	// git reads its index in a child process the test cache does not see; the
	// stat makes the files' presence an input of a cached result.
	for _, report := range reports {
		if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(report))); !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("%s is present (%v); the media report renders with go run ./cmd/compatibility -media markdown|json", report, err)
		}
	}
}
