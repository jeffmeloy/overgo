package repoanalysis

import (
	"io/fs"
	"path/filepath"
	"strings"
	"testing"
)

// TestDocsHoldNoUnplannedBinaries pins the reference-paper decision: docs
// holds only the planned document classes — prose, structured records, and
// the published media evidence areas — so reference material such as papers
// enters through a plan row, never as a stray binary.
func TestDocsHoldNoUnplannedBinaries(t *testing.T) {
	root := filepath.Join("..", "..", "docs")
	planned := map[string]bool{
		".md": true, ".json": true, ".jsonl": true, ".txt": true, ".jinja": true,
	}
	media := map[string]bool{
		".png": true, ".gif": true, ".wav": true, ".svg": true,
	}
	mediaAreas := []string{"assets", "media_samples", "verification"}
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil || entry.IsDir() {
			return walkErr
		}
		relative, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		relative = filepath.ToSlash(relative)
		if relative == ".loop_state" || relative == ".bounded_request" {
			// Loop continuity state and the bounded-request marker: plain
			// text, dot-named by the driver and ignored by git.
			return nil
		}
		extension := strings.ToLower(filepath.Ext(relative))
		if planned[extension] {
			return nil
		}
		for _, area := range mediaAreas {
			if strings.HasPrefix(relative, area+"/") && media[extension] {
				return nil
			}
		}
		t.Errorf("docs holds unplanned file %q: reference material enters through a plan row", relative)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
