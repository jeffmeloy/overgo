package main

import (
	"encoding/json"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gitauthority"
	"overgo/internal/testcheck"
)

// testDataOrigins is docs/verification/test-data-origins.json: for each record of test data, how its data is
// obtained (named sources), re-created (a tool the repository keeps and the commit it ran at), kept as a golden still
// needed before parity (a script docs/verification/goldens-kept.json keeps), re-made by an open row, or recorded in the
// record's own bytes.
type testDataOrigins struct {
	Records []struct {
		Path      string   `json:"path"`
		Obtained  []string `json:"obtained"`
		Recreated *struct {
			Tool   string `json:"tool"`
			Commit string `json:"commit"`
		} `json:"recreated"`
		Golden   string `json:"golden"`
		RemadeBy string `json:"remade_by"`
		Recorded string `json:"recorded"`
	} `json:"records"`
}

// namesArtifact says a JSON document names a stored artifact by its id (kind:sha256:digest) in one of its strings.
func namesArtifact(document []byte) bool {
	for part := range strings.SplitSeq(string(document), `"`) {
		if strings.Contains(part, ":sha256:") {
			if _, err := artifact.ParseID(part); err == nil {
				return true
			}
		}
	}
	return false
}

// TestEveryTestsDataNamesItsOrigin holds every record of test data that names a stored artifact (a JSON file under a
// testdata folder or fixtures) to declaring how its data is had again (data-origin, the owner's review of 2026-11-20):
// obtained from named sources, re-created by a tool the repository keeps at a commit Git holds, kept as a golden
// goldens-kept.json keeps, re-made by a row the plan holds open, or recorded in its own bytes. A record that declares
// none, or names a tool, commit, golden or row that is not there, fails; so does a declaration for a record that is
// gone.
func TestEveryTestsDataNamesItsOrigin(t *testing.T) {
	t.Parallel()
	repo := filepath.Join("..", "..")
	read := func(path string, target any) {
		t.Helper()
		testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(filepath.Join(repo, filepath.FromSlash(path))))(t), target))
	}
	var origins testDataOrigins
	read("docs/verification/test-data-origins.json", &origins)
	var kept goldensKept
	read("docs/verification/goldens-kept.json", &kept)
	var plan struct {
		Items []struct {
			ID string `json:"id"`
		} `json:"items"`
	}
	read("docs/plan.json", &plan)
	goldens, rows := map[string]bool{}, map[string]bool{}
	for _, golden := range kept.Goldens {
		goldens[golden.Name] = true
	}
	for _, item := range plan.Items {
		rows[item.ID] = true
	}
	declared := map[string]bool{}
	for _, record := range origins.Records {
		declared[record.Path] = true
		if _, err := os.Stat(filepath.Join(repo, filepath.FromSlash(record.Path))); err != nil {
			t.Errorf("%s declares an origin for data that is not there: %v", record.Path, err)
		}
		if len(record.Obtained) == 0 && record.Recreated == nil && record.Golden == "" && record.RemadeBy == "" && record.Recorded == "" {
			t.Errorf("%s declares no origin", record.Path)
		}
		if slices.Contains(record.Obtained, "") {
			t.Errorf("%s names an empty source", record.Path)
		}
		if recreated := record.Recreated; recreated != nil {
			if _, err := os.Stat(filepath.Join(repo, filepath.FromSlash(recreated.Tool))); err != nil {
				t.Errorf("%s is re-created by %s, which the repository does not keep: %v", record.Path, recreated.Tool, err)
			}
			if _, err := gitauthority.Query(t.Context(), repo, "cat-file", "-e", recreated.Commit+"^{commit}"); err != nil {
				t.Errorf("%s is re-created at %s, a commit Git does not hold: %v", record.Path, recreated.Commit, err)
			}
		}
		if record.Golden != "" && !goldens[record.Golden] {
			t.Errorf("%s names the golden %s, which goldens-kept.json does not keep", record.Path, record.Golden)
		}
		if record.RemadeBy != "" && !rows[record.RemadeBy] {
			t.Errorf("%s is re-made by %s, which the plan does not hold open", record.Path, record.RemadeBy)
		}
	}
	for _, root := range []string{"internal", "cmd", "fixtures"} {
		testcheck.NoError(t, filepath.WalkDir(filepath.Join(repo, root), func(path string, entry fs.DirEntry, err error) error {
			if err != nil || entry.IsDir() || filepath.Ext(path) != ".json" {
				return err
			}
			relative := filepath.ToSlash(testcheck.Must(filepath.Rel(repo, path))(t))
			if root != "fixtures" && !strings.Contains(relative, "/testdata/") {
				return nil
			}
			if !declared[relative] && namesArtifact(testcheck.Must(os.ReadFile(path))(t)) {
				t.Errorf("%s names stored artifacts and declares no origin in docs/verification/test-data-origins.json", relative)
			}
			return nil
		}))
	}
}
