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
	"overgo/internal/dataroot"
	"overgo/internal/gitauthority"
	"overgo/internal/testcheck"
)

// testDataOrigins is docs/verification/test-data-origins.json: for each record of test data, how its data is
// obtained (named sources), re-created (a tool the repository keeps and the commit it ran at), kept as a golden still
// needed before parity (a script docs/verification/goldens-kept.json keeps), its bytes kept (each artifact it names, a
// file under a folder of the data root's references), re-made by an open row, or recorded in the record's own bytes.
type testDataOrigins struct {
	Records []struct {
		Path      string   `json:"path"`
		Obtained  []string `json:"obtained"`
		Recreated *struct {
			Tool   string `json:"tool"`
			Commit string `json:"commit"`
		} `json:"recreated"`
		Golden string `json:"golden"`
		Kept   *struct {
			Folder string            `json:"folder"`
			Files  map[string]string `json:"files"`
		} `json:"kept"`
		RemadeBy string `json:"remade_by"`
		Recorded string `json:"recorded"`
	} `json:"records"`
}

// artifactIDs are the stored artifacts a JSON document names by their ids (kind:sha256:digest) in its strings, each
// once, in the order it names them.
func artifactIDs(document []byte) []artifact.ID {
	var ids []artifact.ID
	for part := range strings.SplitSeq(string(document), `"`) {
		if strings.Contains(part, ":sha256:") {
			if id, err := artifact.ParseID(part); err == nil && !slices.Contains(ids, id) {
				ids = append(ids, id)
			}
		}
	}
	return ids
}

// TestEveryTestsDataNamesItsOrigin holds every record of test data that names a stored artifact (a JSON file under a
// testdata folder or fixtures) to declaring how its data is had again (data-origin, the owner's review of 2026-11-20):
// obtained from named sources, re-created by a tool the repository keeps at a commit Git holds, kept as a golden
// goldens-kept.json keeps, its bytes kept in the data root (TestAGoldenOriginIsItsDataProducer), re-made by a row the
// plan holds open, or recorded in its own bytes. A record that declares
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
		if len(record.Obtained) == 0 && record.Recreated == nil && record.Golden == "" && record.Kept == nil && record.RemadeBy == "" && record.Recorded == "" {
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
			if !declared[relative] && len(artifactIDs(testcheck.Must(os.ReadFile(path))(t))) > 0 {
				t.Errorf("%s names stored artifacts and declares no origin in docs/verification/test-data-origins.json", relative)
			}
			return nil
		}))
	}
}

// TestAGoldenOriginIsItsDataProducer holds each record of test data to a way back that makes its bytes again
// (oracle-origin, SQA-451): where a record names a golden and its data names the files that produced it (file:sha256:
// ids, as the MOSS oracle's contract names the script that wrote each case), the golden goldens-kept.json keeps is one
// of them by its digest, since a script that writes its own digest into what it saves makes other bytes once it
// changes; and where a record declares its bytes kept, every tensor set its data names (its cases and their inputs, whose
// identity is their file's bytes; a corpus of them is their set) is a file under that folder of the data root's references
// that hashes to it.
func TestAGoldenOriginIsItsDataProducer(t *testing.T) {
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
	digests := map[string]string{}
	for _, golden := range kept.Goldens {
		digests[golden.Name] = golden.SHA256
	}
	references := testcheck.Must(dataroot.Resolve(repo))(t).References()
	for _, record := range origins.Records {
		ids := artifactIDs(testcheck.Must(os.ReadFile(filepath.Join(repo, filepath.FromSlash(record.Path))))(t))
		var producers []string
		for _, id := range ids {
			if id.Kind() == artifact.KindFile {
				producers = append(producers, id.DigestHex())
			}
		}
		if record.Golden != "" && len(producers) > 0 && !slices.Contains(producers, digests[record.Golden]) {
			t.Errorf("%s is re-made by %s, kept at %.12s, which is none of the producers its data names (%d); declare its bytes kept instead",
				record.Path, record.Golden, digests[record.Golden], len(producers))
		}
		if record.Kept == nil {
			continue
		}
		for _, id := range ids {
			if id.Kind() != artifact.KindTensorSet {
				continue
			}
			name, held := record.Kept.Files[id.String()]
			if !held {
				t.Errorf("%s names %s, which it keeps nowhere", record.Path, id)
				continue
			}
			path := filepath.Join(references, filepath.FromSlash(record.Kept.Folder), filepath.FromSlash(name))
			if got := testcheck.Must(artifact.IdentifyFile(artifact.KindFile, path))(t); got.DigestHex() != id.DigestHex() {
				t.Errorf("%s keeps %s at %s, whose bytes hash %.12s", record.Path, id, path, got.DigestHex())
			}
		}
	}
}
