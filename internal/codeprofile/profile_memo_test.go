package codeprofile

import (
	"reflect"
	"strconv"
	"testing"

	"overgo/internal/repoanalysis"
)

// TestProfileBuildReusesUnchangedFileFacts holds the incremental profile to
// the fresh one: a snapshot that changes one file computes facts for that
// file alone, and its profile equals one computed with no remembered facts.
func TestProfileBuildReusesUnchangedFileFacts(t *testing.T) {
	t.Parallel()
	files := map[string][]byte{
		"internal/memoprobe/a.go":      []byte("package memoprobe\nfunc A(v int) int { if v > 0 { return v }; return -v }\n"),
		"internal/memoprobe/b.go":      []byte("package memoprobe\nfunc B(v int) int { if v > 0 { return v }; return -v }\n"),
		"internal/memoprobe/a_test.go": []byte("package memoprobe\nimport \"testing\"\nfunc TestA(t *testing.T) { A(1) }\n"),
	}
	base, err := (repoanalysis.SourceSnapshot{}).Overlay(files)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Build(base); err != nil {
		t.Fatal(err)
	}
	candidate, err := base.Overlay(map[string][]byte{"internal/memoprobe/b.go": []byte("package memoprobe\nfunc B(v int) int { return v * 2 }\n")})
	if err != nil {
		t.Fatal(err)
	}
	computed := func(snapshot repoanalysis.SourceSnapshot) map[string]bool {
		present := map[string]bool{}
		for _, source := range snapshot.Files {
			_, found := fileFactsMemo.Load(source.Path + "\x00" + source.ContentID + "\x00" + strconv.FormatBool(source.Test))
			present[source.Path] = found
		}
		return present
	}
	if before := computed(candidate); !before["internal/memoprobe/a.go"] || !before["internal/memoprobe/a_test.go"] || before["internal/memoprobe/b.go"] {
		t.Fatalf("remembered facts before the candidate build = %v", before)
	}
	incremental, err := Build(candidate)
	if err != nil {
		t.Fatal(err)
	}
	for _, source := range candidate.Files {
		fileFactsMemo.Delete(source.Path + "\x00" + source.ContentID + "\x00" + strconv.FormatBool(source.Test))
	}
	fresh, err := Build(candidate)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(incremental, fresh) || fresh.DuplicateExcessNodes != 0 || len(fresh.Functions) != 3 {
		t.Fatalf("incremental profile %+v differs from fresh %+v", incremental, fresh)
	}
}
