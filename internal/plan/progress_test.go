package plan

import "testing"

// TestAcceptedProgressReadsLandingsFromGit holds progress to what git
// records: a landing of a sibling step or of a prerequisite after the
// checkpoint is progress for the open row, a landing of an unrelated row or
// a commit that names no row is not, and a merge that names no row brings
// none from its side.
func TestAcceptedProgressReadsLandingsFromGit(t *testing.T) {
	t.Parallel()
	document := Plan{Campaign: "progress", Doctrine: "fixture", Items: []Item{
		{ID: "prereq", Status: StatusOpen, Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}}},
		{ID: "row", Status: StatusOpen, Steps: []Step{
			{ID: "first", Status: StatusOpen, Verify: "go test ./..."},
			{ID: "second", Status: StatusOpen, Verify: "go test ./...", DependsOn: []string{"prereq/do"}},
		}},
		{ID: "other", Status: StatusOpen, Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}}},
	}}
	repository := planRepository(t, document)
	checkpoint := git(t, repository, "rev-parse", "HEAD")
	progress := func() bool {
		t.Helper()
		moved, err := AcceptedProgressSince(t.Context(), repository, document, "row", "second", checkpoint)
		if err != nil {
			t.Fatal(err)
		}
		return moved
	}
	land := func(message string) {
		t.Helper()
		git(t, repository, "commit", "-q", "--allow-empty", "-m", message)
	}
	land("unrelated work\n\n" + LandedRowTrailer + ": other/do")
	land("a commit that names no row")
	git(t, repository, "checkout", "-qb", "side")
	land("side work\n\n" + LandedRowTrailer + ": prereq/do")
	git(t, repository, "checkout", "-q", "-")
	git(t, repository, "merge", "-q", "--no-ff", "-m", "merge side", "side")
	if progress() {
		t.Fatal("an unrelated landing, a bare commit or a merged side landing counted as progress")
	}
	land("the prerequisite\n\n" + LandedRowTrailer + ": prereq/do")
	if !progress() {
		t.Fatal("a landed prerequisite did not count as progress")
	}
	checkpoint = git(t, repository, "rev-parse", "HEAD")
	land("a sibling\n\n" + LandedRowTrailer + ": row/first")
	if !progress() {
		t.Fatal("a landed sibling step did not count as progress")
	}
	if _, err := AcceptedProgressSince(t.Context(), repository, document, "row", "second", "not-a-commit"); err == nil {
		t.Fatal("an inexact checkpoint was accepted")
	}
}
