package repoanalysis

import (
	"slices"
	"testing"
)

// TestARenamedTestsCitationsFollowIt holds the rewrite a landing applies to a renamed test's citations (internal/landing
// holds the landing's side): each whole-word citation of the old name becomes the new one, a longer name that begins
// with it and one it ends are left, and the lines citing a name are those that cite it whole.
func TestARenamedTestsCitationsFollowIt(t *testing.T) {
	t.Parallel()
	text := "- a rule (check: internal/p TestOld, internal/p TestOldest)\n- another (check: TestOld)\nXTestOld is no citation\n"
	rewritten, count := FollowRename(text, "TestOld", "TestNew")
	want := "- a rule (check: internal/p TestNew, internal/p TestOldest)\n- another (check: TestNew)\nXTestOld is no citation\n"
	if rewritten != want || count != 2 {
		t.Fatalf("the rewrite gave %d citations:\n%s\nwant 2:\n%s", count, rewritten, want)
	}
	if lines := CitingLines(text, "TestOld"); !slices.Equal(lines, []string{"- a rule (check: internal/p TestOld, internal/p TestOldest)", "- another (check: TestOld)"}) {
		t.Fatalf("the lines citing TestOld are %q", lines)
	}
	if unchanged, count := FollowRename(text, "TestGone", "TestNew"); unchanged != text || count != 0 {
		t.Fatalf("a name nothing cites rewrote %d citations", count)
	}
}
