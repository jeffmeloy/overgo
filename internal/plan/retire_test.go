package plan

import (
	"path/filepath"
	"slices"
	"testing"
)

// TestOwnerDecisionRetiresRow holds plan rows to plain data: a landing may
// delete rows beside the one it completes, and a commit may delete a row
// outright; the commit message, not a guard, records why. Completion credit
// is unchanged: only the served row counts as completed.
func TestOwnerDecisionRetiresRow(t *testing.T) {
	t.Parallel()
	t.Run("a landing retires unrelated rows beside its own", func(t *testing.T) {
		parent := standardCompletionPlan()
		parent.Items = append(parent.Items, Item{ID: "declined", Status: StatusOpen})
		fixture := newCompletionFixture(t, parent, "root", "do")
		fixture.child.Items = slices.DeleteFunc(slices.Clone(fixture.child.Items), func(item Item) bool { return item.ID == "declined" })
		fixture.commit(fixture.canonicalMessage(), true)
		authority, err := resolveFixture(fixture, fixture.child, "HEAD")
		if err != nil {
			t.Fatalf("a landing retiring a row was refused: %v", err)
		}
		if !authority.completed("root/do") || authority.completed("declined/do") {
			t.Fatal("retiring a row changed completion credit")
		}
	})
	t.Run("a commit deletes a row", func(t *testing.T) {
		fixture := newCompletionFixture(t, standardCompletionPlan(), "root", "do")
		fixture.commit(fixture.canonicalMessage(), true)
		withRow := fixture.child
		withRow.Items = append(slices.Clone(withRow.Items), Item{
			ID: "superseded", Status: StatusOpen,
			Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./..."}},
		})
		path := filepath.Join(fixture.repository, filepath.FromSlash(Path))
		if err := Save(path, withRow); err != nil {
			t.Fatal(err)
		}
		runGit(t, fixture.repository, nil, "add", "--", Path)
		runGit(t, fixture.repository, nil, "commit", "-q", "-m", "file a row")
		if err := Save(path, fixture.child); err != nil {
			t.Fatal(err)
		}
		runGit(t, fixture.repository, nil, "add", "--", Path)
		runGit(t, fixture.repository, nil, "commit", "-q", "-m", "retire the row: superseded")
		if _, err := resolveFixture(fixture, fixture.child, "HEAD"); err != nil {
			t.Fatalf("a commit retiring a row was refused: %v", err)
		}
	})
}
