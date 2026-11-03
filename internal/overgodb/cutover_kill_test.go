package overgodb

import (
	"os"
	"strings"
	"testing"
)

// TestAKillBetweenCutoverRenamesStartsNoEmptyStore: a cutover killed after it moved the live store aside and before it
// moved the candidate in leaves the store at its superseded name and nothing at its own. Create there refuses and names
// the store moved aside, where it had started an empty store at sequence 0 beside it, and Open refuses as it did. Moved
// back, the store opens at the head it held.
func TestAKillBetweenCutoverRenamesStartsNoEmptyStore(t *testing.T) {
	t.Parallel()
	root, head := committedJournal(t, false)
	aside := root + supersededMarker + head.String()[:12]
	if err := os.Rename(root, aside); err != nil {
		t.Fatal(err)
	}
	if store, err := Create(root); err == nil || !strings.Contains(err.Error(), aside) {
		if store != nil {
			_ = store.Close()
		}
		t.Fatalf("Create beside a store a cutover moved aside gave %v, want a refusal naming %s", err, aside)
	}
	if _, err := os.Stat(root); !os.IsNotExist(err) {
		t.Fatalf("the refused Create left something at the store's name: %v", err)
	}
	if _, err := Open(root); err == nil {
		t.Fatal("Open found a store where the cutover left none")
	}
	if err := os.Rename(aside, root); err != nil {
		t.Fatal(err)
	}
	store, err := Create(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if got, sequence := store.Head(); got != head || sequence != tailCommits {
		t.Fatalf("the store moved back opens at %s@%d, want %s@%d", got, sequence, head, tailCommits)
	}
}
