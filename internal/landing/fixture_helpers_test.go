package landing

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/gosyntax"
	"overgo/internal/testcheck"
)

// heldTurn holds the lane turn for a test's goroutine: it closes held once the turn is its own and keeps it until let
// closes.
func heldTurn(t *testing.T, held, let chan struct{}) func() {
	return func() {
		if err := HoldLaneTurn(func() error {
			close(held)
			<-let
			return nil
		}); err != nil {
			t.Error(err)
		}
	}
}

// heldFile is what the checkout holds at the slash path name, or nothing where it holds no such file.
func heldFile(checkout *sliceFixture, name string) string {
	data, err := os.ReadFile(filepath.Join(checkout.repo, filepath.FromSlash(name)))
	if err != nil {
		return ""
	}
	return string(data)
}

// packageSnapshot is a syntax snapshot of one package file, internal/p/p.go, holding source.
func packageSnapshot(t *testing.T, source string) gosyntax.Snapshot {
	t.Helper()
	result := testcheck.Must((gosyntax.Snapshot{}).Overlay(map[string][]byte{"internal/p/p.go": []byte(source)}))(t)
	return result
}
