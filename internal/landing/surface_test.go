package landing

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestMediaMovesNeedTheirReconciliation holds a change to the pinned media
// runtime to the reconciliation that answers for it: a production file under
// a pinned path is refused alone and admitted beside the reviewed delta, and
// a test file there is not runtime source.
func TestMediaMovesNeedTheirReconciliation(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	if err := os.MkdirAll(filepath.Join(repo, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, filepath.FromSlash(mediaMergedDocument)), []byte(`{"runtime_paths":["internal/media"]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	moved := "internal/media/decode.go"
	if quiet, err := mediaMoves(repo, []string{moved}); quiet || err == nil || !strings.Contains(err.Error(), moved) {
		t.Fatalf("an unreconciled runtime move = (%v, %v)", quiet, err)
	}
	if quiet, err := mediaMoves(repo, []string{moved, mediaReviewedDeltas}); quiet || err != nil {
		t.Fatalf("a reconciled runtime move = (%v, %v)", quiet, err)
	}
	if quiet, err := mediaMoves(repo, []string{"internal/media/decode_test.go"}); !quiet || err != nil {
		t.Fatalf("a runtime test file = (%v, %v)", quiet, err)
	}
}
