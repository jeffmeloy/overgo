package inference

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

// flashNextMissing is the first of a Flash-Next check's files that is not installed, which skips the check, or none,
// where each is present and the check opens them, an open that fails failing it with its cause (flash-open-fails,
// SQA-453); a file that cannot be read for another cause is an error, which fails the check.
func flashNextMissing(paths ...string) (string, error) {
	for _, path := range paths {
		if _, err := os.Stat(path); errors.Is(err, fs.ErrNotExist) {
			return path, nil
		} else if err != nil {
			return "", err
		}
	}
	return "", nil
}

// requireFlashNext skips t where one of the files is not installed (flashNextMissing) and fails it where one cannot be
// read; it returns where each is present, the check's open of them then holding it.
func requireFlashNext(t testing.TB, paths ...string) {
	t.Helper()
	if missing := testcheck.Must(flashNextMissing(paths...))(t); missing != "" {
		t.Skipf("the check needs %s, which is not installed", missing)
	}
}

// TestAFlashNextCheckpointThatCannotOpenFailsItsCheck holds the Flash-Next checks' prerequisite to the files' presence
// alone (flash-open-fails, SQA-453): a checkpoint that is not installed skips a check, naming it, while an installed one,
// though its header cannot open, is no missing prerequisite, so the check opens it and fails with the open's cause.
// TestFlashNextRoundIsReadyAtOpen and
// TestFlashNextWindowRowsAreTheSingleTokenSteps had skipped on every error their open returned, so an installed
// checkpoint that could not open wrote no line naming them and left a read of their verdicts pending.
func TestAFlashNextCheckpointThatCannotOpenFailsItsCheck(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	model, draft := filepath.Join(root, "flash-next.gguf"), filepath.Join(root, "draft.gguf")
	testcheck.NoError(t, os.WriteFile(model, []byte("not a GGUF header"), 0o600))
	if missing := testcheck.Must(flashNextMissing(model, draft))(t); missing != draft {
		t.Errorf("with the draft not installed the check's prerequisite names %q; want the draft, which skips it", missing)
	}
	testcheck.NoError(t, os.WriteFile(draft, []byte("not a GGUF header either"), 0o600))
	if missing := testcheck.Must(flashNextMissing(model, draft))(t); missing != "" {
		t.Errorf("with both installed, though neither opens, the prerequisite names %q missing; want none, the open failing the check", missing)
	}
}
