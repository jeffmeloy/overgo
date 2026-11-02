package plan

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"

	"overgo/internal/atomicfile"
	"overgo/internal/processlock"
)

// parkedFile holds the rows the driver gave up on, in the worktree's state directory.
const parkedFile = "parked_rows.json"

// parkedRows is what the driver parked and the commit it parked at. A commit clears them, since a commit is what can
// have changed why a row failed.
type parkedRows struct {
	Head string   `json:"head"`
	Rows []string `json:"rows"`
}

// ParkRow records a row that kept failing, so dispatch passes over it until the next commit.
func ParkRow(ctx context.Context, root, row string) error {
	repository, head, err := resolveRevision(ctx, root, gitHeadRevision)
	if err != nil {
		return err
	}
	rows, err := readParked(repository, head)
	if err != nil {
		return err
	}
	if slices.Contains(rows, row) {
		return nil
	}
	data, err := json.Marshal(parkedRows{Head: head, Rows: append(rows, row)})
	if err != nil {
		return err
	}
	if err := processlock.EnsureStateDirectory(repository); err != nil {
		return err
	}
	return atomicfile.Write(filepath.Join(repository, processlock.StateDirectory, parkedFile), data, stopFileMode)
}

// readParked returns the rows parked at head: rows parked at any other commit are cleared.
func readParked(repository, head string) ([]string, error) {
	raw, err := os.ReadFile(filepath.Join(repository, processlock.StateDirectory, parkedFile))
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var parked parkedRows
	if err := json.Unmarshal(raw, &parked); err != nil {
		return nil, err
	}
	if parked.Head != head {
		return nil, nil
	}
	return parked.Rows, nil
}
