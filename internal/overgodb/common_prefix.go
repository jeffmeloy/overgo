package overgodb

import (
	"context"
	"errors"

	"overgo/internal/artifact"
)

// CommitPrefix identifies the greatest identical coordinate shared by two
// replay-validated linear stores. A zero value means the stores share no
// commit; callers must decide whether an unjoined pair is admissible.
type CommitPrefix struct {
	Commit   artifact.CommitID `json:"commit"`
	Sequence uint64            `json:"sequence"`
}

// CommonCommitPrefix finds the greatest shared hash-chain coordinate without
// scanning either journal. Commit identity binds sequence and predecessor, so
// equality at an ordinal proves equality of the complete prefix ending there.
func CommonCommitPrefix(ctx context.Context, left, right *Store) (CommitPrefix, error) {
	if ctx == nil || left == nil || right == nil {
		return CommitPrefix{}, errors.New("overgodb: common prefix requires context and two stores")
	}
	if err := ctx.Err(); err != nil {
		return CommitPrefix{}, err
	}
	_, leftSequence := left.Head()
	_, rightSequence := right.Head()
	high := min(leftSequence, rightSequence)
	var result CommitPrefix
	for low := uint64(1); low <= high; {
		middle := low + (high-low)/2
		leftCommit, leftFound, err := left.CommitAt(ctx, middle)
		if err != nil {
			return CommitPrefix{}, err
		}
		rightCommit, rightFound, err := right.CommitAt(ctx, middle)
		if err != nil {
			return CommitPrefix{}, err
		}
		if !leftFound || !rightFound {
			return CommitPrefix{}, errors.New("overgodb: common prefix coordinate disappeared")
		}
		if leftCommit.ID == rightCommit.ID {
			result = CommitPrefix{Commit: leftCommit.ID, Sequence: middle}
			low = middle + 1
			continue
		}
		if middle == 0 {
			break
		}
		high = middle - 1
	}
	return result, nil
}
