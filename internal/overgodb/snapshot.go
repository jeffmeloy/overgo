package overgodb

import (
	"context"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
)

// snapshotDirectory held the monolithic snapshot a store wrote before its
// checkpoint set was published as generations. Nothing reads it any more; a
// maintenance point removes what an older store left there.
const snapshotDirectory = "snapshots"

// SnapshotInfo names one maintenance point: the commit it anchors to and the
// checkpoint generation carrying the serialized state.
type SnapshotInfo struct {
	Sequence uint64
	Head     artifact.CommitID
	Path     string
}

// SnapshotReplay reports startup checkpoint selection.
type SnapshotReplay struct {
	Path     string
	Loaded   bool
	Fallback string
}

type snapshotAlias struct {
	Name   string      `json:"name"`
	Target artifact.ID `json:"target"`
}

// Snapshot is the store's maintenance point. On a blob-era store it is also
// the rotation boundary: the active journal seals first, so recovery from
// the fresh anchor replays only the tail written after this point. Inline
// legacy content seals with its frames and is read from the segment. The
// state is then written as one checkpoint generation, which is the store's
// only acceleration: an open that finds none that verifies replays the
// sealed segments.
func (s *Store) Snapshot(ctx context.Context) (SnapshotInfo, error) {
	var info SnapshotInfo
	err := s.writeTransaction(ctx, func() error {
		if rotateErr := s.sealLocked(); rotateErr != nil && !isSealRefusal(rotateErr) {
			return rotateErr
		}
		if s.sequence == 0 {
			return errors.New("overgodb: cannot snapshot empty store")
		}
		anchor, err := s.log.anchorDigest(s.replayEnd)
		if err != nil {
			return err
		}
		if err := writeProjectionCheckpoints(s.root, &s.state, s.sequence, s.head, s.replayEnd, hex.EncodeToString(anchor[:])); err != nil {
			return err
		}
		_ = os.RemoveAll(filepath.Join(s.root, snapshotDirectory))
		info = SnapshotInfo{Sequence: s.sequence, Head: s.head, Path: filepath.Join(s.root, checkpointDirectory, checkpointGeneration(s.sequence))}
		return nil
	})
	return info, err
}

// isSealRefusal reports the sanctioned reasons a snapshot proceeds
// without rotating: an empty journal, or a read-only handle asking for a
// snapshot (which ready refuses next).
func isSealRefusal(err error) bool {
	message := err.Error()
	return strings.Contains(message, "cannot seal an empty journal") ||
		strings.Contains(message, "nothing to seal") ||
		errors.Is(err, ErrReadOnly) || errors.Is(err, ErrClosed)
}
