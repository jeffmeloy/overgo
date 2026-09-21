package overgodb

import (
	"bytes"
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/fsatomic"
	"overgo/internal/strictjson"
)

// Per-projection checkpoints are derived acceleration state, never
// transaction authority: each file carries one projection's canonical
// body anchored to the journal head, sequence, log offset, anchor
// digest, and the projection's schema version, with a payload digest
// over the body. A corrupt, stale or unknown-version member discards
// the whole set and open falls back to the next verified anchor -- the
// monolithic snapshot, then full replay. Canonical form is proved where
// the set is written; a projection whose checkpoint shape changes
// advances its version, which TestCheckpointShapeIsVersioned enforces.

const (
	checkpointDirectory = "checkpoints"
	checkpointExtension = ".checkpoint"
)

type checkpointHeader struct {
	Name          string            `json:"name"`
	Version       uint16            `json:"version"`
	Sequence      uint64            `json:"sequence"`
	Head          artifact.CommitID `json:"head"`
	LogOffset     int64             `json:"log_offset"`
	LogAnchor     string            `json:"log_anchor"`
	PayloadDigest string            `json:"payload_digest"`
}

// checkpointGeneration names the directory of the set written at sequence;
// the width keeps the names in sequence order.
func checkpointGeneration(sequence uint64) string {
	return fmt.Sprintf("%020d", sequence)
}

// writeProjectionCheckpoints writes one checkpoint per registered
// projection into a generation of its own, each member staged and renamed.
// A write that is interrupted leaves a generation the loader refuses for a
// missing or torn member and the generation before it intact, so the open
// that follows replays only the journal since that one. Older generations,
// and the flat set a store held before generations, go once this one is
// complete.
func writeProjectionCheckpoints(
	root string,
	state *catalogState,
	sequence uint64,
	head artifact.CommitID,
	offset int64,
	anchor string,
) error {
	parent := filepath.Join(root, checkpointDirectory)
	directory := filepath.Join(parent, checkpointGeneration(sequence))
	if err := os.MkdirAll(directory, storeDirectoryMode); err != nil {
		return fmt.Errorf("overgodb: create checkpoint directory: %w", err)
	}
	for index, registered := range projections(state) {
		payload, err := registered.view.checkpoint()
		if err != nil {
			return fmt.Errorf("overgodb: checkpoint %s: %w", registered.name, err)
		}
		// An open trusts digest and version, so the body is held here, once,
		// to restoring into the bytes it was written from.
		proof := projections(new(catalogState))[index].view
		if err := proof.restore(payload); err != nil {
			return fmt.Errorf("overgodb: checkpoint %s does not restore: %w", registered.name, err)
		}
		if canonical, err := proof.checkpoint(); err != nil || !bytes.Equal(canonical, payload) {
			return fmt.Errorf("overgodb: checkpoint %s is non-canonical", registered.name)
		}
		digest := sha256.Sum256(payload)
		header, err := json.Marshal(checkpointHeader{
			Name: registered.name, Version: registered.version, Sequence: sequence, Head: head,
			LogOffset: offset, LogAnchor: anchor, PayloadDigest: hex.EncodeToString(digest[:]),
		})
		if err != nil {
			return fmt.Errorf("overgodb: checkpoint header %s: %w", registered.name, err)
		}
		document := append(header, '\n')
		document = append(document, payload...)
		destination := filepath.Join(directory, registered.name+checkpointExtension)
		staging, err := os.CreateTemp(directory, ".staging-*")
		if err != nil {
			return fmt.Errorf("overgodb: stage checkpoint %s: %w", registered.name, err)
		}
		stagingPath := staging.Name()
		writeErr := writeAll(staging, document)
		syncErr := staging.Sync()
		closeErr := staging.Close()
		if err := errors.Join(writeErr, syncErr, closeErr); err != nil {
			_ = os.Remove(stagingPath)
			return fmt.Errorf("overgodb: write checkpoint %s: %w", registered.name, err)
		}
		_ = os.Remove(destination)
		if err := os.Rename(stagingPath, destination); err != nil {
			_ = os.Remove(stagingPath)
			return fmt.Errorf("overgodb: publish checkpoint %s: %w", registered.name, err)
		}
	}
	if err := fsatomic.SyncDirectory(directory); err != nil {
		return err
	}
	entries, err := os.ReadDir(parent)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		// A generation a reader still holds open stays until the next write;
		// the loader takes the newest, so a leftover costs only its bytes.
		if entry.Name() != filepath.Base(directory) {
			_ = os.RemoveAll(filepath.Join(parent, entry.Name()))
		}
	}
	return fsatomic.SyncDirectory(parent)
}

// loadProjectionCheckpoints assembles catalog state from the newest
// generation that is complete and mutually consistent, falling back through
// older ones and last to the flat set a store held before generations. When
// none verifies it returns loaded=false with the newest one's reason, and
// the caller falls back to an earlier verified anchor.
func loadProjectionCheckpoints(root string) (catalogState, replayAnchor, bool, string) {
	parent := filepath.Join(root, checkpointDirectory)
	entries, _ := os.ReadDir(parent)
	generations := []string{""}
	for _, entry := range entries {
		if entry.IsDir() {
			generations = append(generations, entry.Name())
		}
	}
	slices.Sort(generations)
	var refused string
	for _, generation := range slices.Backward(generations) {
		directory := filepath.Join(parent, generation)
		state, anchor, loaded, reason := loadCheckpointSet(func(name string) ([]byte, error) {
			return os.ReadFile(filepath.Join(directory, name+checkpointExtension))
		})
		if loaded {
			return state, anchor, true, ""
		}
		refused = cmp.Or(refused, reason)
	}
	return catalogState{}, replayAnchor{}, false, refused
}

// loadCheckpointSet loads every member at once: each restores a facet of its
// own, and decoding them is nearly all an open costs. The members are judged
// in registration order, so the reason named never depends on which finished
// first.
func loadCheckpointSet(read func(name string) ([]byte, error)) (catalogState, replayAnchor, bool, string) {
	state := newCatalogState()
	members := projections(&state)
	anchors := make([]replayAnchor, len(members))
	defects := make([]string, len(members))
	var loading sync.WaitGroup
	for index, registered := range members {
		loading.Go(func() { anchors[index], defects[index] = loadCheckpoint(registered, read) })
	}
	loading.Wait()
	anchor := anchors[0]
	for index, registered := range members {
		if defects[index] == "" && anchors[index] != anchor {
			defects[index] = fmt.Sprintf("checkpoint %s anchors a different head", registered.name)
		}
		if defects[index] != "" {
			return catalogState{}, replayAnchor{}, false, defects[index]
		}
	}
	return state, anchor, true, ""
}

// loadCheckpoint verifies one member against its header and restores its
// facet.
func loadCheckpoint(registered registeredProjection, read func(name string) ([]byte, error)) (replayAnchor, string) {
	document, err := read(registered.name)
	if err != nil {
		return replayAnchor{}, fmt.Sprintf("checkpoint %s unreadable", registered.name)
	}
	line, payload, found := bytes.Cut(document, []byte{'\n'})
	if !found {
		return replayAnchor{}, fmt.Sprintf("checkpoint %s has no header", registered.name)
	}
	var header checkpointHeader
	if err := strictjson.DecodeBytes(line, &header); err != nil {
		return replayAnchor{}, fmt.Sprintf("checkpoint %s header: %v", registered.name, err)
	}
	if header.Name != registered.name || header.Version != registered.version {
		return replayAnchor{}, fmt.Sprintf("checkpoint %s names %s version %d; this reader speaks version %d",
			registered.name, header.Name, header.Version, registered.version)
	}
	digest := sha256.Sum256(payload)
	if hex.EncodeToString(digest[:]) != header.PayloadDigest {
		return replayAnchor{}, fmt.Sprintf("checkpoint %s payload digest differs", registered.name)
	}
	anchorDigest, err := hex.DecodeString(header.LogAnchor)
	if err != nil || len(anchorDigest) != sha256.Size {
		return replayAnchor{}, fmt.Sprintf("checkpoint %s anchor digest invalid", registered.name)
	}
	anchor := replayAnchor{sequence: header.Sequence, head: header.Head, offset: header.LogOffset}
	copy(anchor.digest[:], anchorDigest)
	if err := registered.view.restore(payload); err != nil {
		return replayAnchor{}, fmt.Sprintf("checkpoint %s restore: %v", registered.name, err)
	}
	return anchor, ""
}
