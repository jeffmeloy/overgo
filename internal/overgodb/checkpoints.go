package overgodb

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/fsatomic"
	"overgo/internal/strictjson"
)

// Per-projection checkpoints are derived acceleration state, never
// transaction authority: each file carries one projection's canonical
// body anchored to the journal head, sequence, log offset, anchor
// digest, and the projection's schema version, with a payload digest
// over the body. A corrupt, stale, unknown-version, or non-canonical
// member discards the whole set and open falls back to the next
// verified anchor -- the monolithic snapshot, then full replay.

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

// writeProjectionCheckpoints writes one checkpoint per registered
// projection, staged and renamed so a crash leaves either the previous
// set or the complete new one per file, never a torn member.
func writeProjectionCheckpoints(
	root string,
	state *catalogState,
	sequence uint64,
	head artifact.CommitID,
	offset int64,
	anchor string,
) error {
	directory := filepath.Join(root, checkpointDirectory)
	if err := os.MkdirAll(directory, storeDirectoryMode); err != nil {
		return fmt.Errorf("overgodb: create checkpoint directory: %w", err)
	}
	for _, registered := range projections(state) {
		payload, err := registered.view.checkpoint()
		if err != nil {
			return fmt.Errorf("overgodb: checkpoint %s: %w", registered.name, err)
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
	return fsatomic.SyncDirectory(directory)
}

// loadProjectionCheckpoints assembles catalog state from a complete,
// mutually consistent checkpoint set. Any defect returns loaded=false
// with the reason; the caller falls back to an earlier verified anchor.
func loadProjectionCheckpoints(root string) (catalogState, replayAnchor, bool, string) {
	directory := filepath.Join(root, checkpointDirectory)
	state := newCatalogState()
	var anchor replayAnchor
	anchored := false
	for _, registered := range projections(&state) {
		document, err := os.ReadFile(filepath.Join(directory, registered.name+checkpointExtension))
		if err != nil {
			return catalogState{}, replayAnchor{}, false, fmt.Sprintf("checkpoint %s unreadable", registered.name)
		}
		newline := -1
		for index, b := range document {
			if b == '\n' {
				newline = index
				break
			}
		}
		if newline < 0 {
			return catalogState{}, replayAnchor{}, false, fmt.Sprintf("checkpoint %s has no header", registered.name)
		}
		var header checkpointHeader
		if err := strictjson.DecodeBytes(document[:newline], &header); err != nil {
			return catalogState{}, replayAnchor{}, false, fmt.Sprintf("checkpoint %s header: %v", registered.name, err)
		}
		if header.Name != registered.name || header.Version != registered.version {
			return catalogState{}, replayAnchor{}, false,
				fmt.Sprintf("checkpoint %s names %s version %d; this reader speaks version %d",
					registered.name, header.Name, header.Version, registered.version)
		}
		payload := document[newline+1:]
		digest := sha256.Sum256(payload)
		if hex.EncodeToString(digest[:]) != header.PayloadDigest {
			return catalogState{}, replayAnchor{}, false, fmt.Sprintf("checkpoint %s payload digest differs", registered.name)
		}
		anchorDigest, err := hex.DecodeString(header.LogAnchor)
		if err != nil || len(anchorDigest) != sha256.Size {
			return catalogState{}, replayAnchor{}, false, fmt.Sprintf("checkpoint %s anchor digest invalid", registered.name)
		}
		memberAnchor := replayAnchor{sequence: header.Sequence, head: header.Head, offset: header.LogOffset}
		copy(memberAnchor.digest[:], anchorDigest)
		if !anchored {
			anchor, anchored = memberAnchor, true
		} else if memberAnchor != anchor {
			return catalogState{}, replayAnchor{}, false, fmt.Sprintf("checkpoint %s anchors a different head", registered.name)
		}
		if err := registered.view.restore(payload); err != nil {
			return catalogState{}, replayAnchor{}, false, fmt.Sprintf("checkpoint %s restore: %v", registered.name, err)
		}
		canonical, err := registered.view.checkpoint()
		if err != nil || string(canonical) != string(payload) {
			return catalogState{}, replayAnchor{}, false, fmt.Sprintf("checkpoint %s is non-canonical", registered.name)
		}
	}
	if !anchored {
		return catalogState{}, replayAnchor{}, false, "checkpoint set is empty"
	}
	return state, anchor, true, ""
}
