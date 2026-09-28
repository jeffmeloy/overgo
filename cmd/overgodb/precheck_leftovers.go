package main

import (
	"cmp"
	"os"
	"path/filepath"
	"slices"
	"time"

	"overgo/internal/overgodb"
)

// leftover is one directory a store operation left behind: a store a swap
// superseded, or a sealed backup. A clone shares its blobs with the backup it
// came from by hard link, and a hard link is one of several equal names for
// the same bytes, so removing any leftover removes none of the blobs the store
// in service names. What a leftover is kept for is going back, which is why
// the newest of each kind is marked and the rest are not.
type leftover struct {
	Path     string    `json:"path"`
	Kind     string    `json:"kind"`
	Sequence uint64    `json:"sequence,omitzero"`
	Modified time.Time `json:"modified"`
	Packed   bool      `json:"packed"`
	Keep     bool      `json:"keep"`
}

const (
	leftoverSuperseded = "superseded-store"
	leftoverBackup     = "sealed-backup"
)

// listLeftovers reports the superseded stores beside the store in service and
// the sealed backups under the backup directory, newest first within each
// kind, marking the newest of each to keep. It removes nothing.
func listLeftovers(repository, backups string) ([]leftover, error) {
	superseded, err := filepath.Glob(filepath.Clean(repository) + ".superseded-*")
	if err != nil {
		return nil, err
	}
	sealed, err := filepath.Glob(filepath.Join(backups, "*"))
	if err != nil {
		return nil, err
	}
	var found []leftover
	for _, path := range slices.Concat(superseded, sealed) {
		info, err := os.Stat(path)
		if err != nil || !info.IsDir() {
			continue
		}
		entry := leftover{Path: path, Kind: leftoverSuperseded, Modified: info.ModTime()}
		if filepath.Dir(path) == filepath.Clean(backups) {
			seal, err := overgodb.ReadBackupSeal(path)
			if err != nil {
				continue
			}
			entry.Kind, entry.Sequence = leftoverBackup, seal.Sequence
		}
		_, packErr := os.Stat(filepath.Join(path, "blobs", "small.pack"))
		entry.Packed = packErr == nil
		found = append(found, entry)
	}
	slices.SortFunc(found, func(left, right leftover) int {
		return cmp.Or(cmp.Compare(left.Kind, right.Kind), cmp.Compare(right.Sequence, left.Sequence), right.Modified.Compare(left.Modified))
	})
	seen := map[string]bool{}
	for index := range found {
		found[index].Keep, seen[found[index].Kind] = !seen[found[index].Kind], true
	}
	return found, nil
}
