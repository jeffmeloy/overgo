package overgodb

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

// Per-facet canonical checkpoint bodies. Every serialization is sorted
// and strict-decoded, so serializing a restored checkpoint reproduces
// it byte for byte; the loader's fixed-point check depends on that.

type artifactCheckpointEntry struct {
	Descriptor artifact.Descriptor `json:"descriptor"`
	Sequence   uint64              `json:"sequence"`
	Manifest   *artifact.Manifest  `json:"manifest,omitempty"`
}

func (f *artifactFacet) checkpoint() ([]byte, error) {
	entries := make([]artifactCheckpointEntry, 0, len(f.bySequence))
	for _, id := range f.bySequence {
		record := f.records[id]
		entry := artifactCheckpointEntry{Descriptor: record.descriptor, Sequence: record.sequence}
		if record.hasManifest {
			manifest := record.manifest.Clone()
			entry.Manifest = &manifest
		}
		entries = append(entries, entry)
	}
	return json.Marshal(entries)
}

func (f *artifactFacet) restore(data []byte) error {
	var entries []artifactCheckpointEntry
	if err := strictjson.DecodeBytes(data, &entries); err != nil {
		return fmt.Errorf("artifacts checkpoint: %w", err)
	}
	*f = newArtifactFacet()
	for _, entry := range entries {
		f.add(entry.Descriptor, entry.Sequence)
		if entry.Manifest != nil {
			f.setManifest(*entry.Manifest)
		}
	}
	return nil
}

type contentCheckpointEntry struct {
	Artifact artifact.ID `json:"artifact"`
	Offset   int64       `json:"offset"`
	Size     int64       `json:"size"`
	Sequence uint64      `json:"sequence"`
	Blob     bool        `json:"blob,omitempty"`
}

func (f *contentFacet) checkpoint() ([]byte, error) {
	entries := make([]contentCheckpointEntry, 0, len(f.locators))
	for id, locator := range f.locators {
		entries = append(entries, contentCheckpointEntry{
			Artifact: id, Offset: locator.offset, Size: locator.size, Sequence: locator.sequence, Blob: locator.blob,
		})
	}
	sort.Slice(entries, func(i, j int) bool {
		return artifact.CompareID(entries[i].Artifact, entries[j].Artifact) < 0
	})
	return json.Marshal(entries)
}

func (f *contentFacet) restore(data []byte) error {
	var entries []contentCheckpointEntry
	if err := strictjson.DecodeBytes(data, &entries); err != nil {
		return fmt.Errorf("contents checkpoint: %w", err)
	}
	*f = newContentFacet()
	for _, entry := range entries {
		if entry.Sequence == 0 {
			return errors.New("contents checkpoint: invalid introduction sequence")
		}
		f.set(entry.Artifact, contentLocator{offset: entry.Offset, size: entry.Size, blob: entry.Blob}, entry.Sequence)
	}
	return nil
}

func (f *lineageFacet) checkpoint() ([]byte, error) {
	edges := make([]artifact.Lineage, 0, f.edges)
	children := make([]artifact.ID, 0, len(f.parents))
	for child := range f.parents {
		children = append(children, child)
	}
	slices.SortFunc(children, artifact.CompareID)
	for _, child := range children {
		for _, key := range f.parents[child] {
			edges = append(edges, artifact.Lineage{Child: key.child, Parent: key.parent, Relation: key.relation})
		}
	}
	return json.Marshal(edges)
}

func (f *lineageFacet) restore(data []byte) error {
	var edges []artifact.Lineage
	if err := strictjson.DecodeBytes(data, &edges); err != nil {
		return fmt.Errorf("lineage checkpoint: %w", err)
	}
	*f = newLineageFacet()
	for _, edge := range edges {
		f.add(relationKey{child: edge.Child, parent: edge.Parent, relation: edge.Relation})
	}
	return nil
}

func (f *locationFacet) checkpoint() ([]byte, error) {
	locations := make([]artifact.Location, 0, len(f.byArtifact))
	ids := make([]artifact.ID, 0, len(f.byArtifact))
	for id := range f.byArtifact {
		ids = append(ids, id)
	}
	slices.SortFunc(ids, artifact.CompareID)
	for _, id := range ids {
		locations = append(locations, f.byArtifact[id]...)
	}
	return json.Marshal(locations)
}

func (f *locationFacet) restore(data []byte) error {
	var locations []artifact.Location
	if err := strictjson.DecodeBytes(data, &locations); err != nil {
		return fmt.Errorf("locations checkpoint: %w", err)
	}
	*f = newLocationFacet()
	for _, location := range locations {
		f.apply(artifact.LocationEvent{Location: location, Action: artifact.LocationAdd})
	}
	return nil
}

func (f *aliasFacet) checkpoint() ([]byte, error) {
	names := make([]string, 0, len(f.bindings))
	for name := range f.bindings {
		names = append(names, name)
	}
	sort.Strings(names)
	entries := make([]snapshotAlias, 0, len(names))
	for _, name := range names {
		entries = append(entries, snapshotAlias{Name: name, Target: f.bindings[name]})
	}
	return json.Marshal(entries)
}

func (f *aliasFacet) restore(data []byte) error {
	var entries []snapshotAlias
	if err := strictjson.DecodeBytes(data, &entries); err != nil {
		return fmt.Errorf("aliases checkpoint: %w", err)
	}
	*f = newAliasFacet()
	for _, entry := range entries {
		f.apply(artifact.AliasBinding{Name: entry.Name, Target: entry.Target})
	}
	return nil
}

type commitCheckpointEntry struct {
	Key      string            `json:"key"`
	ID       artifact.CommitID `json:"id"`
	Payload  string            `json:"payload"`
	Sequence uint64            `json:"sequence"`
}

func (f *commitFacet) checkpoint() ([]byte, error) {
	entries := make([]commitCheckpointEntry, 0, len(f.ordered))
	for _, commit := range f.ordered {
		entries = append(entries, commitCheckpointEntry{
			Key: commit.key, ID: commit.id, Payload: hex.EncodeToString(commit.payload[:]), Sequence: commit.sequence,
		})
	}
	return json.Marshal(entries)
}

func (f *commitFacet) restore(data []byte) error {
	var entries []commitCheckpointEntry
	if err := strictjson.DecodeBytes(data, &entries); err != nil {
		return fmt.Errorf("commits checkpoint: %w", err)
	}
	*f = newCommitFacet()
	for _, entry := range entries {
		decoded, err := hex.DecodeString(entry.Payload)
		if err != nil || len(decoded) != sha256.Size {
			return fmt.Errorf("commits checkpoint payload digest: %w", err)
		}
		var payload [sha256.Size]byte
		copy(payload[:], decoded)
		f.add(committedBatch{key: entry.Key, id: entry.ID, payload: payload, sequence: entry.Sequence})
	}
	return nil
}
