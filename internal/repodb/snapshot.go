package repodb

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"sort"

	"overgo/internal/artifact"
)

const (
	snapshotDirectory         = "snapshots"
	snapshotExtension         = ".snapshot"
	snapshotHeaderBytes       = 92
	snapshotVersion           = uint16(1)
	snapshotPayloadMultiplier = 4
	maxSnapshotPayload        = maxFramePayload * snapshotPayloadMultiplier

	snapshotMagicOffset    = 0
	snapshotVersionOffset  = 8
	snapshotFlagsOffset    = 10
	snapshotSequenceOffset = 12
	snapshotHeadOffset     = 20
	snapshotSizeOffset     = 52
	snapshotDigestOffset   = 60
)

var snapshotMagic = [8]byte{'L', '2', 'G', 'S', 'N', 'A', 'P', '1'}

// SnapshotInfo: immutable catalog checkpoint.
type SnapshotInfo struct {
	Sequence uint64
	Head     artifact.CommitID
	Path     string
}

type snapshotAlias struct {
	Name   string      `json:"name"`
	Target artifact.ID `json:"target"`
}

type snapshotCommit struct {
	Key      string            `json:"key"`
	ID       artifact.CommitID `json:"id"`
	Payload  string            `json:"payload"`
	Sequence uint64            `json:"sequence"`
}

type snapshotDocument struct {
	Version   uint16                `json:"version"`
	Sequence  uint64                `json:"sequence"`
	Head      artifact.CommitID     `json:"head"`
	Artifacts []artifact.Descriptor `json:"artifacts"`
	Contents  []artifact.Content    `json:"contents,omitempty"`
	Manifests []artifact.Manifest   `json:"manifests,omitempty"`
	Aliases   []snapshotAlias       `json:"aliases,omitempty"`
	Lineage   []artifact.Lineage    `json:"lineage,omitempty"`
	Locations []artifact.Location   `json:"locations,omitempty"`
	Commits   []snapshotCommit      `json:"commits"`
}

// Snapshot writes one immutable, versioned state segment.
func (s *Store) Snapshot(ctx context.Context) (SnapshotInfo, error) {
	if err := contextError(ctx); err != nil {
		return SnapshotInfo{}, err
	}
	s.mu.RLock()
	if err := s.ready(true); err != nil {
		s.mu.RUnlock()
		return SnapshotInfo{}, err
	}
	document := snapshotDocumentFromState(s.state, s.sequence, s.head)
	root := s.root
	s.mu.RUnlock()
	if document.Sequence == 0 {
		return SnapshotInfo{}, errors.New("repodb: cannot snapshot empty store")
	}
	payload, err := json.Marshal(document)
	if err != nil {
		return SnapshotInfo{}, fmt.Errorf("repodb: encode snapshot: %w", err)
	}
	if len(payload) > maxSnapshotPayload {
		return SnapshotInfo{}, errors.New("repodb: snapshot exceeds payload limit")
	}
	if err := contextError(ctx); err != nil {
		return SnapshotInfo{}, err
	}
	path, err := writeSnapshot(root, document.Sequence, document.Head, payload)
	if err != nil {
		return SnapshotInfo{}, err
	}
	return SnapshotInfo{Sequence: document.Sequence, Head: document.Head, Path: path}, nil
}

func snapshotDocumentFromState(state catalogState, sequence uint64, head artifact.CommitID) snapshotDocument {
	document := snapshotDocument{Version: snapshotVersion, Sequence: sequence, Head: head}
	for _, descriptor := range state.artifacts {
		document.Artifacts = append(document.Artifacts, descriptor)
	}
	for _, content := range state.contents {
		document.Contents = append(document.Contents, content.Clone())
	}
	for _, manifest := range state.manifests {
		document.Manifests = append(document.Manifests, manifest.Clone())
	}
	for name, target := range state.aliases {
		document.Aliases = append(document.Aliases, snapshotAlias{Name: name, Target: target})
	}
	for _, edge := range state.lineage {
		document.Lineage = append(document.Lineage, edge)
	}
	for _, location := range state.locations {
		document.Locations = append(document.Locations, location)
	}
	for key, commit := range state.commits {
		document.Commits = append(document.Commits, snapshotCommit{
			Key: key, ID: commit.id, Payload: hex.EncodeToString(commit.payload[:]), Sequence: commit.sequence,
		})
	}
	sortSnapshotDocument(&document)
	return document
}

func sortSnapshotDocument(document *snapshotDocument) {
	sort.Slice(document.Artifacts, func(i, j int) bool {
		return document.Artifacts[i].ID.String() < document.Artifacts[j].ID.String()
	})
	sort.Slice(document.Contents, func(i, j int) bool {
		return document.Contents[i].Descriptor.ID.String() < document.Contents[j].Descriptor.ID.String()
	})
	sort.Slice(document.Manifests, func(i, j int) bool {
		return document.Manifests[i].ID.String() < document.Manifests[j].ID.String()
	})
	sort.Slice(document.Aliases, func(i, j int) bool { return document.Aliases[i].Name < document.Aliases[j].Name })
	sort.Slice(document.Lineage, func(i, j int) bool {
		left, right := document.Lineage[i], document.Lineage[j]
		if left.Child != right.Child {
			return left.Child.String() < right.Child.String()
		}
		if left.Parent != right.Parent {
			return left.Parent.String() < right.Parent.String()
		}
		return left.Relation < right.Relation
	})
	sort.Slice(document.Locations, func(i, j int) bool {
		left, right := document.Locations[i], document.Locations[j]
		if left.Artifact != right.Artifact {
			return left.Artifact.String() < right.Artifact.String()
		}
		if left.Kind != right.Kind {
			return left.Kind < right.Kind
		}
		return left.Value < right.Value
	})
	sort.Slice(document.Commits, func(i, j int) bool { return document.Commits[i].Sequence < document.Commits[j].Sequence })
}

func stateFromSnapshot(document snapshotDocument) (catalogState, error) {
	if document.Version != snapshotVersion || document.Sequence == 0 || !document.Head.Valid() {
		return catalogState{}, errors.New("repodb: invalid snapshot identity")
	}
	if uint64(len(document.Commits)) != document.Sequence {
		return catalogState{}, errors.New("repodb: snapshot commit count differs")
	}
	batch := artifact.Batch{
		Key: "snapshot/state", Artifacts: slices.Clone(document.Artifacts),
		Contents: cloneContents(document.Contents), Manifests: cloneManifests(document.Manifests),
		Lineage: slices.Clone(document.Lineage),
	}
	for _, alias := range document.Aliases {
		batch.Aliases = append(batch.Aliases, artifact.AliasBinding{Name: alias.Name, Target: alias.Target})
	}
	for _, location := range document.Locations {
		batch.Locations = append(batch.Locations, artifact.LocationEvent{
			Location: location, Action: artifact.LocationAdd,
		})
	}
	normalized, err := normalizeBatch(batch)
	if err != nil {
		return catalogState{}, err
	}
	state := newCatalogState()
	if err := state.validate(normalized); err != nil {
		return catalogState{}, err
	}
	state.apply(normalized)
	for index, entry := range document.Commits {
		if entry.Sequence != uint64(index)+1 {
			return catalogState{}, errors.New("repodb: invalid snapshot commit sequence")
		}
		if _, duplicate := state.commits[entry.Key]; duplicate || !entry.ID.Valid() {
			return catalogState{}, errors.New("repodb: invalid snapshot commit")
		}
		if err := (artifact.Batch{
			Key: entry.Key, Artifacts: []artifact.Descriptor{normalized.Artifacts[0]},
		}).Validate(); err != nil {
			return catalogState{}, errors.New("repodb: invalid snapshot commit key")
		}
		decoded, err := hex.DecodeString(entry.Payload)
		if err != nil || len(decoded) != sha256.Size {
			return catalogState{}, errors.New("repodb: invalid snapshot payload digest")
		}
		var payload [sha256.Size]byte
		copy(payload[:], decoded)
		state.commits[entry.Key] = committedBatch{id: entry.ID, payload: payload, sequence: entry.Sequence}
	}
	if document.Commits[len(document.Commits)-1].ID != document.Head {
		return catalogState{}, errors.New("repodb: snapshot head is not a commit")
	}
	return state, nil
}

func writeSnapshot(root string, sequence uint64, head artifact.CommitID, payload []byte) (string, error) {
	directory := filepath.Join(root, snapshotDirectory)
	if err := os.MkdirAll(directory, storeDirectoryMode); err != nil {
		return "", fmt.Errorf("repodb: create snapshot directory: %w", err)
	}
	name := fmt.Sprintf("%020d-%s%s", sequence, head.String(), snapshotExtension)
	path := filepath.Join(directory, name)
	if _, err := os.Stat(path); err == nil {
		return path, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", fmt.Errorf("repodb: inspect snapshot: %w", err)
	}
	header := encodeSnapshotHeader(sequence, head, payload)
	temporary, err := os.CreateTemp(directory, ".repodb-snapshot-*.tmp")
	if err != nil {
		return "", fmt.Errorf("repodb: create snapshot: %w", err)
	}
	temporaryPath := temporary.Name()
	defer os.Remove(temporaryPath)
	if err = temporary.Chmod(storeFileMode); err == nil {
		err = writeAll(temporary, header)
	}
	if err == nil {
		err = writeAll(temporary, payload)
	}
	if err == nil {
		err = temporary.Sync()
	}
	closeErr := temporary.Close()
	if err == nil {
		err = closeErr
	}
	if err != nil {
		return "", fmt.Errorf("repodb: write snapshot: %w", err)
	}
	if err := os.Link(temporaryPath, path); err != nil {
		if _, statErr := os.Stat(path); statErr == nil {
			return path, nil
		}
		return "", fmt.Errorf("repodb: publish snapshot: %w", err)
	}
	return path, nil
}

func encodeSnapshotHeader(sequence uint64, head artifact.CommitID, payload []byte) []byte {
	header := make([]byte, snapshotHeaderBytes)
	copy(header[snapshotMagicOffset:snapshotVersionOffset], snapshotMagic[:])
	binary.LittleEndian.PutUint16(header[snapshotVersionOffset:snapshotFlagsOffset], snapshotVersion)
	binary.LittleEndian.PutUint16(header[snapshotFlagsOffset:snapshotSequenceOffset], 0)
	binary.LittleEndian.PutUint64(header[snapshotSequenceOffset:snapshotHeadOffset], sequence)
	copy(header[snapshotHeadOffset:snapshotSizeOffset], head[:])
	binary.LittleEndian.PutUint64(header[snapshotSizeOffset:snapshotDigestOffset], uint64(len(payload)))
	digest := sha256.Sum256(payload)
	copy(header[snapshotDigestOffset:snapshotHeaderBytes], digest[:])
	return header
}

func loadLatestSnapshot(root string) (catalogState, replayAnchor, bool) {
	paths, err := filepath.Glob(filepath.Join(root, snapshotDirectory, "*"+snapshotExtension))
	if err != nil {
		return catalogState{}, replayAnchor{}, false
	}
	slices.Sort(paths)
	slices.Reverse(paths)
	for _, path := range paths {
		state, anchor, err := readSnapshot(path)
		if err == nil {
			return state, anchor, true
		}
	}
	return catalogState{}, replayAnchor{}, false
}

func readSnapshot(path string) (catalogState, replayAnchor, error) {
	file, err := os.Open(path)
	if err != nil {
		return catalogState{}, replayAnchor{}, err
	}
	defer file.Close()
	header := make([]byte, snapshotHeaderBytes)
	if _, err := io.ReadFull(file, header); err != nil {
		return catalogState{}, replayAnchor{}, err
	}
	sequence, head, size, digest, err := decodeSnapshotHeader(header)
	if err != nil {
		return catalogState{}, replayAnchor{}, err
	}
	info, err := file.Stat()
	if err != nil || uint64(info.Size()) != uint64(snapshotHeaderBytes)+size {
		return catalogState{}, replayAnchor{}, errors.New("repodb: snapshot size differs")
	}
	payload := make([]byte, int(size))
	if _, err := io.ReadFull(file, payload); err != nil {
		return catalogState{}, replayAnchor{}, err
	}
	if sha256.Sum256(payload) != digest {
		return catalogState{}, replayAnchor{}, errors.New("repodb: snapshot digest differs")
	}
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	var document snapshotDocument
	if err := decoder.Decode(&document); err != nil {
		return catalogState{}, replayAnchor{}, err
	}
	if err := ensureJSONEnd(decoder); err != nil || document.Sequence != sequence || document.Head != head {
		return catalogState{}, replayAnchor{}, errors.New("repodb: snapshot header differs from payload")
	}
	state, err := stateFromSnapshot(document)
	if err != nil {
		return catalogState{}, replayAnchor{}, err
	}
	canonical := snapshotDocumentFromState(state, sequence, head)
	canonicalPayload, err := json.Marshal(canonical)
	if err != nil || !bytes.Equal(payload, canonicalPayload) {
		return catalogState{}, replayAnchor{}, errors.New("repodb: non-canonical snapshot")
	}
	return state, replayAnchor{sequence: sequence, head: head}, nil
}

func decodeSnapshotHeader(header []byte) (uint64, artifact.CommitID, uint64, [sha256.Size]byte, error) {
	if len(header) != snapshotHeaderBytes ||
		!bytes.Equal(header[snapshotMagicOffset:snapshotVersionOffset], snapshotMagic[:]) ||
		binary.LittleEndian.Uint16(header[snapshotVersionOffset:snapshotFlagsOffset]) != snapshotVersion ||
		binary.LittleEndian.Uint16(header[snapshotFlagsOffset:snapshotSequenceOffset]) != 0 {
		return 0, artifact.CommitID{}, 0, [sha256.Size]byte{}, errors.New("repodb: invalid snapshot header")
	}
	sequence := binary.LittleEndian.Uint64(header[snapshotSequenceOffset:snapshotHeadOffset])
	var head artifact.CommitID
	copy(head[:], header[snapshotHeadOffset:snapshotSizeOffset])
	size := binary.LittleEndian.Uint64(header[snapshotSizeOffset:snapshotDigestOffset])
	if sequence == 0 || !head.Valid() || size > maxSnapshotPayload {
		return 0, artifact.CommitID{}, 0, [sha256.Size]byte{}, errors.New("repodb: invalid snapshot bounds")
	}
	var digest [sha256.Size]byte
	copy(digest[:], header[snapshotDigestOffset:snapshotHeaderBytes])
	return sequence, head, size, digest, nil
}
