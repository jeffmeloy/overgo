package repodb

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"os"
	"path/filepath"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
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
	sequence, head, root := s.sequence, s.head, s.root
	if sequence == 0 {
		s.mu.RUnlock()
		return SnapshotInfo{}, errors.New("repodb: cannot snapshot empty store")
	}
	path, err := writeSnapshot(ctx, root, sequence, head, s.state)
	s.mu.RUnlock()
	if err != nil {
		return SnapshotInfo{}, err
	}
	return SnapshotInfo{Sequence: sequence, Head: head, Path: path}, nil
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
		Contents: cloneValues(document.Contents), Manifests: cloneValues(document.Manifests),
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

type snapshotSink struct {
	ctx    context.Context
	writer io.Writer
	digest hash.Hash
	size   uint64
}

// Write streams bytes through size and digest accounting.
func (s *snapshotSink) Write(data []byte) (written int, err error) {
	if err = contextError(s.ctx); err != nil {
		return written, err
	}
	if uint64(len(data)) > uint64(maxSnapshotPayload)-s.size {
		return written, errors.New("repodb: snapshot exceeds payload limit")
	}
	if err = writeAll(s.writer, data); err != nil {
		return written, err
	}
	_, _ = s.digest.Write(data)
	s.size += uint64(len(data))
	return len(data), nil
}

type snapshotStream struct {
	writer io.Writer
	err    error
}

// Write preserves the stream's first failure.
func (s *snapshotStream) Write(data []byte) (written int, err error) {
	if s.err != nil {
		return written, s.err
	}
	if s.err = writeAll(s.writer, data); s.err != nil {
		return written, s.err
	}
	return len(data), nil
}

func (s *snapshotStream) raw(value string) {
	if s.err == nil {
		s.err = writeAll(s.writer, []byte(value))
	}
}

func (s *snapshotStream) value(value any) {
	if s.err != nil {
		return
	}
	data, err := json.Marshal(value)
	if err != nil {
		s.err = err
		return
	}
	s.err = writeAll(s.writer, data)
}

func (s *snapshotStream) array(name string, count int, omitEmpty bool, emit func(int)) {
	if omitEmpty && count == 0 {
		return
	}
	s.raw(`,"` + name + `":[`)
	for index := 0; index < count; index++ {
		if index > 0 {
			s.raw(",")
		}
		emit(index)
	}
	s.raw("]")
}

func writeSnapshotPayload(writer io.Writer, state catalogState, sequence uint64, head artifact.CommitID) error {
	stream := snapshotStream{writer: writer}
	stream.raw(`{"version":`)
	stream.value(snapshotVersion)
	stream.raw(`,"sequence":`)
	stream.value(sequence)
	stream.raw(`,"head":`)
	stream.value(head)

	idLess := func(left, right artifact.ID) bool { return left.String() < right.String() }
	artifactIDs := sortedSnapshotKeys(state.artifacts, idLess)
	stream.array("artifacts", len(artifactIDs), false, func(index int) {
		stream.value(state.artifacts[artifactIDs[index]])
	})
	contentIDs := sortedSnapshotKeys(state.contents, idLess)
	stream.array("contents", len(contentIDs), true, func(index int) {
		id := contentIDs[index]
		stream.raw(`{"descriptor":`)
		stream.value(state.artifacts[id])
		stream.raw(`,"data":"`)
		encoder := base64.NewEncoder(base64.StdEncoding, &stream)
		if _, err := encoder.Write(state.contents[id]); err != nil && stream.err == nil {
			stream.err = err
		}
		if err := encoder.Close(); err != nil && stream.err == nil {
			stream.err = err
		}
		stream.raw(`"}`)
	})
	manifestIDs := sortedSnapshotKeys(state.manifests, idLess)
	stream.array("manifests", len(manifestIDs), true, func(index int) {
		stream.value(state.manifests[manifestIDs[index]])
	})
	aliases := sortedSnapshotKeys(state.aliases, func(left, right string) bool { return left < right })
	stream.array("aliases", len(aliases), true, func(index int) {
		name := aliases[index]
		stream.value(snapshotAlias{Name: name, Target: state.aliases[name]})
	})
	lineage := sortedSnapshotKeys(state.lineage, func(left, right relationKey) bool {
		if left.child != right.child {
			return left.child.String() < right.child.String()
		}
		if left.parent != right.parent {
			return left.parent.String() < right.parent.String()
		}
		return left.relation < right.relation
	})
	stream.array("lineage", len(lineage), true, func(index int) {
		stream.value(state.lineage[lineage[index]])
	})
	locations := sortedSnapshotKeys(state.locations, func(left, right locationKey) bool {
		if left.artifact != right.artifact {
			return left.artifact.String() < right.artifact.String()
		}
		if left.kind != right.kind {
			return left.kind < right.kind
		}
		return left.value < right.value
	})
	stream.array("locations", len(locations), true, func(index int) {
		stream.value(state.locations[locations[index]])
	})
	commits := sortedSnapshotKeys(state.commits, func(left, right string) bool {
		return state.commits[left].sequence < state.commits[right].sequence
	})
	stream.array("commits", len(commits), false, func(index int) {
		key := commits[index]
		commit := state.commits[key]
		stream.value(snapshotCommit{
			Key: key, ID: commit.id, Payload: hex.EncodeToString(commit.payload[:]), Sequence: commit.sequence,
		})
	})
	stream.raw("}")
	return stream.err
}

func sortedSnapshotKeys[K comparable, V any](values map[K]V, less func(K, K) bool) []K {
	keys := make([]K, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(left, right int) bool { return less(keys[left], keys[right]) })
	return keys
}

func writeSnapshot(ctx context.Context, root string, sequence uint64, head artifact.CommitID, state catalogState) (string, error) {
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
	temporary, err := os.CreateTemp(directory, ".repodb-snapshot-*.tmp")
	if err != nil {
		return "", fmt.Errorf("repodb: create snapshot: %w", err)
	}
	temporaryPath := temporary.Name()
	defer os.Remove(temporaryPath)
	if err = temporary.Chmod(storeFileMode); err == nil {
		err = writeAll(temporary, make([]byte, snapshotHeaderBytes))
	}
	if err == nil {
		digest := sha256.New()
		sink := snapshotSink{ctx: ctx, writer: temporary, digest: digest}
		err = writeSnapshotPayload(&sink, state, sequence, head)
		if err == nil {
			var sum [sha256.Size]byte
			copy(sum[:], digest.Sum(nil))
			if _, err = temporary.Seek(snapshotMagicOffset, io.SeekStart); err == nil {
				err = writeAll(temporary, encodeSnapshotHeader(sequence, head, sink.size, sum))
			}
		}
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

func encodeSnapshotHeader(sequence uint64, head artifact.CommitID, size uint64, digest [sha256.Size]byte) []byte {
	header := make([]byte, snapshotHeaderBytes)
	copy(header[snapshotMagicOffset:snapshotVersionOffset], snapshotMagic[:])
	binary.LittleEndian.PutUint16(header[snapshotVersionOffset:snapshotFlagsOffset], snapshotVersion)
	binary.LittleEndian.PutUint16(header[snapshotFlagsOffset:snapshotSequenceOffset], 0)
	binary.LittleEndian.PutUint64(header[snapshotSequenceOffset:snapshotHeadOffset], sequence)
	copy(header[snapshotHeadOffset:snapshotSizeOffset], head[:])
	binary.LittleEndian.PutUint64(header[snapshotSizeOffset:snapshotDigestOffset], size)
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
	limited := &io.LimitedReader{R: file, N: int64(size)}
	hasher := sha256.New()
	var document snapshotDocument
	if err := strictjson.Decode(io.TeeReader(limited, hasher), &document); err != nil {
		return catalogState{}, replayAnchor{}, err
	}
	var actual [sha256.Size]byte
	copy(actual[:], hasher.Sum(nil))
	if limited.N != 0 || actual != digest {
		return catalogState{}, replayAnchor{}, errors.New("repodb: snapshot digest differs")
	}
	if document.Sequence != sequence || document.Head != head {
		return catalogState{}, replayAnchor{}, errors.New("repodb: snapshot header differs from payload")
	}
	state, err := stateFromSnapshot(document)
	if err != nil {
		return catalogState{}, replayAnchor{}, err
	}
	canonicalHash := sha256.New()
	sink := snapshotSink{ctx: context.Background(), writer: io.Discard, digest: canonicalHash}
	if err := writeSnapshotPayload(&sink, state, sequence, head); err != nil {
		return catalogState{}, replayAnchor{}, err
	}
	var canonical [sha256.Size]byte
	copy(canonical[:], canonicalHash.Sum(nil))
	if sink.size != size || canonical != digest {
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
