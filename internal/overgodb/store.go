// Package overgodb is the durable artifact catalog: a hash-chained
// record log replayed into content, manifest, alias, and lineage state,
// queried through bounded projections and compacted by retention.
package overgodb

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/fsatomic"
	"overgo/internal/strictjson"
)

var (
	// ErrClosed reports an operation against a store already closed.
	ErrClosed = errors.New("overgodb: store is closed")
	// ErrReadOnly reports a mutation attempted through a read-only handle.
	ErrReadOnly = errors.New("overgodb: store is read-only")
	// ErrBatchKeyConflict reports a batch key that already names different content.
	ErrBatchKeyConflict = errors.New("overgodb: batch key already names different content")
	// ErrArtifactConflict reports an artifact identity carrying conflicting facts.
	ErrArtifactConflict = errors.New("overgodb: artifact identity has conflicting facts")
	// ErrAliasConflict reports an alias compare-and-set whose expected target no longer holds.
	ErrAliasConflict = errors.New("overgodb: alias compare-and-set failed")
	// ErrHeadConflict reports a commit precondition naming a head the store has moved past.
	ErrHeadConflict = artifact.ErrCommitPrecondition
	// ErrLineageCycle reports a lineage edge that would close a cycle.
	ErrLineageCycle = errors.New("overgodb: lineage cycle")
	// ErrStoreFaulted reports a store that must be reopened after an uncertain append.
	ErrStoreFaulted = errors.New("overgodb: store requires reopen after uncertain append")
	// ErrSnapshotAnchor reports a snapshot whose anchor commit is absent from the chain.
	ErrSnapshotAnchor = errors.New("overgodb: snapshot anchor is absent from commit chain")
	errPayloadLimit   = errors.New("overgodb: payload limit exceeded")
	// ErrNoChange reports a batch that leaves catalog state unchanged.
	ErrNoChange = artifact.ErrNoChange
)

type relationKey struct {
	child    artifact.ID
	parent   artifact.ID
	relation artifact.Relation
}

type committedBatch struct {
	key        string
	id         artifact.CommitID
	payload    [sha256.Size]byte
	sequence   uint64
	coordinate commitCoordinate
}

type persistedTransaction struct {
	Request [sha256.Size]byte     `json:"request"`
	Delta   artifact.Batch        `json:"delta"`
	Content []artifact.Descriptor `json:"content,omitempty"`
	// BlobContent lists descriptors whose bytes were made durable in
	// the blob store before this frame was appended; the envelope
	// carries intent and identity, never the bytes.
	BlobContent []artifact.Descriptor `json:"blob_content,omitempty"`
}

type contentLocator struct {
	offset   int64
	size     int64
	sequence uint64
	// blob marks content whose bytes live in the content-addressed
	// blob store rather than inline in the journal frame.
	blob bool
}

// catalogState is the aggregate transaction view over the six
// concrete facet owners; commit order drives every facet exactly once.
type catalogState struct {
	artifacts artifactFacet
	contents  contentFacet
	lineage   lineageFacet
	causality causalityFacet
	locations locationFacet
	aliases   aliasFacet
	commits   commitFacet
}

func newCatalogState() catalogState {
	return catalogState{
		artifacts: newArtifactFacet(), contents: newContentFacet(), lineage: newLineageFacet(),
		causality: newCausalityFacet(), locations: newLocationFacet(), aliases: newAliasFacet(), commits: newCommitFacet(),
	}
}

func (s catalogState) validate(batch artifact.Batch) error {
	added, err := s.artifacts.validate(batch)
	if err != nil {
		return err
	}
	hasArtifact := func(id artifact.ID) bool { return s.hasArtifact(id, added) }
	if err := s.aliases.validate(batch, hasArtifact); err != nil {
		return err
	}
	if err := s.lineage.validate(batch, hasArtifact); err != nil {
		return err
	}
	if err := s.causality.validate(batch, hasArtifact); err != nil {
		return err
	}
	return s.locations.validate(batch, hasArtifact)
}

func (s catalogState) hasArtifact(id artifact.ID, added map[artifact.ID]struct{}) bool {
	if s.artifacts.has(id) {
		return true
	}
	_, ok := added[id]
	return ok
}

// acceptAll asks every registered projection to accept the commit;
// the first refusal wins and nothing has been published.
func (s *catalogState) acceptAll(batch artifact.Batch, locators map[artifact.ID]contentLocator, sequence uint64) error {
	for _, registered := range projections(s) {
		if err := registered.view.accept(batch, locators, sequence); err != nil {
			return fmt.Errorf("projection %s refuses commit: %w", registered.name, err)
		}
	}
	return nil
}

func (s *catalogState) apply(batch artifact.Batch, locators map[artifact.ID]contentLocator, sequence uint64) {
	for _, registered := range projections(s) {
		registered.view.applyCommit(batch, locators, sequence)
	}
}

func (s catalogState) delta(batch artifact.Batch) artifact.Batch {
	delta := artifact.Batch{Key: batch.Key, ExpectedHead: artifact.ClonePointer(batch.ExpectedHead)}
	s.artifacts.delta(batch, &delta)
	s.contents.delta(batch, &delta)
	s.aliases.delta(batch, &delta)
	s.lineage.delta(batch, &delta)
	s.causality.delta(batch, &delta)
	s.locations.delta(batch, &delta)
	return delta
}

func (s catalogState) commit(key string) (committedBatch, bool) { return s.commits.lookup(key) }

func (s *catalogState) addCommit(commit committedBatch) { s.commits.add(commit) }

// Store is the hash-chained artifact catalog: a replayed record log
// projected into slots, aliases, and lineage under one lock.
type Store struct {
	mu    sync.RWMutex
	log   *recordLog
	blobs blobStore
	// onPublished runs after a commit's head-consistent view is
	// published and the lock is released; subscribers may read the
	// store reentrantly.
	onPublished []func(artifact.CommitID, uint64)
	state       catalogState
	head        artifact.CommitID
	sequence    uint64
	replayEnd   int64
	readOnly    bool
	closed      bool
	fault       error
	root        string
	snapshot    SnapshotReplay
}

// Open replays the store under root and returns a writable handle.
func Open(root string) (*Store, error) {
	return open(root, false)
}

// OpenReadOnly replays the store under root without taking the write lock.
func OpenReadOnly(root string) (*Store, error) {
	return open(root, true)
}

func open(root string, readOnly bool) (*Store, error) {
	if root == "" {
		return nil, errors.New("overgodb: empty root")
	}
	// Per-projection checkpoints are the fastest verified anchor; any
	// defect in the set falls back to the monolithic snapshot, then to
	// full journal replay. Checkpoints are acceleration, not authority.
	state, anchor, loaded, checkpointFallback := loadProjectionCheckpoints(root)
	snapshot := SnapshotReplay{Loaded: loaded}
	if loaded {
		snapshot.Path = filepath.Join(root, checkpointDirectory)
	}
	if !loaded {
		state, anchor, snapshot = loadLatestSnapshot(root)
		if checkpointFallback != "" && snapshot.Fallback == "" {
			snapshot.Fallback = checkpointFallback
		}
		loaded = snapshot.Loaded
	}
	if !loaded {
		state = newCatalogState()
	}
	store := &Store{state: state, readOnly: readOnly, root: root, snapshot: snapshot, blobs: newBlobStore(root)}
	log, replay, err := openRecordLog(root, readOnly, anchor, store.applyRecord)
	if errors.Is(err, ErrSnapshotAnchor) && loaded {
		store.state = newCatalogState()
		store.snapshot.Loaded = false
		store.snapshot.Fallback = ErrSnapshotAnchor.Error()
		log, replay, err = openRecordLog(root, readOnly, replayAnchor{}, store.applyRecord)
	}
	if err != nil {
		return nil, err
	}
	store.log = log
	store.head = replay.head
	store.sequence = replay.sequence
	store.replayEnd = replay.validEnd
	return store, nil
}

// SnapshotReplay reports checkpoint use or fallback.
func (s *Store) SnapshotReplay() SnapshotReplay {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.snapshot
}

func (s *Store) applyRecord(record logRecord) error {
	return commitCoordinator{state: &s.state, log: s.log, blobs: s.blobs}.replay(record)
}

// Refresh applies the validated committed tail to a read-only store.
func (s *Store) Refresh(ctx context.Context) error {
	if err := contextError(ctx); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ready(false); err != nil {
		return err
	}
	if err := contextError(ctx); err != nil {
		return err
	}
	if !s.readOnly {
		return nil
	}
	if info, statErr := os.Stat(filepath.Join(s.root, storeFilename)); statErr == nil && info.Size() < s.replayEnd {
		// The writer sealed the active segment this reader was tailing;
		// rebuild the view from the journal chain at the current head.
		return s.reopenLocked()
	}
	result, err := s.log.refresh(replayAnchor{sequence: s.sequence, head: s.head, offset: s.replayEnd}, s.replayEnd, s.applyRecord)
	if errors.Is(err, ErrSnapshotAnchor) {
		// The tail no longer chains from this reader's head: the writer
		// rotated and refilled the active segment past the old offset.
		return s.reopenLocked()
	}
	if err != nil {
		s.fault = err
		return err
	}
	s.head, s.sequence, s.replayEnd = result.head, result.sequence, result.validEnd
	return nil
}

// reopenLocked rebuilds a read-only handle in place after the writer
// rotated segments underneath it.
func (s *Store) reopenLocked() error {
	_ = s.log.Close()
	s.state = newCatalogState()
	log, replay, err := openRecordLog(s.root, true, replayAnchor{}, s.applyRecord)
	if err != nil {
		s.fault = err
		return err
	}
	s.log = log
	s.head, s.sequence, s.replayEnd = replay.head, replay.sequence, replay.validEnd
	s.snapshot = SnapshotReplay{Fallback: "segment rotation reopen"}
	return nil
}

// sealActiveSegment makes the active journal immutable and starts a
// fresh one: the file is renamed into the sealed set -- named by its
// last sequence so lexical order is chain order -- and a new active
// segment begins at the same commit chain. Sealing refuses while any
// catalog content still lives inline in the journal: legacy stores
// migrate through Rebuild before they segment.
func (s *Store) sealActiveSegment() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ready(true); err != nil {
		return err
	}
	if s.sequence == 0 {
		return errors.New("overgodb: cannot seal an empty journal")
	}
	if s.replayEnd <= storeHeaderBytes {
		return errors.New("overgodb: nothing to seal: the active segment holds no frames")
	}
	for id, locator := range s.state.contents.locators {
		if !locator.blob {
			return fmt.Errorf("overgodb: cannot seal: content %s is inline in the journal; migrate with Rebuild first", id)
		}
	}
	directory := filepath.Join(s.root, segmentDirectory)
	if err := os.MkdirAll(directory, storeDirectoryMode); err != nil {
		return fmt.Errorf("overgodb: create segment directory: %w", err)
	}
	if err := s.log.file.Sync(); err != nil {
		return fmt.Errorf("overgodb: sync active segment: %w", err)
	}
	// Copy-then-truncate keeps the active file handle stable for every
	// concurrent reader (Windows refuses to rename a tailed file): the
	// sealed bytes publish through a staged rename inside the segment
	// directory, and only then does the active file shrink back to its
	// header -- which is exactly the signal tailing readers rebuild on.
	if _, err := s.log.file.Seek(0, io.SeekStart); err != nil {
		return fmt.Errorf("overgodb: seek active segment: %w", err)
	}
	staging, err := os.CreateTemp(directory, ".staging-*")
	if err != nil {
		return fmt.Errorf("overgodb: stage sealed segment: %w", err)
	}
	stagingPath := staging.Name()
	_, copyErr := io.Copy(staging, s.log.file)
	syncErr := staging.Sync()
	closeErr := staging.Close()
	if err := errors.Join(copyErr, syncErr, closeErr); err != nil {
		_ = os.Remove(stagingPath)
		return fmt.Errorf("overgodb: copy sealed segment: %w", err)
	}
	sealed := filepath.Join(directory, fmt.Sprintf("%020d%s", s.sequence, segmentExtension))
	if _, err := os.Stat(sealed); err == nil {
		_ = os.Remove(stagingPath)
		return fmt.Errorf("overgodb: sealed segment %s already exists", filepath.Base(sealed))
	}
	if err := os.Rename(stagingPath, sealed); err != nil {
		_ = os.Remove(stagingPath)
		return fmt.Errorf("overgodb: publish sealed segment: %w", err)
	}
	if err := fsatomic.SyncDirectory(directory); err != nil {
		return err
	}
	for index := range s.state.commits.ordered {
		commit := &s.state.commits.ordered[index]
		if commit.coordinate.segment == activeSegment && commit.coordinate.valid() {
			commit.coordinate.segment = s.sequence
		}
	}
	if err := s.log.file.Truncate(storeHeaderBytes); err != nil {
		return fmt.Errorf("overgodb: reset active segment: %w", err)
	}
	if err := s.log.file.Sync(); err != nil {
		return fmt.Errorf("overgodb: sync fresh active segment: %w", err)
	}
	if _, err := s.log.file.Seek(0, io.SeekEnd); err != nil {
		return fmt.Errorf("overgodb: seek fresh active end: %w", err)
	}
	s.replayEnd = storeHeaderBytes
	return nil
}

// subscribePublished registers a head-publication observer. Callbacks
// run outside the store lock, after every projection has applied, so
// an observer reads one consistent head or a later one -- never a
// mixed view.
func (s *Store) subscribePublished(observer func(artifact.CommitID, uint64)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.onPublished = append(s.onPublished, observer)
}

// Commit appends one effective transaction: the batch is normalized,
// reduced to its delta against current state, and chained onto the head.
func (s *Store) Commit(ctx context.Context, batch artifact.Batch) (artifact.CommitID, error) {
	id, advanced, err := s.commitPublished(ctx, batch)
	if err != nil || !advanced {
		return id, err
	}
	s.mu.RLock()
	observers := slices.Clone(s.onPublished)
	head, sequence := s.head, s.sequence
	s.mu.RUnlock()
	for _, observer := range observers {
		observer(head, sequence)
	}
	return id, nil
}

func (s *Store) commitPublished(ctx context.Context, batch artifact.Batch) (artifact.CommitID, bool, error) {
	if err := contextError(ctx); err != nil {
		return artifact.CommitID{}, false, err
	}
	normalized, payloadHash, err := encodeBatch(batch)
	if err != nil {
		return artifact.CommitID{}, false, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ready(true); err != nil {
		return artifact.CommitID{}, false, err
	}
	if err := contextError(ctx); err != nil {
		return artifact.CommitID{}, false, err
	}
	coordinator := commitCoordinator{state: &s.state, log: s.log, blobs: s.blobs}
	advance, replayed, err := coordinator.commit(normalized, payloadHash, s.head, s.sequence)
	if err != nil {
		var fault appendFault
		if errors.As(err, &fault) {
			s.fault = fault.cause
			return advance.id, false, fmt.Errorf("%w: %w", ErrStoreFaulted, fault.cause)
		}
		return artifact.CommitID{}, false, err
	}
	if replayed {
		return advance.id, false, nil
	}
	s.sequence = advance.sequence
	s.head = advance.id
	s.replayEnd = advance.replayEnd
	return advance.id, true, nil
}

func (s *Store) transactionFits(batch artifact.Batch) (bool, error) {
	normalized, requestHash, err := encodeBatch(batch)
	if errors.Is(err, errPayloadLimit) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(true); err != nil {
		return false, err
	}
	if err := s.state.validate(normalized); err != nil {
		return false, err
	}
	delta := s.state.delta(normalized)
	if delta.Empty() {
		return true, nil
	}
	_, _, err = encodeTransaction(requestHash, delta)
	if errors.Is(err, errPayloadLimit) {
		return false, nil
	}
	return err == nil, err
}

// Artifact returns the committed descriptor for id, reporting presence.
func (s *Store) Artifact(ctx context.Context, id artifact.ID) (artifact.Descriptor, bool, error) {
	if err := contextError(ctx); err != nil {
		return artifact.Descriptor{}, false, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return artifact.Descriptor{}, false, err
	}
	record, ok := s.state.artifacts.record(id)
	if !ok {
		return artifact.Descriptor{}, false, nil
	}
	return record.descriptor, true, nil
}

// Manifest returns a clone of the committed manifest for id, reporting presence.
func (s *Store) Manifest(ctx context.Context, id artifact.ID) (artifact.Manifest, bool, error) {
	if err := contextError(ctx); err != nil {
		return artifact.Manifest{}, false, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return artifact.Manifest{}, false, err
	}
	record, ok := s.state.artifacts.record(id)
	if !ok || !record.hasManifest {
		return artifact.Manifest{}, false, nil
	}
	return record.manifest.Clone(), true, nil
}

// OpenContent returns the descriptor and a payload reader positioned at
// the committed content bytes for id, reporting presence.
func (s *Store) OpenContent(ctx context.Context, id artifact.ID) (artifact.Descriptor, io.Reader, bool, error) {
	if err := contextError(ctx); err != nil {
		return artifact.Descriptor{}, nil, false, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return artifact.Descriptor{}, nil, false, err
	}
	record, ok := s.state.artifacts.record(id)
	locator, hasContent := s.state.contents.locator(id)
	if !ok || !hasContent {
		return artifact.Descriptor{}, nil, false, nil
	}
	reader, err := s.openLocator(id, locator)
	if err != nil {
		return artifact.Descriptor{}, nil, false, err
	}
	return record.descriptor, reader, true, nil
}

// PresentContents reports, in caller order, the subset of ids whose
// content bytes are committed, under one state acquisition.
func (s *Store) PresentContents(ctx context.Context, ids []artifact.ID) ([]artifact.ID, error) {
	if err := contextError(ctx); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return nil, err
	}
	present := make([]artifact.ID, 0, len(ids))
	for _, id := range ids {
		if s.state.contents.has(id) {
			present = append(present, id)
		}
	}
	return present, nil
}

// VisitContents streams requested content in storage order.
func (s *Store) VisitContents(ctx context.Context, ids []artifact.ID, visit func(artifact.Descriptor, io.Reader) error) error {
	if err := contextError(ctx); err != nil {
		return err
	}
	if visit == nil {
		return errors.New("overgodb: nil content visitor")
	}
	type entry struct {
		id         artifact.ID
		descriptor artifact.Descriptor
		locator    contentLocator
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return err
	}
	entries := make([]entry, 0, len(ids))
	seen := make(map[artifact.ID]struct{}, len(ids))
	for _, id := range ids {
		if _, duplicate := seen[id]; duplicate {
			continue
		}
		seen[id] = struct{}{}
		record, found := s.state.artifacts.record(id)
		locator, hasContent := s.state.contents.locator(id)
		if !found || !hasContent {
			return fmt.Errorf("overgodb: content is absent: %s", id)
		}
		entries = append(entries, entry{id: id, descriptor: record.descriptor, locator: locator})
	}
	sort.Slice(entries, func(i, j int) bool {
		left, right := entries[i].locator, entries[j].locator
		return left.offset < right.offset
	})
	for _, entry := range entries {
		if err := contextError(ctx); err != nil {
			return err
		}
		reader, err := s.openLocator(entry.id, entry.locator)
		if err != nil {
			return err
		}
		err = visit(entry.descriptor, reader)
		if closer, ok := reader.(io.Closer); ok {
			_ = closer.Close()
		}
		if err != nil {
			return err
		}
	}
	return nil
}

// openLocator opens content bytes wherever they live: the blob store
// for referenced content, the journal frame for legacy inline frames.
// An absent required blob is exact evidence, never a silent degrade.
func (s *Store) openLocator(id artifact.ID, locator contentLocator) (io.Reader, error) {
	if !locator.blob {
		return s.log.openContent(locator), nil
	}
	reader, err := s.blobs.open(id, uint64(locator.size))
	if err != nil {
		return nil, fmt.Errorf("overgodb: committed content %s requires an absent blob: %w", id, err)
	}
	return reader, nil
}

func (s *Store) materializeContent(id artifact.ID, locator contentLocator) ([]byte, error) {
	reader, err := s.openLocator(id, locator)
	if err != nil {
		return nil, err
	}
	if closer, ok := reader.(io.Closer); ok {
		defer closer.Close()
	}
	data, err := io.ReadAll(io.LimitReader(reader, locator.size+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) != locator.size {
		return nil, errors.New("overgodb: content locator size differs")
	}
	return data, nil
}

// HasContent reports whether committed payload bytes exist for id.
func (s *Store) HasContent(ctx context.Context, id artifact.ID) (bool, error) {
	if err := contextError(ctx); err != nil {
		return false, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return false, err
	}
	return s.state.contents.has(id), nil
}

// ResolveAlias returns the artifact currently bound to name, reporting presence.
func (s *Store) ResolveAlias(ctx context.Context, name string) (artifact.ID, bool, error) {
	if err := contextError(ctx); err != nil {
		return artifact.ID{}, false, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return artifact.ID{}, false, err
	}
	id, ok := s.state.aliases.resolve(name)
	return id, ok, nil
}

// Parents returns the lineage edges naming id as child, sorted deterministically.
func (s *Store) Parents(ctx context.Context, id artifact.ID) ([]artifact.Lineage, error) {
	return s.lineageFor(ctx, id, true)
}

// Children returns the lineage edges naming id as parent, sorted deterministically.
func (s *Store) Children(ctx context.Context, id artifact.ID) ([]artifact.Lineage, error) {
	return s.lineageFor(ctx, id, false)
}

// Locations returns the committed on-disk locations recorded for id.
func (s *Store) Locations(ctx context.Context, id artifact.ID) ([]artifact.Location, error) {
	if err := contextError(ctx); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return nil, err
	}
	return slices.Clone(s.state.locations.of(id)), nil
}

func (s *Store) lineageFor(ctx context.Context, id artifact.ID, parents bool) ([]artifact.Lineage, error) {
	if err := contextError(ctx); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return nil, err
	}
	edges := s.state.lineage.childrenOf(id)
	if parents {
		edges = s.state.lineage.parentsOf(id)
	}
	result := make([]artifact.Lineage, 0, len(edges))
	for _, key := range edges {
		result = append(result, artifact.Lineage{Child: key.child, Parent: key.parent, Relation: key.relation})
	}
	sort.Slice(result, func(i, j int) bool {
		left, right := result[i], result[j]
		if left.Relation != right.Relation {
			return left.Relation < right.Relation
		}
		if left.Child != right.Child {
			return left.Child.String() < right.Child.String()
		}
		return left.Parent.String() < right.Parent.String()
	})
	return result, nil
}

// Head returns the current chain head commit and its sequence number.
func (s *Store) Head() (artifact.CommitID, uint64) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.head, s.sequence
}

// Close releases the record log and file lock; further calls are no-ops.
func (s *Store) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	return s.log.Close()
}

func (s *Store) ready(write bool) error {
	if s == nil || s.closed || s.log == nil {
		return ErrClosed
	}
	if s.fault != nil {
		return fmt.Errorf("%w: %w", ErrStoreFaulted, s.fault)
	}
	if write && s.readOnly {
		return ErrReadOnly
	}
	return nil
}

// encodeBatch normalizes one batch and returns it with its canonical
// request digest; the encoded bytes exist only to bound and hash the
// request, so they are not returned.
func encodeBatch(batch artifact.Batch) (artifact.Batch, [sha256.Size]byte, error) {
	normalized, err := normalizeBatch(batch)
	if err != nil {
		return artifact.Batch{}, [sha256.Size]byte{}, err
	}
	payload, err := json.Marshal(normalized)
	if err != nil {
		return artifact.Batch{}, [sha256.Size]byte{}, fmt.Errorf("overgodb: encode batch: %w", err)
	}
	if len(payload) > maxFramePayload {
		return artifact.Batch{}, [sha256.Size]byte{}, fmt.Errorf("%w: batch", errPayloadLimit)
	}
	return normalized, sha256.Sum256(payload), nil
}

// encodeTransaction persists a small commit envelope: ordering,
// identities, descriptors, and transaction intent. Content bytes are
// referenced through the blob store, never embedded.
func encodeTransaction(request [sha256.Size]byte, delta artifact.Batch) ([]byte, map[artifact.ID]contentLocator, error) {
	contents := delta.Contents
	delta.Contents = nil
	descriptors := make([]artifact.Descriptor, len(contents))
	for index := range contents {
		descriptors[index] = contents[index].Descriptor
	}
	metadata, err := json.Marshal(persistedTransaction{Request: request, Delta: delta, BlobContent: descriptors})
	if err != nil {
		return nil, nil, fmt.Errorf("overgodb: encode transaction: %w", err)
	}
	payload := binary.LittleEndian.AppendUint32(nil, uint32(len(metadata)))
	payload = append(payload, metadata...)
	locators := make(map[artifact.ID]contentLocator, len(contents))
	for _, content := range contents {
		locators[content.Descriptor.ID] = contentLocator{size: int64(len(content.Data)), blob: true}
	}
	if len(payload) > maxFramePayload {
		return nil, nil, fmt.Errorf("%w: transaction", errPayloadLimit)
	}
	return payload, locators, nil
}

func decodeRecord(record logRecord) (artifact.Batch, [sha256.Size]byte, map[artifact.ID]contentLocator, error) {
	if len(record.payload) < binary.Size(uint32(0)) {
		return artifact.Batch{}, [sha256.Size]byte{}, nil, errors.New("overgodb: short transaction metadata")
	}
	metadataSize := int(binary.LittleEndian.Uint32(record.payload))
	metadataOffset := binary.Size(uint32(0))
	if metadataSize > len(record.payload)-metadataOffset {
		return artifact.Batch{}, [sha256.Size]byte{}, nil, errors.New("overgodb: invalid transaction metadata size")
	}
	metadata := record.payload[metadataOffset : metadataOffset+metadataSize]
	var transaction persistedTransaction
	if err := strictjson.DecodeBytes(metadata, &transaction); err != nil {
		return artifact.Batch{}, [sha256.Size]byte{}, nil, fmt.Errorf("overgodb: decode transaction: %w", err)
	}
	normalized, err := normalizeBatch(transaction.Delta)
	if err != nil {
		return artifact.Batch{}, [sha256.Size]byte{}, nil, err
	}
	canonical, err := json.Marshal(persistedTransaction{Request: transaction.Request, Delta: normalized, Content: transaction.Content, BlobContent: transaction.BlobContent})
	if err != nil {
		return artifact.Batch{}, [sha256.Size]byte{}, nil, fmt.Errorf("overgodb: canonicalize transaction: %w", err)
	}
	if !bytes.Equal(canonical, metadata) {
		return artifact.Batch{}, [sha256.Size]byte{}, nil, errors.New("overgodb: non-canonical transaction metadata")
	}
	dataOffset := metadataOffset + metadataSize
	locators := make(map[artifact.ID]contentLocator, len(transaction.Content)+len(transaction.BlobContent))
	for _, descriptor := range transaction.BlobContent {
		if err := descriptor.Validate(); err != nil {
			return artifact.Batch{}, [sha256.Size]byte{}, nil, err
		}
		normalized.Contents = append(normalized.Contents, artifact.Content{Descriptor: descriptor})
		locators[descriptor.ID] = contentLocator{size: int64(descriptor.Size), blob: true}
	}
	for _, descriptor := range transaction.Content {
		end := dataOffset + int(descriptor.Size)
		if end < dataOffset || end > len(record.payload) {
			return artifact.Batch{}, [sha256.Size]byte{}, nil, errors.New("overgodb: invalid transaction content bounds")
		}
		content := artifact.Content{Descriptor: descriptor, Data: record.payload[dataOffset:end]}
		if err := content.Validate(); err != nil {
			return artifact.Batch{}, [sha256.Size]byte{}, nil, err
		}
		normalized.Contents = append(normalized.Contents, content)
		locators[descriptor.ID] = contentLocator{offset: int64(dataOffset), size: int64(descriptor.Size)}
		dataOffset = end
	}
	if dataOffset != len(record.payload) {
		return artifact.Batch{}, [sha256.Size]byte{}, nil, errors.New("overgodb: trailing transaction content")
	}
	bindContentLocators(locators, record.offset)
	return normalized, transaction.Request, locators, nil
}

func bindContentLocators(locators map[artifact.ID]contentLocator, payloadOffset int64) {
	for id, locator := range locators {
		locator.offset += payloadOffset
		locators[id] = locator
	}
}

func normalizeBatch(batch artifact.Batch) (artifact.Batch, error) {
	result := artifact.Batch{
		Key:          batch.Key,
		ExpectedHead: artifact.ClonePointer(batch.ExpectedHead),
		Artifacts:    slices.Clone(batch.Artifacts),
		Contents:     cloneValues(batch.Contents),
		Manifests:    cloneValues(batch.Manifests),
		Lineage:      slices.Clone(batch.Lineage),
		Causality:    cloneValues(batch.Causality),
		Aliases:      artifact.CloneAliasBindings(batch.Aliases),
		Locations:    slices.Clone(batch.Locations),
	}
	for _, content := range result.Contents {
		result.Artifacts = append(result.Artifacts, content.Descriptor)
	}
	for _, manifest := range result.Manifests {
		descriptor, err := manifest.Descriptor()
		if err != nil {
			return artifact.Batch{}, err
		}
		result.Artifacts = append(result.Artifacts, descriptor)
		result.Lineage = append(result.Lineage, manifest.Lineage()...)
	}
	if err := result.Validate(); err != nil {
		return artifact.Batch{}, err
	}
	sort.Slice(result.Artifacts, func(i, j int) bool {
		return result.Artifacts[i].ID.String() < result.Artifacts[j].ID.String()
	})
	artifacts := result.Artifacts[:0]
	var previousArtifact artifact.Descriptor
	haveArtifact := false
	for _, descriptor := range result.Artifacts {
		if haveArtifact && previousArtifact.ID == descriptor.ID {
			if previousArtifact != descriptor {
				return artifact.Batch{}, fmt.Errorf("%w: %s", ErrArtifactConflict, descriptor.ID)
			}
			continue
		}
		artifacts = append(artifacts, descriptor)
		previousArtifact, haveArtifact = descriptor, true
	}
	result.Artifacts = artifacts
	sort.Slice(result.Contents, func(i, j int) bool {
		return result.Contents[i].Descriptor.ID.String() < result.Contents[j].Descriptor.ID.String()
	})
	if duplicate, found := adjacentDuplicate(result.Contents, func(left, right artifact.Content) bool {
		return left.Descriptor.ID == right.Descriptor.ID
	}); found {
		return artifact.Batch{}, fmt.Errorf("overgodb: duplicate content %s", duplicate.Descriptor.ID)
	}
	sort.Slice(result.Manifests, func(i, j int) bool {
		return result.Manifests[i].ID.String() < result.Manifests[j].ID.String()
	})
	if duplicate, found := adjacentDuplicate(result.Manifests, func(left, right artifact.Manifest) bool {
		return left.ID == right.ID
	}); found {
		return artifact.Batch{}, fmt.Errorf("overgodb: duplicate manifest %s", duplicate.ID)
	}
	sort.Slice(result.Aliases, func(i, j int) bool {
		return result.Aliases[i].Name < result.Aliases[j].Name
	})
	if duplicate, found := adjacentDuplicate(result.Aliases, func(left, right artifact.AliasBinding) bool {
		return left.Name == right.Name
	}); found {
		return artifact.Batch{}, fmt.Errorf("overgodb: duplicate alias %q", duplicate.Name)
	}
	sort.Slice(result.Lineage, func(i, j int) bool {
		left, right := result.Lineage[i], result.Lineage[j]
		if left.Child != right.Child {
			return left.Child.String() < right.Child.String()
		}
		if left.Parent != right.Parent {
			return left.Parent.String() < right.Parent.String()
		}
		return left.Relation < right.Relation
	})
	lineage := result.Lineage[:0]
	var previousEdge artifact.Lineage
	haveEdge := false
	for _, edge := range result.Lineage {
		if haveEdge && previousEdge == edge {
			continue
		}
		lineage = append(lineage, edge)
		previousEdge, haveEdge = edge, true
	}
	result.Lineage = lineage
	sort.Slice(result.Causality, func(i, j int) bool {
		return artifact.CompareID(result.Causality[i].Execution, result.Causality[j].Execution) < 0
	})
	causality := result.Causality[:0]
	for _, link := range result.Causality {
		if len(causality) != 0 && causality[len(causality)-1].Execution == link.Execution {
			if !causality[len(causality)-1].Equal(link) {
				return artifact.Batch{}, fmt.Errorf("overgodb: conflicting causal execution %s", link.Execution)
			}
			continue
		}
		causality = append(causality, link)
	}
	result.Causality = causality
	sort.Slice(result.Locations, func(i, j int) bool {
		left, right := result.Locations[i], result.Locations[j]
		if left.Artifact != right.Artifact {
			return left.Artifact.String() < right.Artifact.String()
		}
		if left.Kind != right.Kind {
			return left.Kind < right.Kind
		}
		return left.Value < right.Value
	})
	if duplicate, found := adjacentDuplicate(result.Locations, func(left, right artifact.LocationEvent) bool {
		if left.Artifact == right.Artifact && left.Kind == right.Kind && left.Value == right.Value {
			return true
		}
		return false
	}); found {
		return artifact.Batch{}, fmt.Errorf("overgodb: duplicate location mutation %q", duplicate.Value)
	}
	return result, nil
}

func adjacentDuplicate[T any](values []T, equal func(T, T) bool) (T, bool) {
	var previous T
	havePrevious := false
	for _, value := range values {
		if havePrevious && equal(previous, value) {
			return value, true
		}
		previous, havePrevious = value, true
	}
	var zero T
	return zero, false
}

func cloneValues[T interface{ Clone() T }](values []T) []T {
	if values == nil {
		return nil
	}
	result := make([]T, len(values))
	for index := range result {
		result[index] = values[index].Clone()
	}
	return result
}

func contextError(ctx context.Context) error {
	if ctx == nil {
		return errors.New("overgodb: nil context")
	}
	return ctx.Err()
}

var _ artifact.Repository = (*Store)(nil)
