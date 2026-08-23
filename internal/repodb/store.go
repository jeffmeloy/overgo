package repodb

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"sort"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

var (
	ErrClosed           = errors.New("repodb: store is closed")
	ErrReadOnly         = errors.New("repodb: store is read-only")
	ErrBatchKeyConflict = errors.New("repodb: batch key already names different content")
	ErrArtifactConflict = errors.New("repodb: artifact identity has conflicting facts")
	ErrAliasConflict    = errors.New("repodb: alias compare-and-set failed")
	ErrHeadConflict     = artifact.ErrCommitPrecondition
	ErrLineageCycle     = errors.New("repodb: lineage cycle")
	ErrStoreFaulted     = errors.New("repodb: store requires reopen after uncertain append")
	ErrSnapshotAnchor   = errors.New("repodb: snapshot anchor is absent from commit chain")
	// ErrNoChange reports a batch with no effective catalog mutation.
	ErrNoChange = errors.New("repodb: batch has no effective catalog change")
)

type relationKey struct {
	child    artifact.ID
	parent   artifact.ID
	relation artifact.Relation
}

type locationKey struct {
	artifact artifact.ID
	kind     artifact.LocationKind
	value    string
}

type committedBatch struct {
	id       artifact.CommitID
	payload  [sha256.Size]byte
	sequence uint64
}

type persistedTransaction struct {
	Request [sha256.Size]byte     `json:"request"`
	Delta   artifact.Batch        `json:"delta"`
	Content []artifact.Descriptor `json:"content,omitempty"`
}

type contentLocator struct {
	offset        int64
	size          int64
	frameVersion  uint16
	payloadOffset int64
	payloadSize   int64
}

type catalogState struct {
	artifacts           map[artifact.ID]artifact.Descriptor
	artifactsByMedia    map[string]map[artifact.ID]struct{}
	artifactsBySchema   map[string]map[artifact.ID]struct{}
	contents            map[artifact.ID]contentLocator
	manifests           map[artifact.ID]artifact.Manifest
	aliases             map[string]artifact.ID
	lineage             map[relationKey]artifact.Lineage
	parentEdges         map[artifact.ID]map[relationKey]struct{}
	childEdges          map[artifact.ID]map[relationKey]struct{}
	locations           map[locationKey]artifact.Location
	locationsByArtifact map[artifact.ID]map[locationKey]artifact.Location
	commits             map[string]committedBatch
}

func newCatalogState() catalogState {
	return catalogState{
		artifacts:           map[artifact.ID]artifact.Descriptor{},
		artifactsByMedia:    map[string]map[artifact.ID]struct{}{},
		artifactsBySchema:   map[string]map[artifact.ID]struct{}{},
		contents:            map[artifact.ID]contentLocator{},
		manifests:           map[artifact.ID]artifact.Manifest{},
		aliases:             map[string]artifact.ID{},
		lineage:             map[relationKey]artifact.Lineage{},
		parentEdges:         map[artifact.ID]map[relationKey]struct{}{},
		childEdges:          map[artifact.ID]map[relationKey]struct{}{},
		locations:           map[locationKey]artifact.Location{},
		locationsByArtifact: map[artifact.ID]map[locationKey]artifact.Location{},
		commits:             map[string]committedBatch{},
	}
}

func (s catalogState) validate(batch artifact.Batch) error {
	added := make(map[artifact.ID]struct{}, len(batch.Artifacts))
	for _, descriptor := range batch.Artifacts {
		if current, ok := s.artifacts[descriptor.ID]; ok && current != descriptor {
			return fmt.Errorf("%w: %s", ErrArtifactConflict, descriptor.ID)
		}
		added[descriptor.ID] = struct{}{}
	}
	for _, manifest := range batch.Manifests {
		if current, ok := s.manifests[manifest.ID]; ok && !sameManifest(current, manifest) {
			return fmt.Errorf("%w: manifest %s", ErrArtifactConflict, manifest.ID)
		}
	}
	for _, binding := range batch.Aliases {
		if !binding.Remove {
			if _, stored := s.artifacts[binding.Target]; !stored {
				if _, pending := added[binding.Target]; !pending {
					return fmt.Errorf("repodb: alias %q targets unknown artifact %s", binding.Name, binding.Target)
				}
			}
		}
		current, exists := s.aliases[binding.Name]
		if binding.Previous == nil && exists {
			return fmt.Errorf("%w: %q is already bound", ErrAliasConflict, binding.Name)
		}
		if binding.Previous != nil && (!exists || current != *binding.Previous) {
			return fmt.Errorf("%w: %q has unexpected target", ErrAliasConflict, binding.Name)
		}
	}
	pendingParents := map[artifact.ID]map[artifact.ID]struct{}{}
	for _, edge := range batch.Lineage {
		if !s.hasArtifact(edge.Child, added) {
			return fmt.Errorf("repodb: lineage child is unknown: %s", edge.Child)
		}
		if !s.hasArtifact(edge.Parent, added) {
			return fmt.Errorf("repodb: lineage parent is unknown: %s", edge.Parent)
		}
		key := relationKey{child: edge.Child, parent: edge.Parent, relation: edge.Relation}
		if _, exists := s.lineage[key]; exists {
			continue
		}
		if s.reaches(edge.Parent, edge.Child, pendingParents) {
			return fmt.Errorf("%w: %s -> %s", ErrLineageCycle, edge.Child, edge.Parent)
		}
		parents := pendingParents[edge.Child]
		if parents == nil {
			parents = map[artifact.ID]struct{}{}
			pendingParents[edge.Child] = parents
		}
		parents[edge.Parent] = struct{}{}
	}
	for _, event := range batch.Locations {
		if !s.hasArtifact(event.Artifact, added) {
			return fmt.Errorf("repodb: location artifact is unknown: %s", event.Artifact)
		}
		key := locationKey{artifact: event.Artifact, kind: event.Kind, value: event.Value}
		_, exists := s.locations[key]
		if event.Action == artifact.LocationRemove && !exists {
			return fmt.Errorf("repodb: remove unknown location %q", event.Value)
		}
	}
	return nil
}

func (s catalogState) hasArtifact(id artifact.ID, added map[artifact.ID]struct{}) bool {
	if _, ok := s.artifacts[id]; ok {
		return true
	}
	_, ok := added[id]
	return ok
}

func (s *catalogState) apply(batch artifact.Batch, locators map[artifact.ID]contentLocator) {
	for _, descriptor := range batch.Artifacts {
		s.artifacts[descriptor.ID] = descriptor
		indexDescriptor(s.artifactsByMedia, descriptor.MediaType, descriptor.ID)
		indexDescriptor(s.artifactsBySchema, descriptor.Schema, descriptor.ID)
	}
	for _, content := range batch.Contents {
		s.contents[content.Descriptor.ID] = locators[content.Descriptor.ID]
	}
	for _, manifest := range batch.Manifests {
		s.manifests[manifest.ID] = manifest.Clone()
	}
	for _, binding := range batch.Aliases {
		if binding.Remove {
			delete(s.aliases, binding.Name)
		} else {
			s.aliases[binding.Name] = binding.Target
		}
	}
	for _, edge := range batch.Lineage {
		key := relationKey{child: edge.Child, parent: edge.Parent, relation: edge.Relation}
		if _, exists := s.lineage[key]; exists {
			continue
		}
		s.lineage[key] = edge
		indexRelation(s.parentEdges, edge.Child, key)
		indexRelation(s.childEdges, edge.Parent, key)
	}
	for _, event := range batch.Locations {
		key := locationKey{artifact: event.Artifact, kind: event.Kind, value: event.Value}
		if event.Action == artifact.LocationAdd {
			s.locations[key] = event.Location
			locations := s.locationsByArtifact[event.Artifact]
			if locations == nil {
				locations = map[locationKey]artifact.Location{}
				s.locationsByArtifact[event.Artifact] = locations
			}
			locations[key] = event.Location
		} else {
			delete(s.locations, key)
			locations := s.locationsByArtifact[event.Artifact]
			delete(locations, key)
			if len(locations) == 0 {
				delete(s.locationsByArtifact, event.Artifact)
			}
		}
	}
}

func (s catalogState) delta(batch artifact.Batch) artifact.Batch {
	delta := artifact.Batch{Key: batch.Key, ExpectedHead: cloneCommitID(batch.ExpectedHead)}
	for _, descriptor := range batch.Artifacts {
		if _, exists := s.artifacts[descriptor.ID]; !exists {
			delta.Artifacts = append(delta.Artifacts, descriptor)
		}
	}
	for _, content := range batch.Contents {
		if _, exists := s.contents[content.Descriptor.ID]; !exists {
			delta.Contents = append(delta.Contents, content)
		}
	}
	for _, manifest := range batch.Manifests {
		if _, exists := s.manifests[manifest.ID]; !exists {
			delta.Manifests = append(delta.Manifests, manifest)
		}
	}
	for _, binding := range batch.Aliases {
		current, exists := s.aliases[binding.Name]
		if binding.Remove || !exists || current != binding.Target {
			delta.Aliases = append(delta.Aliases, binding)
		}
	}
	for _, edge := range batch.Lineage {
		key := relationKey{child: edge.Child, parent: edge.Parent, relation: edge.Relation}
		if _, exists := s.lineage[key]; !exists {
			delta.Lineage = append(delta.Lineage, edge)
		}
	}
	for _, event := range batch.Locations {
		key := locationKey{artifact: event.Artifact, kind: event.Kind, value: event.Value}
		_, exists := s.locations[key]
		if event.Action == artifact.LocationRemove || !exists {
			delta.Locations = append(delta.Locations, event)
		}
	}
	return delta
}

func indexDescriptor(index map[string]map[artifact.ID]struct{}, key string, id artifact.ID) {
	if key == "" {
		return
	}
	ids := index[key]
	if ids == nil {
		ids = map[artifact.ID]struct{}{}
		index[key] = ids
	}
	ids[id] = struct{}{}
}

func indexRelation(
	index map[artifact.ID]map[relationKey]struct{},
	id artifact.ID,
	key relationKey,
) {
	edges := index[id]
	if edges == nil {
		edges = map[relationKey]struct{}{}
		index[id] = edges
	}
	edges[key] = struct{}{}
}

func sameManifest(left, right artifact.Manifest) bool {
	return left.Version == right.Version && left.ID == right.ID && slices.Equal(left.Components, right.Components)
}

func (s catalogState) reaches(start, target artifact.ID, pending map[artifact.ID]map[artifact.ID]struct{}) bool {
	if start == target {
		return true
	}
	seen := map[artifact.ID]struct{}{start: {}}
	queue := []artifact.ID{start}
	visit := func(parent artifact.ID) bool {
		if parent == target {
			return true
		}
		if _, ok := seen[parent]; ok {
			return false
		}
		seen[parent] = struct{}{}
		queue = append(queue, parent)
		return false
	}
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		for key := range s.parentEdges[current] {
			if visit(key.parent) {
				return true
			}
		}
		for parent := range pending[current] {
			if visit(parent) {
				return true
			}
		}
	}
	return false
}

// Store: hash-chained artifact catalog
type Store struct {
	mu        sync.RWMutex
	log       *recordLog
	state     catalogState
	head      artifact.CommitID
	sequence  uint64
	replayEnd int64
	readOnly  bool
	closed    bool
	fault     error
	root      string
}

func Open(root string) (*Store, error) {
	return open(root, false)
}

func OpenReadOnly(root string) (*Store, error) {
	return open(root, true)
}

func open(root string, readOnly bool) (*Store, error) {
	if root == "" {
		return nil, errors.New("repodb: empty root")
	}
	state, anchor, loaded := loadLatestSnapshot(root)
	if !loaded {
		state = newCatalogState()
	}
	store := &Store{state: state, readOnly: readOnly, root: root}
	log, replay, err := openRecordLog(root, readOnly, anchor, store.applyRecord)
	if errors.Is(err, ErrSnapshotAnchor) && loaded {
		store.state = newCatalogState()
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

func (s *Store) applyRecord(record logRecord) error {
	batch, payloadHash, locators, err := decodeRecord(record)
	if err != nil {
		return err
	}
	if batch.ExpectedHead != nil && *batch.ExpectedHead != record.previous {
		return fmt.Errorf("%w: recorded predecessor differs", ErrHeadConflict)
	}
	if _, exists := s.state.commits[batch.Key]; exists {
		return fmt.Errorf("%w: %q repeats in log", ErrBatchKeyConflict, batch.Key)
	}
	if err := s.state.validate(batch); err != nil {
		return err
	}
	s.state.apply(batch, locators)
	s.state.commits[batch.Key] = committedBatch{
		id: record.id, payload: payloadHash, sequence: record.sequence,
	}
	return nil
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
	result, err := s.log.refresh(replayAnchor{sequence: s.sequence, head: s.head}, s.replayEnd, s.applyRecord)
	if err != nil {
		s.fault = err
		return err
	}
	s.head, s.sequence, s.replayEnd = result.head, result.sequence, result.validEnd
	return nil
}

func (s *Store) Commit(ctx context.Context, batch artifact.Batch) (artifact.CommitID, error) {
	if err := contextError(ctx); err != nil {
		return artifact.CommitID{}, err
	}
	payload, normalized, payloadHash, err := encodeBatch(batch)
	if err != nil {
		return artifact.CommitID{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ready(true); err != nil {
		return artifact.CommitID{}, err
	}
	if err := contextError(ctx); err != nil {
		return artifact.CommitID{}, err
	}
	if committed, ok := s.state.commits[normalized.Key]; ok {
		if committed.payload == payloadHash {
			return committed.id, nil
		}
		return artifact.CommitID{}, fmt.Errorf("%w: %q", ErrBatchKeyConflict, normalized.Key)
	}
	if normalized.ExpectedHead != nil && *normalized.ExpectedHead != s.head {
		return artifact.CommitID{}, fmt.Errorf("%w: expected %s, have %s", ErrHeadConflict, *normalized.ExpectedHead, s.head)
	}
	if err := s.state.validate(normalized); err != nil {
		return artifact.CommitID{}, err
	}
	delta := s.state.delta(normalized)
	if delta.Empty() {
		return artifact.CommitID{}, ErrNoChange
	}
	payload, locators, err := encodeTransaction(payloadHash, delta)
	if err != nil {
		return artifact.CommitID{}, err
	}
	sequence := s.sequence + 1
	id, payloadOffset, replayEnd, err := s.log.append(sequence, s.head, payload)
	if err != nil {
		s.fault = err
		return id, fmt.Errorf("%w: %w", ErrStoreFaulted, err)
	}
	bindContentLocators(locators, payloadOffset, int64(len(payload)), frameVersion)
	s.state.apply(delta, locators)
	s.state.commits[normalized.Key] = committedBatch{id: id, payload: payloadHash, sequence: sequence}
	s.sequence = sequence
	s.head = id
	s.replayEnd = replayEnd
	return id, nil
}

func (s *Store) Artifact(ctx context.Context, id artifact.ID) (artifact.Descriptor, bool, error) {
	if err := contextError(ctx); err != nil {
		return artifact.Descriptor{}, false, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return artifact.Descriptor{}, false, err
	}
	value, ok := s.state.artifacts[id]
	return value, ok, nil
}

func (s *Store) Manifest(ctx context.Context, id artifact.ID) (artifact.Manifest, bool, error) {
	if err := contextError(ctx); err != nil {
		return artifact.Manifest{}, false, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return artifact.Manifest{}, false, err
	}
	value, ok := s.state.manifests[id]
	return value.Clone(), ok, nil
}

func (s *Store) OpenContent(ctx context.Context, id artifact.ID) (artifact.Descriptor, io.Reader, bool, error) {
	if err := contextError(ctx); err != nil {
		return artifact.Descriptor{}, nil, false, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return artifact.Descriptor{}, nil, false, err
	}
	locator, ok := s.state.contents[id]
	if !ok {
		return artifact.Descriptor{}, nil, false, nil
	}
	reader, err := s.log.openContent(locator, id)
	if err != nil {
		return artifact.Descriptor{}, nil, false, err
	}
	return s.state.artifacts[id], reader, true, nil
}

// VisitContents streams requested content in storage order.
func (s *Store) VisitContents(ctx context.Context, ids []artifact.ID, visit func(artifact.Descriptor, io.Reader) error) error {
	if err := contextError(ctx); err != nil {
		return err
	}
	if visit == nil {
		return errors.New("repodb: nil content visitor")
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
		locator, found := s.state.contents[id]
		if !found {
			return fmt.Errorf("repodb: content is absent: %s", id)
		}
		entries = append(entries, entry{id: id, descriptor: s.state.artifacts[id], locator: locator})
	}
	sort.Slice(entries, func(i, j int) bool {
		left, right := entries[i].locator, entries[j].locator
		if left.frameVersion != right.frameVersion {
			return left.frameVersion < right.frameVersion
		}
		if left.payloadOffset != right.payloadOffset {
			return left.payloadOffset < right.payloadOffset
		}
		return left.offset < right.offset
	})
	var legacyOffset int64
	var haveLegacy bool
	var legacy map[artifact.ID][]byte
	for _, entry := range entries {
		if err := contextError(ctx); err != nil {
			return err
		}
		var reader io.Reader
		if entry.locator.frameVersion < frameVersion {
			if !haveLegacy || entry.locator.payloadOffset != legacyOffset {
				var err error
				legacy, err = s.legacyContents(entry.locator)
				if err != nil {
					return err
				}
				legacyOffset = entry.locator.payloadOffset
				haveLegacy = true
			}
			data, found := legacy[entry.id]
			if !found {
				return errors.New("repodb: legacy content is absent")
			}
			reader = bytes.NewReader(data)
		} else {
			var err error
			reader, err = s.log.openContent(entry.locator, entry.id)
			if err != nil {
				return err
			}
		}
		if err := visit(entry.descriptor, reader); err != nil {
			return err
		}
	}
	return nil
}

func (s *Store) materializeContent(locator contentLocator, id artifact.ID) ([]byte, error) {
	reader, err := s.log.openContent(locator, id)
	if err != nil {
		return nil, err
	}
	data, err := io.ReadAll(io.LimitReader(reader, locator.size+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) != locator.size {
		return nil, errors.New("repodb: content locator size differs")
	}
	return data, nil
}

func (s *Store) materializeQueryContent(
	locator contentLocator,
	id artifact.ID,
	legacy map[int64]map[artifact.ID][]byte,
) ([]byte, error) {
	if locator.frameVersion >= frameVersion {
		return s.materializeContent(locator, id)
	}
	contents := legacy[locator.payloadOffset]
	if contents == nil {
		var err error
		contents, err = s.legacyContents(locator)
		if err != nil {
			return nil, err
		}
		legacy[locator.payloadOffset] = contents
	}
	data, found := contents[id]
	if !found {
		return nil, errors.New("repodb: legacy content is absent")
	}
	return data, nil
}

func (s *Store) legacyContents(locator contentLocator) (map[artifact.ID][]byte, error) {
	payload := make([]byte, locator.payloadSize)
	if _, err := s.log.file.ReadAt(payload, locator.payloadOffset); err != nil {
		return nil, fmt.Errorf("repodb: read legacy content frame: %w", err)
	}
	batch, _, err := decodeLegacyRecord(logRecord{version: locator.frameVersion, payload: payload})
	if err != nil {
		return nil, err
	}
	contents := make(map[artifact.ID][]byte, len(batch.Contents))
	for _, content := range batch.Contents {
		contents[content.Descriptor.ID] = content.Data
	}
	return contents, nil
}

func (s *Store) HasContent(ctx context.Context, id artifact.ID) (bool, error) {
	if err := contextError(ctx); err != nil {
		return false, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return false, err
	}
	_, ok := s.state.contents[id]
	return ok, nil
}

func (s *Store) ResolveAlias(ctx context.Context, name string) (artifact.ID, bool, error) {
	if err := contextError(ctx); err != nil {
		return artifact.ID{}, false, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return artifact.ID{}, false, err
	}
	id, ok := s.state.aliases[name]
	return id, ok, nil
}

func (s *Store) Parents(ctx context.Context, id artifact.ID) ([]artifact.Lineage, error) {
	return s.lineageFor(ctx, id, true)
}

func (s *Store) Children(ctx context.Context, id artifact.ID) ([]artifact.Lineage, error) {
	return s.lineageFor(ctx, id, false)
}

func (s *Store) Locations(ctx context.Context, id artifact.ID) ([]artifact.Location, error) {
	if err := contextError(ctx); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return nil, err
	}
	locations := s.state.locationsByArtifact[id]
	result := make([]artifact.Location, 0, len(locations))
	for _, location := range locations {
		result = append(result, location)
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].Kind != result[j].Kind {
			return result[i].Kind < result[j].Kind
		}
		return result[i].Value < result[j].Value
	})
	return result, nil
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
	edges := s.state.childEdges[id]
	if parents {
		edges = s.state.parentEdges[id]
	}
	result := make([]artifact.Lineage, 0, len(edges))
	for key := range edges {
		result = append(result, s.state.lineage[key])
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

func (s *Store) Head() (artifact.CommitID, uint64) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.head, s.sequence
}

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

func encodeBatch(batch artifact.Batch) ([]byte, artifact.Batch, [sha256.Size]byte, error) {
	normalized, err := normalizeBatch(batch)
	if err != nil {
		return nil, artifact.Batch{}, [sha256.Size]byte{}, err
	}
	payload, err := json.Marshal(normalized)
	if err != nil {
		return nil, artifact.Batch{}, [sha256.Size]byte{}, fmt.Errorf("repodb: encode batch: %w", err)
	}
	if len(payload) > maxFramePayload {
		return nil, artifact.Batch{}, [sha256.Size]byte{}, errors.New("repodb: batch exceeds payload limit")
	}
	return payload, normalized, sha256.Sum256(payload), nil
}

func decodeBatch(payload []byte) (artifact.Batch, [sha256.Size]byte, error) {
	var batch artifact.Batch
	if err := strictjson.DecodeBytes(payload, &batch); err != nil {
		return artifact.Batch{}, [sha256.Size]byte{}, fmt.Errorf("repodb: decode batch: %w", err)
	}
	normalized, err := normalizeBatch(batch)
	if err != nil {
		return artifact.Batch{}, [sha256.Size]byte{}, err
	}
	canonical, err := json.Marshal(normalized)
	if err != nil {
		return artifact.Batch{}, [sha256.Size]byte{}, fmt.Errorf("repodb: canonicalize batch: %w", err)
	}
	if !bytes.Equal(canonical, payload) {
		return artifact.Batch{}, [sha256.Size]byte{}, errors.New("repodb: non-canonical batch payload")
	}
	return normalized, sha256.Sum256(payload), nil
}

func encodeTransaction(request [sha256.Size]byte, delta artifact.Batch) ([]byte, map[artifact.ID]contentLocator, error) {
	contents := delta.Contents
	delta.Contents = nil
	descriptors := make([]artifact.Descriptor, len(contents))
	for index := range contents {
		descriptors[index] = contents[index].Descriptor
	}
	metadata, err := json.Marshal(persistedTransaction{Request: request, Delta: delta, Content: descriptors})
	if err != nil {
		return nil, nil, fmt.Errorf("repodb: encode transaction: %w", err)
	}
	payload := binary.LittleEndian.AppendUint32(nil, uint32(len(metadata)))
	payload = append(payload, metadata...)
	locators := make(map[artifact.ID]contentLocator, len(contents))
	for _, content := range contents {
		locators[content.Descriptor.ID] = contentLocator{offset: int64(len(payload)), size: int64(len(content.Data))}
		payload = append(payload, content.Data...)
	}
	if len(payload) > maxFramePayload {
		return nil, nil, errors.New("repodb: transaction exceeds payload limit")
	}
	return payload, locators, nil
}

func decodeRecord(record logRecord) (artifact.Batch, [sha256.Size]byte, map[artifact.ID]contentLocator, error) {
	if record.version < frameVersion {
		batch, request, err := decodeLegacyRecord(record)
		if err != nil {
			return artifact.Batch{}, [sha256.Size]byte{}, nil, err
		}
		locators := make(map[artifact.ID]contentLocator, len(batch.Contents))
		for _, content := range batch.Contents {
			locators[content.Descriptor.ID] = contentLocator{
				size: int64(len(content.Data)), frameVersion: record.version,
				payloadOffset: record.offset, payloadSize: int64(len(record.payload)),
			}
		}
		return batch, request, locators, nil
	}
	if len(record.payload) < binary.Size(uint32(0)) {
		return artifact.Batch{}, [sha256.Size]byte{}, nil, errors.New("repodb: short transaction metadata")
	}
	metadataSize := int(binary.LittleEndian.Uint32(record.payload))
	metadataOffset := binary.Size(uint32(0))
	if metadataSize > len(record.payload)-metadataOffset {
		return artifact.Batch{}, [sha256.Size]byte{}, nil, errors.New("repodb: invalid transaction metadata size")
	}
	metadata := record.payload[metadataOffset : metadataOffset+metadataSize]
	var transaction persistedTransaction
	if err := strictjson.DecodeBytes(metadata, &transaction); err != nil {
		return artifact.Batch{}, [sha256.Size]byte{}, nil, fmt.Errorf("repodb: decode transaction: %w", err)
	}
	normalized, err := normalizeBatch(transaction.Delta)
	if err != nil {
		return artifact.Batch{}, [sha256.Size]byte{}, nil, err
	}
	canonical, err := json.Marshal(persistedTransaction{Request: transaction.Request, Delta: normalized, Content: transaction.Content})
	if err != nil {
		return artifact.Batch{}, [sha256.Size]byte{}, nil, fmt.Errorf("repodb: canonicalize transaction: %w", err)
	}
	if !bytes.Equal(canonical, metadata) {
		return artifact.Batch{}, [sha256.Size]byte{}, nil, errors.New("repodb: non-canonical transaction metadata")
	}
	dataOffset := metadataOffset + metadataSize
	locators := make(map[artifact.ID]contentLocator, len(transaction.Content))
	for _, descriptor := range transaction.Content {
		end := dataOffset + int(descriptor.Size)
		if end < dataOffset || end > len(record.payload) {
			return artifact.Batch{}, [sha256.Size]byte{}, nil, errors.New("repodb: invalid transaction content bounds")
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
		return artifact.Batch{}, [sha256.Size]byte{}, nil, errors.New("repodb: trailing transaction content")
	}
	bindContentLocators(locators, record.offset, int64(len(record.payload)), record.version)
	return normalized, transaction.Request, locators, nil
}

func decodeLegacyRecord(record logRecord) (artifact.Batch, [sha256.Size]byte, error) {
	if record.version < transactionFrameVersion {
		return decodeBatch(record.payload)
	}
	var transaction struct {
		Request [sha256.Size]byte `json:"request"`
		Delta   artifact.Batch    `json:"delta"`
	}
	if err := strictjson.DecodeBytes(record.payload, &transaction); err != nil {
		return artifact.Batch{}, [sha256.Size]byte{}, fmt.Errorf("repodb: decode transaction: %w", err)
	}
	normalized, err := normalizeBatch(transaction.Delta)
	if err != nil {
		return artifact.Batch{}, [sha256.Size]byte{}, err
	}
	canonical, err := json.Marshal(transaction)
	if err != nil || !bytes.Equal(canonical, record.payload) {
		return artifact.Batch{}, [sha256.Size]byte{}, errors.New("repodb: non-canonical transaction payload")
	}
	return normalized, transaction.Request, nil
}

func bindContentLocators(locators map[artifact.ID]contentLocator, payloadOffset, payloadSize int64, version uint16) {
	for id, locator := range locators {
		locator.offset += payloadOffset
		locator.frameVersion = version
		locator.payloadOffset = payloadOffset
		locator.payloadSize = payloadSize
		locators[id] = locator
	}
}

func normalizeBatch(batch artifact.Batch) (artifact.Batch, error) {
	result := artifact.Batch{
		Key:          batch.Key,
		ExpectedHead: cloneCommitID(batch.ExpectedHead),
		Artifacts:    slices.Clone(batch.Artifacts),
		Contents:     cloneValues(batch.Contents),
		Manifests:    cloneValues(batch.Manifests),
		Lineage:      slices.Clone(batch.Lineage),
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
		return artifact.Batch{}, fmt.Errorf("repodb: duplicate content %s", duplicate.Descriptor.ID)
	}
	sort.Slice(result.Manifests, func(i, j int) bool {
		return result.Manifests[i].ID.String() < result.Manifests[j].ID.String()
	})
	if duplicate, found := adjacentDuplicate(result.Manifests, func(left, right artifact.Manifest) bool {
		return left.ID == right.ID
	}); found {
		return artifact.Batch{}, fmt.Errorf("repodb: duplicate manifest %s", duplicate.ID)
	}
	sort.Slice(result.Aliases, func(i, j int) bool {
		return result.Aliases[i].Name < result.Aliases[j].Name
	})
	if duplicate, found := adjacentDuplicate(result.Aliases, func(left, right artifact.AliasBinding) bool {
		return left.Name == right.Name
	}); found {
		return artifact.Batch{}, fmt.Errorf("repodb: duplicate alias %q", duplicate.Name)
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
		return artifact.Batch{}, fmt.Errorf("repodb: duplicate location mutation %q", duplicate.Value)
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

func cloneCommitID(value *artifact.CommitID) *artifact.CommitID {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
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
		return errors.New("repodb: nil context")
	}
	return ctx.Err()
}

var _ artifact.Repository = (*Store)(nil)
