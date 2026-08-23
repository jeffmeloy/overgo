package repodb

import (
	"bytes"
	"cmp"
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
	errPayloadLimit     = errors.New("repodb: payload limit exceeded")
	// ErrNoChange reports a batch with no effective catalog mutation.
	ErrNoChange = errors.New("repodb: batch has no effective catalog change")
)

type relationKey struct {
	child    artifact.ID
	parent   artifact.ID
	relation artifact.Relation
}

type committedBatch struct {
	key      string
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

type artifactSlot struct {
	descriptor  artifact.Descriptor
	content     contentLocator
	manifest    artifact.Manifest
	parents     []relationKey
	children    []relationKey
	locations   []artifact.Location
	sequence    uint64
	hasContent  bool
	hasManifest bool
}

type catalogState struct {
	slots       map[artifact.ID]*artifactSlot
	byMedia     map[string][]artifact.ID
	bySchema    map[string][]artifact.ID
	bySequence  []artifact.ID
	aliases     map[string]artifact.ID
	commits     []committedBatch
	commitByKey map[string]int
	edgeCount   int
}

func newCatalogState() catalogState {
	return catalogState{
		slots: map[artifact.ID]*artifactSlot{}, byMedia: map[string][]artifact.ID{},
		bySchema: map[string][]artifact.ID{}, aliases: map[string]artifact.ID{},
		commitByKey: map[string]int{},
	}
}

func (s catalogState) validate(batch artifact.Batch) error {
	added := make(map[artifact.ID]struct{}, len(batch.Artifacts))
	for _, descriptor := range batch.Artifacts {
		if slot, ok := s.slots[descriptor.ID]; ok && slot.descriptor != descriptor {
			return fmt.Errorf("%w: %s", ErrArtifactConflict, descriptor.ID)
		}
		added[descriptor.ID] = struct{}{}
	}
	for _, manifest := range batch.Manifests {
		if slot, ok := s.slots[manifest.ID]; ok && slot.hasManifest && !sameManifest(slot.manifest, manifest) {
			return fmt.Errorf("%w: manifest %s", ErrArtifactConflict, manifest.ID)
		}
	}
	for _, binding := range batch.Aliases {
		if !binding.Remove {
			if _, stored := s.slots[binding.Target]; !stored {
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
		if s.hasRelation(key) {
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
		exists := s.hasLocation(event.Location)
		if event.Action == artifact.LocationRemove && !exists {
			return fmt.Errorf("repodb: remove unknown location %q", event.Value)
		}
	}
	return nil
}

func (s catalogState) hasArtifact(id artifact.ID, added map[artifact.ID]struct{}) bool {
	if _, ok := s.slots[id]; ok {
		return true
	}
	_, ok := added[id]
	return ok
}

func (s *catalogState) apply(batch artifact.Batch, locators map[artifact.ID]contentLocator, sequence uint64) {
	for _, descriptor := range batch.Artifacts {
		s.addArtifact(descriptor, sequence)
	}
	for _, content := range batch.Contents {
		slot := s.slots[content.Descriptor.ID]
		slot.content, slot.hasContent = locators[content.Descriptor.ID], true
	}
	for _, manifest := range batch.Manifests {
		slot := s.slots[manifest.ID]
		slot.manifest, slot.hasManifest = manifest.Clone(), true
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
		if s.hasRelation(key) {
			continue
		}
		child, parent := s.slots[edge.Child], s.slots[edge.Parent]
		child.parents = insertRelation(child.parents, key)
		parent.children = insertRelation(parent.children, key)
		s.edgeCount++
	}
	for _, event := range batch.Locations {
		slot := s.slots[event.Artifact]
		index, found := slices.BinarySearchFunc(slot.locations, event.Location, compareLocation)
		if event.Action == artifact.LocationAdd {
			if !found {
				slot.locations = slices.Insert(slot.locations, index, event.Location)
			}
		} else if found {
			slot.locations = slices.Delete(slot.locations, index, index+1)
		}
	}
}

func (s *catalogState) addArtifact(descriptor artifact.Descriptor, sequence uint64) {
	if _, found := s.slots[descriptor.ID]; found {
		return
	}
	s.slots[descriptor.ID] = &artifactSlot{descriptor: descriptor, sequence: sequence}
	indexDescriptor(s.byMedia, descriptor.MediaType, descriptor.ID)
	indexDescriptor(s.bySchema, descriptor.Schema, descriptor.ID)
	s.bySequence = append(s.bySequence, descriptor.ID)
}

func (s catalogState) delta(batch artifact.Batch) artifact.Batch {
	delta := artifact.Batch{Key: batch.Key, ExpectedHead: cloneCommitID(batch.ExpectedHead)}
	for _, descriptor := range batch.Artifacts {
		if _, exists := s.slots[descriptor.ID]; !exists {
			delta.Artifacts = append(delta.Artifacts, descriptor)
		}
	}
	for _, content := range batch.Contents {
		if slot, exists := s.slots[content.Descriptor.ID]; !exists || !slot.hasContent {
			delta.Contents = append(delta.Contents, content)
		}
	}
	for _, manifest := range batch.Manifests {
		if slot, exists := s.slots[manifest.ID]; !exists || !slot.hasManifest {
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
		if !s.hasRelation(key) {
			delta.Lineage = append(delta.Lineage, edge)
		}
	}
	for _, event := range batch.Locations {
		exists := s.hasLocation(event.Location)
		if event.Action == artifact.LocationRemove || !exists {
			delta.Locations = append(delta.Locations, event)
		}
	}
	return delta
}

func indexDescriptor(index map[string][]artifact.ID, key string, id artifact.ID) {
	if key == "" {
		return
	}
	ids := index[key]
	offset, found := slices.BinarySearchFunc(ids, id, artifact.CompareID)
	if !found {
		index[key] = slices.Insert(ids, offset, id)
	}
}

func compareRelation(left, right relationKey) int {
	if order := artifact.CompareID(left.child, right.child); order != 0 {
		return order
	}
	if order := artifact.CompareID(left.parent, right.parent); order != 0 {
		return order
	}
	return cmp.Compare(left.relation, right.relation)
}

func insertRelation(edges []relationKey, key relationKey) []relationKey {
	index, found := slices.BinarySearchFunc(edges, key, compareRelation)
	if found {
		return edges
	}
	return slices.Insert(edges, index, key)
}

func (s catalogState) hasRelation(key relationKey) bool {
	slot := s.slots[key.child]
	if slot == nil {
		return false
	}
	_, found := slices.BinarySearchFunc(slot.parents, key, compareRelation)
	return found
}

func compareLocation(left, right artifact.Location) int {
	if order := cmp.Compare(left.Kind, right.Kind); order != 0 {
		return order
	}
	return cmp.Compare(left.Value, right.Value)
}

func (s catalogState) hasLocation(location artifact.Location) bool {
	slot := s.slots[location.Artifact]
	if slot == nil {
		return false
	}
	_, found := slices.BinarySearchFunc(slot.locations, location, compareLocation)
	return found
}

func sameManifest(left, right artifact.Manifest) bool {
	return left.Version == right.Version && left.ID == right.ID && slices.Equal(left.Components, right.Components)
}

func (s catalogState) commit(key string) (committedBatch, bool) {
	index, found := s.commitByKey[key]
	if !found {
		return committedBatch{}, false
	}
	return s.commits[index], true
}

func (s *catalogState) addCommit(commit committedBatch) {
	s.commitByKey[commit.key] = len(s.commits)
	s.commits = append(s.commits, commit)
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
		if slot := s.slots[current]; slot != nil {
			for _, key := range slot.parents {
				if visit(key.parent) {
					return true
				}
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
	snapshot  SnapshotReplay
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
	state, anchor, snapshot := loadLatestSnapshot(root)
	loaded := snapshot.Loaded
	if !loaded {
		state = newCatalogState()
	}
	store := &Store{state: state, readOnly: readOnly, root: root, snapshot: snapshot}
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
	batch, payloadHash, locators, err := decodeRecord(record)
	if err != nil {
		return err
	}
	if batch.ExpectedHead != nil && *batch.ExpectedHead != record.previous {
		return fmt.Errorf("%w: recorded predecessor differs", ErrHeadConflict)
	}
	if _, exists := s.state.commit(batch.Key); exists {
		return fmt.Errorf("%w: %q repeats in log", ErrBatchKeyConflict, batch.Key)
	}
	if err := s.state.validate(batch); err != nil {
		return err
	}
	s.state.apply(batch, locators, record.sequence)
	s.state.addCommit(committedBatch{key: batch.Key, id: record.id, payload: payloadHash, sequence: record.sequence})
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
	if committed, ok := s.state.commit(normalized.Key); ok {
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
	s.state.apply(delta, locators, sequence)
	s.state.addCommit(committedBatch{key: normalized.Key, id: id, payload: payloadHash, sequence: sequence})
	s.sequence = sequence
	s.head = id
	s.replayEnd = replayEnd
	return id, nil
}

func (s *Store) transactionFits(batch artifact.Batch) (bool, error) {
	_, normalized, requestHash, err := encodeBatch(batch)
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

func (s *Store) Artifact(ctx context.Context, id artifact.ID) (artifact.Descriptor, bool, error) {
	if err := contextError(ctx); err != nil {
		return artifact.Descriptor{}, false, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return artifact.Descriptor{}, false, err
	}
	slot, ok := s.state.slots[id]
	if !ok {
		return artifact.Descriptor{}, false, nil
	}
	return slot.descriptor, true, nil
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
	slot, ok := s.state.slots[id]
	if !ok || !slot.hasManifest {
		return artifact.Manifest{}, false, nil
	}
	return slot.manifest.Clone(), true, nil
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
	slot, ok := s.state.slots[id]
	if !ok || !slot.hasContent {
		return artifact.Descriptor{}, nil, false, nil
	}
	reader, err := s.log.openContent(slot.content, id)
	if err != nil {
		return artifact.Descriptor{}, nil, false, err
	}
	return slot.descriptor, reader, true, nil
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
		slot, found := s.state.slots[id]
		if !found || !slot.hasContent {
			return fmt.Errorf("repodb: content is absent: %s", id)
		}
		entries = append(entries, entry{id: id, descriptor: slot.descriptor, locator: slot.content})
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
	slot, ok := s.state.slots[id]
	return ok && slot.hasContent, nil
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
	slot := s.state.slots[id]
	if slot == nil {
		return nil, nil
	}
	return slices.Clone(slot.locations), nil
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
	slot := s.state.slots[id]
	if slot == nil {
		return nil, nil
	}
	edges := slot.children
	if parents {
		edges = slot.parents
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
		return nil, artifact.Batch{}, [sha256.Size]byte{}, fmt.Errorf("%w: batch", errPayloadLimit)
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
		return nil, nil, fmt.Errorf("%w: transaction", errPayloadLimit)
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
