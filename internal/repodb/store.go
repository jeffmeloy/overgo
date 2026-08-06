package repodb

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"sort"
	"sync"

	"llamacpp2go/internal/artifact"
)

var (
	ErrClosed           = errors.New("repodb: store is closed")
	ErrReadOnly         = errors.New("repodb: store is read-only")
	ErrBatchKeyConflict = errors.New("repodb: batch key already names different content")
	ErrArtifactConflict = errors.New("repodb: artifact identity has conflicting facts")
	ErrAliasConflict    = errors.New("repodb: alias compare-and-set failed")
	ErrLineageCycle     = errors.New("repodb: lineage cycle")
	ErrStoreFaulted     = errors.New("repodb: store requires reopen after uncertain append")
	ErrSnapshotAnchor   = errors.New("repodb: snapshot anchor is absent from commit chain")
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

type catalogState struct {
	artifacts           map[artifact.ID]artifact.Descriptor
	contents            map[artifact.ID]artifact.Content
	manifests           map[artifact.ID]artifact.Manifest
	aliases             map[string]artifact.ID
	lineage             map[relationKey]artifact.Lineage
	parentEdges         map[artifact.ID]map[relationKey]artifact.Lineage
	childEdges          map[artifact.ID]map[relationKey]artifact.Lineage
	parents             map[artifact.ID]map[artifact.ID]struct{}
	locations           map[locationKey]artifact.Location
	locationsByArtifact map[artifact.ID]map[locationKey]artifact.Location
	commits             map[string]committedBatch
}

func newCatalogState() catalogState {
	return catalogState{
		artifacts:           map[artifact.ID]artifact.Descriptor{},
		contents:            map[artifact.ID]artifact.Content{},
		manifests:           map[artifact.ID]artifact.Manifest{},
		aliases:             map[string]artifact.ID{},
		lineage:             map[relationKey]artifact.Lineage{},
		parentEdges:         map[artifact.ID]map[relationKey]artifact.Lineage{},
		childEdges:          map[artifact.ID]map[relationKey]artifact.Lineage{},
		parents:             map[artifact.ID]map[artifact.ID]struct{}{},
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
	for _, content := range batch.Contents {
		if current, ok := s.contents[content.Descriptor.ID]; ok && !sameContent(current, content) {
			return fmt.Errorf("%w: content %s", ErrArtifactConflict, content.Descriptor.ID)
		}
	}
	for _, binding := range batch.Aliases {
		if _, stored := s.artifacts[binding.Target]; !stored {
			if _, pending := added[binding.Target]; !pending {
				return fmt.Errorf("repodb: alias %q targets unknown artifact %s", binding.Name, binding.Target)
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

func (s *catalogState) apply(batch artifact.Batch) {
	for _, descriptor := range batch.Artifacts {
		s.artifacts[descriptor.ID] = descriptor
	}
	for _, content := range batch.Contents {
		s.contents[content.Descriptor.ID] = content.Clone()
	}
	for _, manifest := range batch.Manifests {
		s.manifests[manifest.ID] = manifest.Clone()
	}
	for _, binding := range batch.Aliases {
		s.aliases[binding.Name] = binding.Target
	}
	for _, edge := range batch.Lineage {
		key := relationKey{child: edge.Child, parent: edge.Parent, relation: edge.Relation}
		if _, exists := s.lineage[key]; exists {
			continue
		}
		s.lineage[key] = edge
		indexRelation(s.parentEdges, edge.Child, key, edge)
		indexRelation(s.childEdges, edge.Parent, key, edge)
		parents := s.parents[edge.Child]
		if parents == nil {
			parents = map[artifact.ID]struct{}{}
			s.parents[edge.Child] = parents
		}
		parents[edge.Parent] = struct{}{}
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

func indexRelation(
	index map[artifact.ID]map[relationKey]artifact.Lineage,
	id artifact.ID,
	key relationKey,
	edge artifact.Lineage,
) {
	edges := index[id]
	if edges == nil {
		edges = map[relationKey]artifact.Lineage{}
		index[id] = edges
	}
	edges[key] = edge
}

func sameManifest(left, right artifact.Manifest) bool {
	return left.Version == right.Version && left.ID == right.ID && slices.Equal(left.Components, right.Components)
}

func sameContent(left, right artifact.Content) bool {
	return left.Descriptor == right.Descriptor && bytes.Equal(left.Data, right.Data)
}

func (s catalogState) reaches(start, target artifact.ID, pending map[artifact.ID]map[artifact.ID]struct{}) bool {
	if start == target {
		return true
	}
	seen := map[artifact.ID]struct{}{start: {}}
	queue := []artifact.ID{start}
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		for _, parents := range []map[artifact.ID]struct{}{s.parents[current], pending[current]} {
			for parent := range parents {
				if parent == target {
					return true
				}
				if _, ok := seen[parent]; ok {
					continue
				}
				seen[parent] = struct{}{}
				queue = append(queue, parent)
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
	recovered bool
	readOnly  bool
	closed    bool
	fault     error
	root      string
	snapshot  replayAnchor
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
	store := &Store{state: state, readOnly: readOnly, root: root, snapshot: anchor}
	apply := func(record logRecord) error {
		batch, payloadHash, err := decodeBatch(record.payload)
		if err != nil {
			return err
		}
		if _, exists := store.state.commits[batch.Key]; exists {
			return fmt.Errorf("%w: %q repeats in log", ErrBatchKeyConflict, batch.Key)
		}
		if err := store.state.validate(batch); err != nil {
			return err
		}
		store.state.apply(batch)
		store.state.commits[batch.Key] = committedBatch{
			id: record.id, payload: payloadHash, sequence: record.sequence,
		}
		return nil
	}
	log, replay, err := openRecordLog(root, readOnly, anchor, apply)
	if errors.Is(err, ErrSnapshotAnchor) && loaded {
		store.state = newCatalogState()
		store.snapshot = replayAnchor{}
		log, replay, err = openRecordLog(root, readOnly, replayAnchor{}, apply)
	}
	if err != nil {
		return nil, err
	}
	store.log = log
	store.head = replay.head
	store.sequence = replay.sequence
	store.recovered = replay.recovered
	return store, nil
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
	if err := s.state.validate(normalized); err != nil {
		return artifact.CommitID{}, err
	}
	sequence := s.sequence + 1
	id, err := s.log.append(sequence, s.head, payload)
	if err != nil {
		s.fault = err
		return id, fmt.Errorf("%w: %v", ErrStoreFaulted, err)
	}
	s.state.apply(normalized)
	s.state.commits[normalized.Key] = committedBatch{id: id, payload: payloadHash, sequence: sequence}
	s.sequence = sequence
	s.head = id
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

func (s *Store) Content(ctx context.Context, id artifact.ID) (artifact.Content, bool, error) {
	if err := contextError(ctx); err != nil {
		return artifact.Content{}, false, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return artifact.Content{}, false, err
	}
	value, ok := s.state.contents[id]
	return value.Clone(), ok, nil
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
	for _, edge := range edges {
		result = append(result, edge)
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

func (s *Store) Recovered() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.recovered
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
		return fmt.Errorf("%w: %v", ErrStoreFaulted, s.fault)
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
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	var batch artifact.Batch
	if err := decoder.Decode(&batch); err != nil {
		return artifact.Batch{}, [sha256.Size]byte{}, fmt.Errorf("repodb: decode batch: %w", err)
	}
	if err := ensureJSONEnd(decoder); err != nil {
		return artifact.Batch{}, [sha256.Size]byte{}, err
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

func ensureJSONEnd(decoder *json.Decoder) error {
	var extra any
	err := decoder.Decode(&extra)
	if errors.Is(err, io.EOF) {
		return nil
	}
	if err == nil {
		return errors.New("repodb: batch has trailing JSON value")
	}
	return fmt.Errorf("repodb: decode batch tail: %w", err)
}

func normalizeBatch(batch artifact.Batch) (artifact.Batch, error) {
	result := artifact.Batch{
		Key:       batch.Key,
		Artifacts: slices.Clone(batch.Artifacts),
		Contents:  cloneContents(batch.Contents),
		Manifests: cloneManifests(batch.Manifests),
		Lineage:   slices.Clone(batch.Lineage),
		Aliases:   artifact.CloneAliasBindings(batch.Aliases),
		Locations: slices.Clone(batch.Locations),
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
	for _, descriptor := range result.Artifacts {
		if len(artifacts) > 0 && artifacts[len(artifacts)-1].ID == descriptor.ID {
			if artifacts[len(artifacts)-1] != descriptor {
				return artifact.Batch{}, fmt.Errorf("%w: %s", ErrArtifactConflict, descriptor.ID)
			}
			continue
		}
		artifacts = append(artifacts, descriptor)
	}
	result.Artifacts = artifacts
	sort.Slice(result.Contents, func(i, j int) bool {
		return result.Contents[i].Descriptor.ID.String() < result.Contents[j].Descriptor.ID.String()
	})
	for index := 1; index < len(result.Contents); index++ {
		if result.Contents[index-1].Descriptor.ID == result.Contents[index].Descriptor.ID {
			return artifact.Batch{}, fmt.Errorf("repodb: duplicate content %s", result.Contents[index].Descriptor.ID)
		}
	}
	sort.Slice(result.Manifests, func(i, j int) bool {
		return result.Manifests[i].ID.String() < result.Manifests[j].ID.String()
	})
	for index := 1; index < len(result.Manifests); index++ {
		if result.Manifests[index-1].ID == result.Manifests[index].ID {
			return artifact.Batch{}, fmt.Errorf("repodb: duplicate manifest %s", result.Manifests[index].ID)
		}
	}
	sort.Slice(result.Aliases, func(i, j int) bool {
		return result.Aliases[i].Name < result.Aliases[j].Name
	})
	for index := 1; index < len(result.Aliases); index++ {
		if result.Aliases[index-1].Name == result.Aliases[index].Name {
			return artifact.Batch{}, fmt.Errorf("repodb: duplicate alias %q", result.Aliases[index].Name)
		}
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
	for _, edge := range result.Lineage {
		if len(lineage) > 0 && lineage[len(lineage)-1] == edge {
			continue
		}
		lineage = append(lineage, edge)
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
	for index := 1; index < len(result.Locations); index++ {
		left, right := result.Locations[index-1], result.Locations[index]
		if left.Artifact == right.Artifact && left.Kind == right.Kind && left.Value == right.Value {
			return artifact.Batch{}, fmt.Errorf("repodb: duplicate location mutation %q", right.Value)
		}
	}
	return result, nil
}

func cloneManifests(manifests []artifact.Manifest) []artifact.Manifest {
	result := slices.Clone(manifests)
	for index := range result {
		result[index] = result[index].Clone()
	}
	return result
}

func cloneContents(contents []artifact.Content) []artifact.Content {
	result := slices.Clone(contents)
	for index := range result {
		result[index] = result[index].Clone()
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
