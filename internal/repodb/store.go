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
)

type relationKey struct {
	child    artifact.ID
	parent   artifact.ID
	relation artifact.Relation
}

type committedBatch struct {
	id      artifact.CommitID
	payload [sha256.Size]byte
}

type catalogState struct {
	artifacts map[artifact.ID]artifact.Descriptor
	aliases   map[string]artifact.ID
	lineage   map[relationKey]artifact.Lineage
	parents   map[artifact.ID]map[artifact.ID]struct{}
	commits   map[string]committedBatch
}

func newCatalogState() catalogState {
	return catalogState{
		artifacts: map[artifact.ID]artifact.Descriptor{},
		aliases:   map[string]artifact.ID{},
		lineage:   map[relationKey]artifact.Lineage{},
		parents:   map[artifact.ID]map[artifact.ID]struct{}{},
		commits:   map[string]committedBatch{},
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
	for _, binding := range batch.Aliases {
		s.aliases[binding.Name] = binding.Target
	}
	for _, edge := range batch.Lineage {
		key := relationKey{child: edge.Child, parent: edge.Parent, relation: edge.Relation}
		if _, exists := s.lineage[key]; exists {
			continue
		}
		s.lineage[key] = edge
		parents := s.parents[edge.Child]
		if parents == nil {
			parents = map[artifact.ID]struct{}{}
			s.parents[edge.Child] = parents
		}
		parents[edge.Parent] = struct{}{}
	}
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
	store := &Store{state: newCatalogState(), readOnly: readOnly}
	log, replay, err := openRecordLog(root, readOnly, func(record logRecord) error {
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
		store.state.commits[batch.Key] = committedBatch{id: record.id, payload: payloadHash}
		return nil
	})
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
	s.state.commits[normalized.Key] = committedBatch{id: id, payload: payloadHash}
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

func (s *Store) lineageFor(ctx context.Context, id artifact.ID, parents bool) ([]artifact.Lineage, error) {
	if err := contextError(ctx); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return nil, err
	}
	result := make([]artifact.Lineage, 0)
	for _, edge := range s.state.lineage {
		if parents && edge.Child == id || !parents && edge.Parent == id {
			result = append(result, edge)
		}
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
		Lineage:   slices.Clone(batch.Lineage),
		Aliases:   cloneAliases(batch.Aliases),
	}
	if err := result.Validate(); err != nil {
		return artifact.Batch{}, err
	}
	sort.Slice(result.Artifacts, func(i, j int) bool {
		return result.Artifacts[i].ID.String() < result.Artifacts[j].ID.String()
	})
	for index := 1; index < len(result.Artifacts); index++ {
		if result.Artifacts[index-1].ID == result.Artifacts[index].ID {
			return artifact.Batch{}, fmt.Errorf("repodb: duplicate artifact %s", result.Artifacts[index].ID)
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
	for index := 1; index < len(result.Lineage); index++ {
		if result.Lineage[index-1] == result.Lineage[index] {
			return artifact.Batch{}, fmt.Errorf("repodb: duplicate lineage edge")
		}
	}
	return result, nil
}

func cloneAliases(bindings []artifact.AliasBinding) []artifact.AliasBinding {
	result := slices.Clone(bindings)
	for index := range result {
		if result[index].Previous == nil {
			continue
		}
		previous := *result[index].Previous
		result[index].Previous = &previous
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
