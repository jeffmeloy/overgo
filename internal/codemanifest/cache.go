package codemanifest

import (
	"errors"
	"reflect"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/repoanalysis"
)

// CacheKey binds every input that can change generated manifest semantics.
type CacheKey struct {
	SourceIdentity string          `json:"source_identity"`
	Analyzer       Analyzer        `json:"analyzer"`
	Schema         string          `json:"schema"`
	BuildContexts  []BuildContext  `json:"build_contexts"`
	ExternalInputs []ExternalInput `json:"external_inputs,omitempty"`
}

type cacheEntry struct {
	key      artifact.ID
	manifest Manifest
	used     uint64
}

// Cache owns a caller-declared capacity; Put evicts the least-recently used
// entry and uses content identity as the deterministic tie-breaker.
type Cache struct {
	capacity int
	clock    uint64
	entries  map[artifact.ID]cacheEntry
}

// NewCache creates a manifest cache with explicit owned capacity.
func NewCache(capacity int) (*Cache, error) {
	if capacity <= 0 {
		return nil, errors.New("code manifest cache: capacity must be positive")
	}
	return &Cache{capacity: capacity, entries: map[artifact.ID]cacheEntry{}}, nil
}

// Key returns the canonical identity for one complete manifest authority.
func (key CacheKey) id() (artifact.ID, error) {
	canonical := key
	canonical.BuildContexts = slices.Clone(key.BuildContexts)
	canonical.ExternalInputs = slices.Clone(key.ExternalInputs)
	slices.SortFunc(canonical.BuildContexts, func(left, right BuildContext) int { return strings.Compare(left.ID, right.ID) })
	slices.SortFunc(canonical.ExternalInputs, func(left, right ExternalInput) int { return strings.Compare(left.Path, right.Path) })
	if !validDigest(canonical.SourceIdentity) || canonical.Analyzer.Name == "" || canonical.Analyzer.Version == "" || canonical.Schema == "" || len(canonical.BuildContexts) == 0 {
		return artifact.ID{}, errors.New("code manifest cache: incomplete authority key")
	}
	return artifact.JSONID(artifact.KindRecipe, canonical)
}

func (cache *Cache) get(key CacheKey) (Manifest, bool) {
	if cache == nil {
		return Manifest{}, false
	}
	id, err := key.id()
	if err != nil {
		return Manifest{}, false
	}
	entry, found := cache.entries[id]
	if !found {
		return Manifest{}, false
	}
	cache.clock++
	entry.used = cache.clock
	cache.entries[id] = entry
	return clone(entry.manifest), true
}

func (cache *Cache) put(key CacheKey, manifest Manifest) error {
	if cache == nil || cache.capacity <= 0 {
		return errors.New("code manifest cache: uninitialized")
	}
	id, err := key.id()
	if err != nil {
		return err
	}
	if err := manifest.Validate(); err != nil {
		return err
	}
	if manifest.SourceIdentity != key.SourceIdentity || manifest.Analyzer != key.Analyzer || key.Schema != Schema ||
		!reflect.DeepEqual(manifest.BuildContexts, key.BuildContexts) || !slices.Equal(manifest.ExternalInputs, key.ExternalInputs) {
		return errors.New("code manifest cache: manifest differs from authority key")
	}
	cache.clock++
	cache.entries[id] = cacheEntry{key: id, manifest: clone(manifest), used: cache.clock}
	for len(cache.entries) > cache.capacity {
		var victim cacheEntry
		first := true
		for _, candidate := range cache.entries {
			if first || candidate.used < victim.used || candidate.used == victim.used && candidate.key.String() < victim.key.String() {
				victim, first = candidate, false
			}
		}
		delete(cache.entries, victim.key)
	}
	return nil
}

// Generate reuses an exact authority match or generates and records a new
// manifest. Reused is false on every incomplete or changed authority.
func (cache *Cache) Generate(snapshot repoanalysis.SourceSnapshot, selections []repoanalysis.BuildSelection, external []ExternalInput) (manifest Manifest, reused bool, err error) {
	contexts := make([]BuildContext, 0, len(selections))
	for _, selection := range selections {
		context, contextErr := buildContext(selection)
		if contextErr != nil {
			return Manifest{}, false, contextErr
		}
		contexts = append(contexts, context)
	}
	slices.SortFunc(contexts, func(left, right BuildContext) int { return strings.Compare(left.ID, right.ID) })
	inputs := slices.Clone(external)
	slices.SortFunc(inputs, func(left, right ExternalInput) int { return strings.Compare(left.Path, right.Path) })
	key := CacheKey{
		SourceIdentity: snapshot.Identity(), Analyzer: Analyzer{Name: analyzerName, Version: analyzerVersion},
		Schema: Schema, BuildContexts: contexts, ExternalInputs: inputs,
	}
	if manifest, found := cache.get(key); found {
		return manifest, true, nil
	}
	manifest, err = Generate(snapshot, selections, external)
	if err != nil {
		return Manifest{}, false, err
	}
	if err := cache.put(key, manifest); err != nil {
		return Manifest{}, false, err
	}
	return manifest, false, nil
}
