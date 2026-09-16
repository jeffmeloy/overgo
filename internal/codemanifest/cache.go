package codemanifest

import (
	"errors"
	"reflect"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/codeprofile"
	"overgo/internal/gosource"
	"overgo/internal/repoanalysis"
)

// cacheKey binds every input that can change generated manifest semantics.
type cacheKey struct {
	SelectionIdentity string          `json:"selection_identity"`
	SourceIdentity    string          `json:"source_identity"`
	Analyzer          Analyzer        `json:"analyzer"`
	Schema            string          `json:"schema"`
	BuildContexts     []BuildContext  `json:"build_contexts"`
	ExternalInputs    []ExternalInput `json:"external_inputs,omitempty"`
}

type cacheEntry struct {
	selectionIdentity string
	key               artifact.ID
	manifest          Manifest
	used              uint64
	profileSource     artifact.ID
	profile           *codeprofile.Profile
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
func (key cacheKey) canonical() (cacheKey, error) {
	canonical := key
	canonical.BuildContexts = slices.Clone(key.BuildContexts)
	canonical.ExternalInputs = slices.Clone(key.ExternalInputs)
	slices.SortFunc(canonical.BuildContexts, func(left, right BuildContext) int { return strings.Compare(left.ID, right.ID) })
	slices.SortFunc(canonical.ExternalInputs, func(left, right ExternalInput) int { return strings.Compare(left.Path, right.Path) })
	if !validDigest(canonical.SelectionIdentity) || !validDigest(canonical.SourceIdentity) || canonical.Analyzer.Name == "" || canonical.Analyzer.Version == "" || canonical.Schema == "" || len(canonical.BuildContexts) == 0 {
		return cacheKey{}, errors.New("code manifest cache: incomplete authority key")
	}
	return canonical, nil
}

func (key cacheKey) id() (artifact.ID, error) {
	canonical, err := key.canonical()
	if err != nil {
		return artifact.ID{}, err
	}
	return artifact.JSONID(artifact.KindRecipe, canonical)
}

func (cache *Cache) get(key cacheKey) (Manifest, bool) {
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

func (cache *Cache) put(key cacheKey, manifest Manifest) error {
	if cache == nil || cache.capacity <= 0 {
		return errors.New("code manifest cache: uninitialized")
	}
	canonical, err := key.canonical()
	if err != nil {
		return err
	}
	id, err := canonical.id()
	if err != nil {
		return err
	}
	if err := manifest.Validate(); err != nil {
		return err
	}
	if manifest.SourceIdentity != canonical.SourceIdentity || manifest.Analyzer != canonical.Analyzer || canonical.Schema != Schema ||
		!reflect.DeepEqual(manifest.BuildContexts, canonical.BuildContexts) || !slices.Equal(manifest.ExternalInputs, canonical.ExternalInputs) {
		return errors.New("code manifest cache: manifest differs from authority key")
	}
	cache.clock++
	cache.entries[id] = cacheEntry{key: id, selectionIdentity: canonical.SelectionIdentity, manifest: clone(manifest), used: cache.clock}
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
func (cache *Cache) Generate(snapshot repoanalysis.SourceSnapshot, selections []gosource.BuildSelection, external []ExternalInput) (manifest Manifest, reused bool, err error) {
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
	type analysisSelection struct {
		Selection   gosource.BuildSelection
		ImportNames map[string]string
	}
	selected := make([]analysisSelection, len(selections))
	for index, selection := range selections {
		names, err := repoanalysis.PackageNames(snapshot, selection)
		if err != nil {
			return Manifest{}, false, err
		}
		selection.Root = ""
		selected[index] = analysisSelection{selection, names}
	}
	slices.SortFunc(selected, func(left, right analysisSelection) int {
		return strings.Compare(left.Selection.Context, right.Selection.Context)
	})
	selectionID, err := artifact.JSONID(artifact.KindRecipe, struct {
		Selections []analysisSelection
		Files      []repoanalysis.GoFile
	}{selected, snapshot.Files})
	if err != nil {
		return Manifest{}, false, err
	}
	key := cacheKey{
		SelectionIdentity: selectionID.DigestHex(),
		SourceIdentity:    snapshot.Identity(), Analyzer: Analyzer{Name: analyzerName, Version: analyzerVersion},
		Schema: Schema, BuildContexts: contexts, ExternalInputs: inputs,
	}
	if manifest, found := cache.get(key); found {
		return manifest, true, nil
	}
	// Source, compiler membership and resolved import names bind analysis.
	if cache != nil {
		for _, entry := range cache.entries {
			prior := entry.manifest
			if entry.selectionIdentity != key.SelectionIdentity || prior.SourceIdentity != key.SourceIdentity || prior.Analyzer != key.Analyzer || !reflect.DeepEqual(prior.BuildContexts, contexts) {
				continue
			}
			manifest = clone(prior)
			manifest.ExternalInputs = inputs
			manifest, err = identifyManifest(manifest)
			if err != nil {
				return Manifest{}, false, err
			}
			if err := cache.put(key, manifest); err != nil {
				return Manifest{}, false, err
			}
			cache.attachProfile(key, entry.profileSource, entry.profile)
			return manifest, false, nil
		}
	}
	profile, found := cache.Profile(snapshot)
	if !found {
		profile, err = codeprofile.Build(snapshot)
		if err != nil {
			return Manifest{}, false, err
		}
	}
	manifest, err = generate(snapshot, profile, selections, external)
	if err != nil {
		return Manifest{}, false, err
	}
	if err := cache.put(key, manifest); err != nil {
		return Manifest{}, false, err
	}
	source, err := artifact.JSONID(artifact.KindProfile, snapshot.Files)
	if err != nil {
		return Manifest{}, false, err
	}
	cache.attachProfile(key, source, &profile)
	return manifest, false, nil
}

func (cache *Cache) attachProfile(key cacheKey, source artifact.ID, profile *codeprofile.Profile) {
	id, err := key.id()
	if err != nil {
		return
	}
	entry := cache.entries[id]
	entry.profileSource, entry.profile = source, profile
	cache.entries[id] = entry
}

// Profile returns a copy of facts already built for an identical snapshot.
// Facts share the manifest entry's lifetime and eviction; policy is not cached.
// Files bind content identity, order, path and test classification. Analyzer
// and toolchain identity are fixed for this in-process cache.
func (cache *Cache) Profile(snapshot repoanalysis.SourceSnapshot) (codeprofile.Profile, bool) {
	if cache == nil {
		return codeprofile.Profile{}, false
	}
	source, err := artifact.JSONID(artifact.KindProfile, snapshot.Files)
	if err != nil {
		return codeprofile.Profile{}, false
	}
	for _, entry := range cache.entries {
		if entry.profile == nil || entry.profileSource != source {
			continue
		}
		profile := *entry.profile
		profile.Functions = slices.Clone(profile.Functions)
		profile.Clones = slices.Clone(profile.Clones)
		for i := range profile.Clones {
			profile.Clones[i].Functions = slices.Clone(profile.Clones[i].Functions)
		}
		return profile, true
	}
	return codeprofile.Profile{}, false
}
