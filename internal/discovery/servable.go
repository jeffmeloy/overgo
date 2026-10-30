// Package discovery owns the servable predicate (floor component 9): a
// servable model is a model manifest with an ACTIVE inference recipe whose
// bytes are PRESENT at a recorded location. The smoke lane, the discovery
// query, and the evidence tier all consume this one predicate, so discovery
// cannot drift from recipe truth.
package discovery

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/gguf"
	// A model registered from its own files opens as the view the converter registers with gguf.
	_ "overgo/internal/hfconvert"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/remoteprovider"
)

type Entry struct {
	Model    artifact.ID
	Recipe   artifact.ID
	Tier     recipe.EvidenceTier
	Location string
	Present  bool
	// Stale carries the activation defect when the active recipe cannot be
	// trusted. A stale entry is reported, never served: one broken
	// activation must not blind discovery to every healthy model.
	Stale string
}

// Servable lists models with active inference recipes; presence is a stat of
// recorded locations (manifest first, then components; recorded-but-missing
// or unrecorded reports Present=false).
func Servable(ctx context.Context, store *overgodb.Store, limit int) ([]Entry, error) {
	return ServableWithMemo(ctx, store, limit, nil)
}

// ServableWithMemo is Servable with digest reuse for interactive callers: a
// nil memo hashes every file fresh, exactly as Servable always has.
// Exact model identities or locations filter metadata before any file hashing
// or active-recipe resolution. An empty selection retains full-catalog discovery.
func ServableWithMemo(ctx context.Context, store *overgodb.Store, limit int, memo *Memo, selectedInputs ...string) ([]Entry, error) {
	result, err := store.Query(ctx, overgodb.Query{
		Kind: artifact.KindModel, MaxResults: limit, Projection: overgodb.ProjectManifests,
	})
	if err != nil {
		return nil, err
	}
	if result.Truncated {
		return nil, fmt.Errorf("discovery: model catalog exceeds listing bound %d", limit)
	}
	var entries []Entry
	identities := map[string]fileIdentity{}
	for _, manifest := range result.Manifests {
		selected, err := matchesSelectedInput(ctx, store, manifest, selectedInputs)
		if err != nil {
			return nil, err
		}
		if !selected {
			continue
		}
		declared, err := modelrecipe.HasActiveRecipe(ctx, store, manifest.ID, recipe.TaskInference)
		if err != nil {
			return nil, err
		}
		if !declared {
			continue
		}
		location, present := presence(ctx, store, manifest, identities, memo)
		if !present {
			continue
		}
		activation, active, err := modelrecipe.ActiveRecord(ctx, store, manifest.ID, recipe.TaskInference)
		if err != nil {
			entries = append(entries, Entry{
				Model: manifest.ID, Location: location, Present: present,
				Stale: fmt.Sprintf("activation cannot be trusted: %v", err),
			})
			continue
		}
		if !active {
			continue
		}
		entry := Entry{
			Model: manifest.ID, Recipe: activation.Definition.ID, Tier: activation.Tier,
			Location: location, Present: present,
		}
		// The loader refuses an inference recipe without its runtime
		// policy, so the servable listing must report it stale exactly as
		// the capability catalog does -- eval and swap targets derived
		// from this listing never include a model the loader refuses.
		if _, err := modelrecipe.ResolveRuntimePolicy(ctx, store, activation.Definition); err != nil {
			entry.Stale = fmt.Sprintf("not servable: %v (recipe policy <model> binds it)", err)
		}
		// A remote model lists with its provider's refusal: the key its
		// declaration names is absent from the environment.
		if provider, remote, err := remoteprovider.Resolve(ctx, store, manifest); err != nil {
			entry.Stale = fmt.Sprintf("not servable: %v", err)
		} else if refusal := remoteprovider.Refusal(provider); remote && refusal != "" {
			entry.Stale = "not servable: " + refusal
		}
		entries = append(entries, entry)
	}
	return entries, nil
}

func matchesSelectedInput(ctx context.Context, store artifact.Reader, manifest artifact.Manifest, selected []string) (bool, error) {
	if len(selected) == 0 {
		return true, nil
	}
	if slices.Contains(selected, manifest.ID.String()) {
		return true, nil
	}
	for _, component := range manifest.Components {
		locations, err := store.Locations(ctx, component.Artifact)
		if err != nil {
			return false, err
		}
		for _, location := range locations {
			if location.Kind != artifact.LocationFile {
				continue
			}
			for _, path := range selected {
				if strings.EqualFold(filepath.ToSlash(filepath.Clean(location.Value)), filepath.ToSlash(filepath.Clean(path))) {
					return true, nil
				}
			}
		}
	}
	return false, nil
}

type fileIdentity struct {
	id      artifact.ID
	size    uint64
	present bool
}

// presence requires recorded bytes, not a path that now names replacement bytes.
func presence(
	ctx context.Context,
	store *overgodb.Store,
	manifest artifact.Manifest,
	identities map[string]fileIdentity,
	memo *Memo,
) (string, bool) {
	recorded, servingLocation := "", ""
	for _, component := range manifest.Components {
		descriptor, ok, err := store.Artifact(ctx, component.Artifact)
		if err != nil || !ok {
			return recorded, false
		}
		locations, err := store.Locations(ctx, component.Artifact)
		if err != nil {
			return recorded, false
		}
		found, componentRecorded := "", ""
		for _, location := range locations {
			// A remote model's bytes live with its provider: the remote
			// location is its serving location and its presence.
			if location.Kind == artifact.LocationRemote {
				return location.Value, true
			}
			if location.Kind != artifact.LocationFile {
				continue
			}
			componentRecorded = cmp.Or(componentRecorded, location.Value)
			if holds(location.Value, descriptor, identities, memo) {
				found = location.Value
				break
			}
		}
		recorded = cmp.Or(recorded, componentRecorded)
		if found == "" && componentRecorded != "" {
			found = moved(componentRecorded, descriptor, identities, memo)
		}
		// A component with no file is a model served from its own files: present
		// where the directory it reads from serves its bytes (identifyDirectory).
		for _, location := range locations {
			if found == "" && componentRecorded == "" && location.Kind == artifact.LocationDirectory {
				recorded = cmp.Or(recorded, location.Value)
				if holds(location.Value, descriptor, identities, memo) {
					found = location.Value
				}
			}
		}
		if found == "" {
			return recorded, false
		}
		servingLocation = cmp.Or(servingLocation, found)
	}
	if servingLocation != "" {
		return servingLocation, true
	}
	if recorded != "" {
		return recorded, false
	}
	locations, _ := store.Locations(ctx, manifest.ID)
	for _, location := range locations {
		if location.Kind == artifact.LocationDirectory {
			return location.Value, false
		}
	}
	return "", false
}

func holds(path string, descriptor artifact.Descriptor, identities map[string]fileIdentity, memo *Memo) bool {
	identity := identifyLocation(path, descriptor.ID.Kind(), identities, memo)
	return identity.present && identity.id == descriptor.ID && identity.size == descriptor.Size
}

// moved finds bytes whose recorded file is gone by their identity: the
// recorded path's trailing components under each model root, longest first,
// so a model directory moved between roots keeps resolving.
func moved(recorded string, descriptor artifact.Descriptor, identities map[string]fileIdentity, memo *Memo) string {
	if _, err := os.Stat(recorded); err == nil {
		return ""
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return ""
	}
	parts := strings.Split(filepath.ToSlash(filepath.Clean(recorded)), "/")
	for first := range parts {
		tail := filepath.Join(parts[first:]...)
		for _, root := range roots.ModelRoots() {
			candidate := filepath.Join(root, tail)
			if holds(candidate, descriptor, identities, memo) {
				return candidate
			}
		}
	}
	return ""
}

// WeightsIdentity is the model identity of the weights file at path,
// the digest a capability claim binds to (catalogs identify manifests;
// claims identify the bytes they measured, exactly as the benchmark
// publisher does), through the memo so a file hashed for the listing
// is not hashed again. A path that does not name present bytes returns
// an error.
func WeightsIdentity(path string, memo *Memo) (artifact.ID, error) {
	identity := identifyLocation(path, artifact.KindTensorSet, map[string]fileIdentity{}, memo)
	if !identity.present {
		return artifact.ID{}, fmt.Errorf("discovery: %s does not name present weights", path)
	}
	// Artifact kinds share the byte digest. Reuse discovery's stat-validated
	// tensor identity and retain the model kind required by verification claims.
	_, digest, _ := strings.Cut(identity.id.String(), ":")
	return artifact.ParseID(artifact.KindModel.String() + ":" + digest)
}

// FileIdentity names a file's bytes as kind through the memo, hashing them only when the memo holds no identity for
// the file's present size and modification time; false where path names no readable file.
func FileIdentity(path string, kind artifact.Kind, memo *Memo) (artifact.ID, bool) {
	identity := identifyLocation(path, kind, map[string]fileIdentity{}, memo)
	return identity.id, identity.present
}

func identifyLocation(path string, kind artifact.Kind, identities map[string]fileIdentity, memo *Memo) fileIdentity {
	key := path + "\x00" + kind.String()
	if identity, ok := identities[key]; ok {
		return identity
	}
	identity := fileIdentity{}
	before, err := os.Stat(path)
	if err == nil && before.IsDir() {
		identity = identifyDirectory(path, key, kind, memo)
		identities[key] = identity
		return identity
	}
	if err != nil {
		identities[key] = identity
		return identity
	}
	if remembered, ok := memo.lookup(key, before); ok {
		identities[key] = remembered
		return remembered
	}
	file, err := os.Open(path)
	if err != nil {
		identities[key] = identity
		return identity
	}
	id, size, identifyErr := artifact.Identify(kind, file)
	closeErr := file.Close()
	after, statErr := os.Stat(path)
	if errors.Join(identifyErr, closeErr, statErr) == nil &&
		before.Size() == after.Size() && before.ModTime().Equal(after.ModTime()) && size == uint64(after.Size()) {
		identity = fileIdentity{id: id, size: size, present: true}
	}
	if identity.present {
		memo.record(key, after, identity)
	}
	identities[key] = identity
	return identity
}

// identifyDirectory names a model directory as the image it serves as a GGUF
// (gguf.Open's view of it), through the memo under the signature of the files
// it holds (directorySignature), hashing the image only when they changed. A
// directory gguf.Open does not serve as a view names nothing.
func identifyDirectory(path, key string, kind artifact.Kind, memo *Memo) fileIdentity {
	before, err := directorySignature(path)
	if err != nil {
		return fileIdentity{}
	}
	if remembered, ok := memo.lookup(key, before); ok {
		return remembered
	}
	file, err := gguf.Open(path)
	if err != nil {
		return fileIdentity{}
	}
	shard := file.Shards()[0]
	id, size, identifyErr := artifact.Identify(kind, io.NewSectionReader(shard.Source, 0, int64(shard.Size)))
	viewed := file.ViewDirectory() != ""
	closeErr := file.Close()
	after, statErr := directorySignature(path)
	if !viewed || errors.Join(identifyErr, closeErr, statErr) != nil || before.Size() != after.Size() || !before.ModTime().Equal(after.ModTime()) || size != shard.Size {
		return fileIdentity{}
	}
	identity := fileIdentity{id: id, size: size, present: true}
	memo.record(key, after, identity)
	return identity
}

// directorySignature stats a directory as its files: their sizes summed and
// the newest change among them, so a change to any file the directory holds
// changes it, as a file's size and time do for a file.
func directorySignature(path string) (os.FileInfo, error) {
	entries, err := os.ReadDir(path)
	if err != nil {
		return nil, err
	}
	signature := directoryInfo{name: filepath.Base(path)}
	for _, entry := range entries {
		if !entry.Type().IsRegular() {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			return nil, err
		}
		signature.size += info.Size()
		if info.ModTime().After(signature.modified) {
			signature.modified = info.ModTime()
		}
	}
	return signature, nil
}

// directoryInfo is a directory's signature as a file stat (directorySignature).
type directoryInfo struct {
	name     string
	size     int64
	modified time.Time
}

// Name is the directory's base name.
func (info directoryInfo) Name() string { return info.name }

// Size is the sum of its files' sizes.
func (info directoryInfo) Size() int64 { return info.size }

// Mode says it is a directory.
func (info directoryInfo) Mode() fs.FileMode { return fs.ModeDir }

// ModTime is its newest file's time.
func (info directoryInfo) ModTime() time.Time { return info.modified }

// IsDir is true.
func (info directoryInfo) IsDir() bool { return true }

// Sys is nil: the signature has no system stat.
func (info directoryInfo) Sys() any { return nil }

// SizedEntry reports the total bytes of a local model's distinct weight artifacts.
type SizedEntry struct {
	Entry
	Bytes int64
}

// LocalServable lists the whole servable catalog's present, fresh entries
// with local weights, smallest first by total weight bytes and then by location.
// Split models include every shard rather than only their entry file.
// Both quick measurements and benchmark reports consume this ordering.
func LocalServable(ctx context.Context, store *overgodb.Store) ([]SizedEntry, error) {
	// The denominator is the whole catalog: the listing bound is the
	// widest the query accepts, never a sample.
	entries, err := ServableWithMemo(ctx, store, math.MaxInt, LoadMemo(ctx, store))
	if err != nil {
		return nil, err
	}
	var sized []SizedEntry
	for _, entry := range entries {
		// A hosted model is present when its key is set, but has no weights here.
		if !entry.Present || entry.Stale != "" || entry.Location == "" || remoteprovider.IsRemoteLocation(entry.Location) {
			continue
		}
		manifest, found, err := store.Manifest(ctx, entry.Model)
		if err != nil || !found {
			return nil, errors.Join(err, fmt.Errorf("discovery: missing weight inventory for %s", entry.Model))
		}
		var total uint64
		seen := make(map[artifact.ID]bool)
		for _, component := range manifest.Components {
			if seen[component.Artifact] || component.Role != artifact.ComponentWeights && component.Role != artifact.ComponentWeightsShard {
				continue
			}
			descriptor, found, err := store.Artifact(ctx, component.Artifact)
			if err != nil || !found {
				return nil, errors.Join(err, fmt.Errorf("discovery: missing weights %s", component.Artifact))
			}
			if descriptor.Size > math.MaxInt64-total {
				return nil, fmt.Errorf("discovery: total weight bytes exceed int64 for %s", entry.Model)
			}
			total += descriptor.Size
			seen[component.Artifact] = true
		}
		if total == 0 {
			return nil, fmt.Errorf("discovery: no weight bytes declared for %s", entry.Model)
		}
		sized = append(sized, SizedEntry{Entry: entry, Bytes: int64(total)})
	}
	slices.SortFunc(sized, func(left, right SizedEntry) int {
		return cmp.Or(cmp.Compare(left.Bytes, right.Bytes), strings.Compare(left.Location, right.Location))
	})
	return sized, nil
}
