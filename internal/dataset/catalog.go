package dataset

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

const (
	// CatalogAlias names the active catalog document.
	CatalogAlias = "datasets/catalog/active"
	// RegisteredAliasPrefix scopes stable dataset names.
	RegisteredAliasPrefix = "dataset.registered."
	legacyRegistryName    = "datasets.log"
	legacyUnknownFact     = "unknown"
	rootInventorySource   = "content-root"
)

// CompiledCatalog owns one import-ready dataset catalog.
type CompiledCatalog struct {
	Catalog     Document
	Datasets    []Document
	Inventories []Inventory
	Locations   []artifact.Location
}

// CatalogEntryStatus classifies one published dataset binding.
type CatalogEntryStatus string

const (
	// CatalogEntryPublished marks a valid registered dataset.
	CatalogEntryPublished CatalogEntryStatus = "published"
	// CatalogEntryMissing marks an absent registered alias.
	CatalogEntryMissing CatalogEntryStatus = "missing"
	// CatalogEntryInvalid marks inconsistent catalog facts.
	CatalogEntryInvalid CatalogEntryStatus = "invalid"
)

// CatalogCoverage reports active dataset catalog integrity.
type CatalogCoverage struct {
	Catalog    *artifact.ID           `json:"catalog,omitempty"`
	Registered int                    `json:"registered"`
	Published  int                    `json:"published"`
	Available  int                    `json:"available"`
	Complete   bool                   `json:"complete"`
	Entries    []CatalogCoverageEntry `json:"entries"`
}

// CatalogCoverageEntry reports one dataset binding.
type CatalogCoverageEntry struct {
	Entry     CatalogEntry       `json:"entry"`
	Alias     string             `json:"alias"`
	Status    CatalogEntryStatus `json:"status"`
	Available bool               `json:"available"`
	Location  string             `json:"location,omitzero"`
	Detail    string             `json:"detail,omitzero"`
}

// CatalogPublication reports one atomic dataset catalog publication.
type CatalogPublication struct {
	Commit   artifact.CommitID `json:"commit"`
	Changed  bool              `json:"changed"`
	Coverage CatalogCoverage   `json:"coverage"`
}

// CompileLegacyCatalog converts the legacy registry into current documents.
func CompileLegacyCatalog(legacyStore, contentRoot string) (CompiledCatalog, error) {
	registry, err := readLegacyRegistry(filepath.Join(legacyStore, legacyRegistryName))
	if err != nil {
		return CompiledCatalog{}, err
	}
	root, err := filepath.Abs(contentRoot)
	if err != nil {
		return CompiledCatalog{}, err
	}
	physical, err := os.ReadDir(root)
	if err != nil {
		return CompiledCatalog{}, err
	}
	remaining := make(map[string]os.DirEntry, len(physical))
	for _, entry := range physical {
		remaining[entry.Name()] = entry
	}
	bundle := CompiledCatalog{
		Datasets:    make([]Document, 0, len(registry.Datasets)),
		Inventories: make([]Inventory, 0, len(registry.Datasets)),
		Locations:   make([]artifact.Location, 0, len(registry.Datasets)),
	}
	entries := make([]CatalogEntry, 0, len(registry.Datasets))
	for _, source := range registry.Datasets {
		files := registry.Files[source.ID]
		entryName, locationPath, kind, err := legacyDatasetLocation(root, source, files)
		if err != nil {
			return CompiledCatalog{}, err
		}
		_, available := remaining[entryName]
		if available {
			delete(remaining, entryName)
		}
		inventoryFiles, formats, bytes, err := compileLegacyFiles(files)
		if err != nil {
			return CompiledCatalog{}, fmt.Errorf("dataset: compile %q: %w", source.Name, err)
		}
		if source.FileCount != int64(len(files)) || source.ByteCount != bytes {
			return CompiledCatalog{}, fmt.Errorf("dataset: legacy rollup differs for %q", source.Name)
		}
		inventory, err := NewInventory(inventoryFiles)
		if err != nil {
			return CompiledCatalog{}, fmt.Errorf("dataset: inventory %q: %w", source.Name, err)
		}
		version, err := NewVersion([]Asset{{Name: InventoryAssetName, Artifact: inventory.ID, Records: uint64(len(files))}})
		if err != nil {
			return CompiledCatalog{}, err
		}
		bundle.Inventories = append(bundle.Inventories, inventory)
		bundle.Datasets = append(bundle.Datasets, version)
		if available {
			location, err := artifact.CanonicalLocalLocation(version.ID, kind, locationPath)
			if err != nil {
				return CompiledCatalog{}, err
			}
			bundle.Locations = append(bundle.Locations, location)
		}
		entries = append(entries, CatalogEntry{
			Name: source.Name, Dataset: version.ID, Inventory: inventory.ID,
			StorageKind: normalizedLegacyFact(source.Kind), Source: normalizedLegacyFact(source.Source),
			Modality: normalizedLegacyFact(source.Modality), Formats: formats,
			Files: uint64(len(files)), Bytes: uint64(bytes),
		})
	}
	for _, entry := range sortedDirectoryEntries(remaining) {
		inventoryFiles, bytes, err := inventoryRootEntry(root, entry)
		if err != nil {
			return CompiledCatalog{}, err
		}
		inventory, err := NewInventory(inventoryFiles)
		if err != nil {
			return CompiledCatalog{}, err
		}
		version, err := NewVersion([]Asset{{Name: InventoryAssetName, Artifact: inventory.ID, Records: uint64(len(inventoryFiles))}})
		if err != nil {
			return CompiledCatalog{}, err
		}
		kind, storageKind := artifact.LocationFile, "file"
		if entry.IsDir() {
			kind, storageKind = artifact.LocationDirectory, "directory"
		}
		location, err := artifact.CanonicalLocalLocation(version.ID, kind, filepath.Join(root, entry.Name()))
		if err != nil {
			return CompiledCatalog{}, err
		}
		bundle.Inventories = append(bundle.Inventories, inventory)
		bundle.Datasets = append(bundle.Datasets, version)
		bundle.Locations = append(bundle.Locations, location)
		entries = append(entries, CatalogEntry{
			Name: entry.Name(), Dataset: version.ID, Inventory: inventory.ID, StorageKind: storageKind,
			Source: rootInventorySource, Modality: legacyUnknownFact, Formats: []string{legacyUnknownFact},
			Files: uint64(len(inventoryFiles)), Bytes: uint64(bytes),
		})
	}
	bundle.Catalog, err = NewCatalog(entries)
	return bundle, err
}

func sortedDirectoryEntries(entries map[string]os.DirEntry) []os.DirEntry {
	result := slices.SortedFunc(maps.Values(entries), func(left, right os.DirEntry) int {
		return strings.Compare(left.Name(), right.Name())
	})
	switch {
	case result == nil:
		result = []os.DirEntry{}
	}
	return result
}

func inventoryRootEntry(root string, entry os.DirEntry) ([]InventoryFile, int64, error) {
	entryPath := filepath.Join(root, entry.Name())
	if !entry.IsDir() {
		info, err := entry.Info()
		if err != nil {
			return nil, 0, err
		}
		return []InventoryFile{filesystemInventoryFile(entry.Name(), info)}, info.Size(), nil
	}
	var files []InventoryFile
	var bytes int64
	err := filepath.WalkDir(entryPath, func(path string, child os.DirEntry, walkErr error) error {
		if walkErr != nil || child.IsDir() {
			return walkErr
		}
		info, err := child.Info()
		if err != nil {
			return err
		}
		relative, err := filepath.Rel(entryPath, path)
		if err != nil {
			return err
		}
		files = append(files, filesystemInventoryFile(filepath.ToSlash(relative), info))
		bytes += info.Size()
		return nil
	})
	return files, bytes, err
}

func filesystemInventoryFile(path string, info os.FileInfo) InventoryFile {
	return InventoryFile{
		Path: path, OriginalName: info.Name(), Extension: filepath.Ext(info.Name()),
		Modality: legacyUnknownFact, Format: legacyUnknownFact, Bytes: uint64(info.Size()), ModifiedUnix: info.ModTime().Unix(),
	}
}

// PublishLegacyCatalog atomically publishes catalog documents and locations.
func PublishLegacyCatalog(
	ctx context.Context,
	repository artifact.Repository,
	legacyStore, contentRoot string,
) (CatalogPublication, error) {
	bundle, err := CompileLegacyCatalog(legacyStore, contentRoot)
	if err != nil {
		return CatalogPublication{}, err
	}
	return PublishCatalog(ctx, repository, bundle)
}

// PublishCatalog atomically publishes compiled dataset authority.
func PublishCatalog(ctx context.Context, repository artifact.Repository, bundle CompiledCatalog) (CatalogPublication, error) {
	if ctx == nil || repository == nil {
		return CatalogPublication{}, errors.New("dataset: nil catalog context or repository")
	}
	if err := validateCompiledCatalog(bundle); err != nil {
		return CatalogPublication{}, err
	}
	coverage, err := InspectCatalog(ctx, repository)
	if err != nil {
		return CatalogPublication{}, err
	}
	// A dataset can appear on disk after its documents published: the
	// aliases are complete but its location fact is still unrecorded.
	// Idempotence therefore requires no pending locations, not only
	// complete aliases -- "catalog first, download later" must converge.
	pending, err := pendingLocations(ctx, repository, bundle.Locations)
	if err != nil {
		return CatalogPublication{}, err
	}
	if coverage.Complete && coverage.Catalog != nil && *coverage.Catalog == bundle.Catalog.ID && len(pending) == 0 {
		commit, _ := repository.Head()
		return CatalogPublication{Commit: commit, Coverage: coverage}, nil
	}
	batch, err := catalogBatch(ctx, repository, bundle, pending)
	if err != nil {
		return CatalogPublication{}, err
	}
	commit, err := artifact.CommitBatch(ctx, repository, batch)
	if err != nil {
		return CatalogPublication{}, err
	}
	coverage, err = InspectCatalog(ctx, repository)
	if err != nil {
		return CatalogPublication{}, err
	}
	if !coverage.Complete || coverage.Catalog == nil || *coverage.Catalog != bundle.Catalog.ID {
		return CatalogPublication{}, errors.New("dataset: catalog publication is incomplete")
	}
	return CatalogPublication{Commit: commit, Changed: true, Coverage: coverage}, nil
}

func validateCompiledCatalog(bundle CompiledCatalog) error {
	if bundle.Catalog.Type != TypeCatalog || len(bundle.Catalog.Catalog) != len(bundle.Datasets) ||
		len(bundle.Datasets) != len(bundle.Inventories) || len(bundle.Locations) > len(bundle.Datasets) {
		return errors.New("dataset: invalid compiled catalog")
	}
	if err := bundle.Catalog.ValidateIdentity(); err != nil {
		return err
	}
	documents := make(map[artifact.ID]Document, len(bundle.Datasets))
	for _, document := range bundle.Datasets {
		if err := document.ValidateIdentity(); err != nil || document.Type != TypeVersion {
			return errors.New("dataset: invalid compiled dataset version")
		}
		documents[document.ID] = document
	}
	inventories := make(map[artifact.ID]struct{}, len(bundle.Inventories))
	for _, inventory := range bundle.Inventories {
		if _, err := inventoryContent(inventory); err != nil {
			return err
		}
		inventories[inventory.ID] = struct{}{}
	}
	for _, entry := range bundle.Catalog.Catalog {
		document, found := documents[entry.Dataset]
		_, inventoryFound := inventories[entry.Inventory]
		if !found || !inventoryFound || len(document.Assets) != 1 || document.Assets[0].Artifact != entry.Inventory {
			return errors.New("dataset: compiled catalog entry differs from its documents")
		}
	}
	for _, location := range bundle.Locations {
		if _, found := documents[location.Artifact]; !found {
			return errors.New("dataset: compiled catalog location has no dataset")
		}
	}
	return nil
}

// ResolveCatalog returns the active dataset catalog document.
func ResolveCatalog(ctx context.Context, reader artifact.Reader) (Document, bool, error) {
	return Resolve(ctx, reader, CatalogAlias)
}

// InspectCatalog validates active aliases, documents, inventories, and locations.
func InspectCatalog(ctx context.Context, reader artifact.Reader) (CatalogCoverage, error) {
	if ctx == nil || reader == nil {
		return CatalogCoverage{}, errors.New("dataset: nil catalog context or reader")
	}
	catalog, found, err := ResolveCatalog(ctx, reader)
	if err != nil || !found {
		return CatalogCoverage{}, err
	}
	if catalog.Type != TypeCatalog {
		return CatalogCoverage{}, errors.New("dataset: active catalog alias is not a catalog")
	}
	coverage := CatalogCoverage{
		Catalog: artifact.IDPointer(catalog.ID), Registered: len(catalog.Catalog),
		Entries: make([]CatalogCoverageEntry, len(catalog.Catalog)),
	}
	for index, entry := range catalog.Catalog {
		row := CatalogCoverageEntry{Entry: entry, Alias: registeredAlias(entry.Name), Status: CatalogEntryMissing}
		target, aliasFound, resolveErr := reader.ResolveAlias(ctx, row.Alias)
		if resolveErr != nil {
			return CatalogCoverage{}, resolveErr
		}
		if aliasFound && target == entry.Dataset {
			row.Status, row.Detail = validateCatalogEntry(ctx, reader, entry)
			if row.Status == CatalogEntryPublished {
				coverage.Published++
			}
		} else if aliasFound {
			row.Status, row.Detail = CatalogEntryInvalid, "registered alias target differs"
		}
		locations, locationErr := reader.Locations(ctx, entry.Dataset)
		if locationErr != nil {
			return CatalogCoverage{}, locationErr
		}
		for _, location := range locations {
			if location.Kind != artifact.LocationFile && location.Kind != artifact.LocationDirectory {
				continue
			}
			if _, statErr := os.Stat(location.Value); statErr == nil {
				row.Available, row.Location = true, location.Value
				coverage.Available++
				break
			}
		}
		coverage.Entries[index] = row
	}
	coverage.Complete = coverage.Registered != 0 && coverage.Published == coverage.Registered
	return coverage, nil
}

func legacyDatasetLocation(root string, source legacyDataset, files []legacyDatasetFile) (string, string, artifact.LocationKind, error) {
	switch source.Kind {
	case "directory":
		return source.Name, filepath.Join(root, source.Name), artifact.LocationDirectory, nil
	case "file":
		if len(files) != 1 || files[0].Path == "" || filepath.Base(files[0].Path) != files[0].Path {
			return "", "", artifact.LocationInvalid, fmt.Errorf("dataset: invalid file-backed legacy dataset %q", source.Name)
		}
		return files[0].Path, filepath.Join(root, files[0].Path), artifact.LocationFile, nil
	default:
		return "", "", artifact.LocationInvalid, fmt.Errorf("dataset: invalid legacy storage kind %q", source.Kind)
	}
}

func compileLegacyFiles(files []legacyDatasetFile) ([]InventoryFile, []string, int64, error) {
	result := make([]InventoryFile, len(files))
	formatSet := map[string]struct{}{}
	var bytes int64
	for index, file := range files {
		if file.Bytes < 0 || bytes > int64(^uint64(0)>>1)-file.Bytes {
			return nil, nil, 0, errors.New("invalid legacy file byte count")
		}
		bytes += file.Bytes
		format := normalizedLegacyFact(file.Format)
		formatSet[format] = struct{}{}
		result[index] = InventoryFile{
			Path: file.Path, OriginalName: file.OriginalName, Extension: file.Extension,
			Modality: normalizedLegacyFact(file.Modality), Format: format, Bytes: uint64(file.Bytes),
			ModifiedUnix: file.ModifiedUnix, Structured: file.Structured, Attributes: file.Attributes,
		}
	}
	formats := slices.Sorted(maps.Keys(formatSet))
	switch {
	case formats == nil:
		formats = []string{}
	}
	return result, formats, bytes, nil
}

func normalizedLegacyFact(value string) string {
	if value = strings.TrimSpace(value); value != "" {
		return value
	}
	return legacyUnknownFact
}

func validateCatalogEntry(ctx context.Context, reader artifact.Reader, entry CatalogEntry) (CatalogEntryStatus, string) {
	document, found, err := Load(ctx, reader, entry.Dataset)
	if err != nil || !found {
		return CatalogEntryInvalid, errorDetail(err, "dataset document is absent")
	}
	if document.Type != TypeVersion || len(document.Assets) != 1 || document.Assets[0].Artifact != entry.Inventory || document.Assets[0].Records != entry.Files {
		return CatalogEntryInvalid, "dataset version differs from catalog"
	}
	inventory, found, err := loadInventory(ctx, reader, entry.Inventory)
	if err != nil || !found {
		return CatalogEntryInvalid, errorDetail(err, "dataset inventory is absent")
	}
	var bytes uint64
	for _, file := range inventory.Files {
		bytes += file.Bytes
	}
	if len(inventory.Files) != int(entry.Files) || bytes != entry.Bytes {
		return CatalogEntryInvalid, "dataset inventory rollup differs"
	}
	return CatalogEntryPublished, ""
}

func errorDetail(err error, fallback string) string {
	if err != nil {
		return err.Error()
	}
	return fallback
}

// pendingLocations filters the compiled location facts to those the
// store has not yet recorded for their artifact.
func pendingLocations(ctx context.Context, reader artifact.Reader, locations []artifact.Location) ([]artifact.Location, error) {
	var pending []artifact.Location
	for _, location := range locations {
		recorded, err := reader.Locations(ctx, location.Artifact)
		if err != nil {
			return nil, err
		}
		if !slices.Contains(recorded, location) {
			pending = append(pending, location)
		}
	}
	return pending, nil
}

func catalogBatch(ctx context.Context, repository artifact.Repository, bundle CompiledCatalog, pending []artifact.Location) (artifact.Batch, error) {
	batch := artifact.Batch{}
	for _, inventory := range bundle.Inventories {
		content, err := inventoryContent(inventory)
		if err != nil {
			return artifact.Batch{}, err
		}
		batch.Contents = append(batch.Contents, content)
	}
	for _, document := range append(slices.Clone(bundle.Datasets), bundle.Catalog) {
		content, err := document.Content()
		if err != nil {
			return artifact.Batch{}, err
		}
		batch.Contents = append(batch.Contents, content)
		batch.Lineage = append(batch.Lineage, document.Lineage()...)
	}
	for _, location := range pending {
		batch.Locations = append(batch.Locations, artifact.LocationEvent{Location: location, Action: artifact.LocationAdd})
	}
	bindings := make([]struct {
		name   string
		target artifact.ID
	}, 0, len(bundle.Catalog.Catalog)+1)
	bindings = append(bindings, struct {
		name   string
		target artifact.ID
	}{CatalogAlias, bundle.Catalog.ID})
	for _, entry := range bundle.Catalog.Catalog {
		bindings = append(bindings, struct {
			name   string
			target artifact.ID
		}{registeredAlias(entry.Name), entry.Dataset})
	}
	digest := sha256.New()
	// Location facts join the batch identity: a location-only
	// republication must key differently from the alias publication
	// that preceded it, or the store refuses the retry as a conflict.
	for _, location := range pending {
		fmt.Fprintf(digest, "location\x00%s\x00%d\x00%s\x00", location.Artifact, location.Kind, location.Value)
	}
	for _, binding := range bindings {
		current, found, err := repository.ResolveAlias(ctx, binding.name)
		if err != nil {
			return artifact.Batch{}, err
		}
		fmt.Fprintf(digest, "%s\x00%s\x00", binding.name, binding.target)
		if found && current == binding.target {
			continue
		}
		alias := artifact.AliasBinding{Name: binding.name, Target: binding.target}
		if found {
			alias.Previous = artifact.IDPointer(current)
			fmt.Fprintf(digest, "previous=%s\x00", current)
		}
		batch.Aliases = append(batch.Aliases, alias)
	}
	batch.Key = fmt.Sprintf("dataset/catalog/%x", digest.Sum(nil))
	return batch, nil
}

func registeredAlias(name string) string {
	return RegisteredAliasPrefix + name
}
