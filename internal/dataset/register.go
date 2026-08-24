package dataset

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

// modalityByExtension maps file formats to their modality class; a
// format outside the table reports the honest unknown rather than a
// guess. Format knowledge is generic -- no dataset names appear here.
var modalityByExtension = map[string]string{
	".mp4": "video", ".webm": "video", ".mov": "video", ".mkv": "video",
	".jpg": "image", ".jpeg": "image", ".png": "image", ".gif": "image",
	".wav": "audio", ".mp3": "audio", ".flac": "audio", ".ogg": "audio",
	".json": "structured", ".jsonl": "structured", ".csv": "structured", ".parquet": "structured",
	".txt": "text", ".md": "text",
}

// RegisteredDataset reports one completed registration.
type RegisteredDataset struct {
	Commit  artifact.CommitID `json:"commit"`
	Dataset artifact.ID       `json:"dataset"`
	Catalog artifact.ID       `json:"catalog"`
	Files   uint64            `json:"files"`
	Bytes   uint64            `json:"bytes"`
	Changed bool              `json:"changed"`
}

// RegisterDirectoryDataset compiles one on-disk directory into store
// authority: a file inventory, a version document, a location fact,
// and an updated active catalog carrying the new entry beside every
// existing one. Registration is idempotent -- re-registering an
// unchanged directory leaves the store untouched -- and supersession
// is explicit: a changed directory re-registers under the same name
// with compare-and-set aliases.
func RegisterDirectoryDataset(
	ctx context.Context,
	repository artifact.Repository,
	name, root string,
) (RegisteredDataset, error) {
	if ctx == nil || repository == nil {
		return RegisteredDataset{}, errors.New("dataset: nil register context or repository")
	}
	if strings.TrimSpace(name) == "" {
		return RegisteredDataset{}, errors.New("dataset: register requires a dataset name")
	}
	absolute, err := filepath.Abs(root)
	if err != nil {
		return RegisteredDataset{}, err
	}
	files, formats, totalBytes, err := walkDirectoryInventory(absolute)
	if err != nil {
		return RegisteredDataset{}, err
	}
	if len(files) == 0 {
		return RegisteredDataset{}, fmt.Errorf("dataset: %q holds no files to register", root)
	}
	inventory, err := NewInventory(files)
	if err != nil {
		return RegisteredDataset{}, err
	}
	version, err := NewVersion([]Asset{{Name: "inventory", Artifact: inventory.ID, Records: uint64(len(files))}})
	if err != nil {
		return RegisteredDataset{}, err
	}
	entry := CatalogEntry{
		Name: name, Dataset: version.ID, Inventory: inventory.ID,
		StorageKind: "directory", Source: "local",
		Modality: dominantModality(files), Formats: formats,
		Files: uint64(len(files)), Bytes: totalBytes,
	}
	catalog, err := replacedCatalog(ctx, repository, entry)
	if err != nil {
		return RegisteredDataset{}, err
	}
	batch, changed, err := registerBatch(ctx, repository, name, absolute, inventory, version, catalog)
	if err != nil {
		return RegisteredDataset{}, err
	}
	result := RegisteredDataset{
		Dataset: version.ID, Catalog: catalog.ID,
		Files: uint64(len(files)), Bytes: totalBytes, Changed: changed,
	}
	if !changed {
		result.Commit, _ = repository.Head()
		return result, nil
	}
	commit, err := artifact.CommitBatch(ctx, repository, batch)
	if err != nil {
		return RegisteredDataset{}, err
	}
	result.Commit = commit
	return result, nil
}

func walkDirectoryInventory(root string) ([]InventoryFile, []string, uint64, error) {
	var files []InventoryFile
	formats := map[string]bool{}
	var total uint64
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil || entry.IsDir() {
			return err
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		relative, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		extension := strings.ToLower(filepath.Ext(entry.Name()))
		modality, known := modalityByExtension[extension]
		if !known {
			modality = legacyUnknownFact
		}
		format := strings.TrimPrefix(extension, ".")
		if format == "" {
			format = legacyUnknownFact
		}
		formats[format] = true
		total += uint64(info.Size())
		files = append(files, InventoryFile{
			Path: filepath.ToSlash(relative), OriginalName: entry.Name(), Extension: extension,
			Modality: modality, Format: format, Bytes: uint64(info.Size()),
			ModifiedUnix: info.ModTime().Unix(), Structured: modality == "structured",
		})
		return nil
	})
	if err != nil {
		return nil, nil, 0, err
	}
	slices.SortFunc(files, func(left, right InventoryFile) int { return strings.Compare(left.Path, right.Path) })
	names := make([]string, 0, len(formats))
	for format := range formats {
		names = append(names, format)
	}
	slices.Sort(names)
	return files, names, total, nil
}

// dominantModality reports the modality carrying the most bytes: the
// class the dataset is FOR, with sidecar metadata never outvoting it.
func dominantModality(files []InventoryFile) string {
	weights := map[string]uint64{}
	for _, file := range files {
		weights[file.Modality] += file.Bytes
	}
	dominant, best := legacyUnknownFact, uint64(0)
	for modality, weight := range weights {
		if weight > best || (weight == best && modality < dominant) {
			dominant, best = modality, weight
		}
	}
	return dominant
}

// replacedCatalog returns the active catalog with the entry replacing
// any same-named predecessor, or appended when the name is new.
func replacedCatalog(ctx context.Context, reader artifact.Reader, entry CatalogEntry) (Document, error) {
	current, found, err := ResolveCatalog(ctx, reader)
	if err != nil {
		return Document{}, err
	}
	var entries []CatalogEntry
	if found {
		entries = slices.Clone(current.Catalog)
	}
	replaced := false
	for index := range entries {
		if entries[index].Name == entry.Name {
			entries[index], replaced = entry, true
			break
		}
	}
	if !replaced {
		entries = append(entries, entry)
	}
	return NewCatalog(entries)
}

func registerBatch(
	ctx context.Context,
	repository artifact.Repository,
	name, root string,
	inventory Inventory,
	version, catalog Document,
) (artifact.Batch, bool, error) {
	batch := artifact.Batch{}
	inventoryContent, err := inventoryContent(inventory)
	if err != nil {
		return artifact.Batch{}, false, err
	}
	versionContent, err := version.Content()
	if err != nil {
		return artifact.Batch{}, false, err
	}
	catalogContent, err := catalog.Content()
	if err != nil {
		return artifact.Batch{}, false, err
	}
	batch.Contents = append(batch.Contents, inventoryContent, versionContent, catalogContent)
	batch.Lineage = append(batch.Lineage, version.Lineage()...)
	batch.Lineage = append(batch.Lineage, catalog.Lineage()...)
	location, err := artifact.CanonicalLocalLocation(version.ID, artifact.LocationDirectory, root)
	if err != nil {
		return artifact.Batch{}, false, err
	}
	batch.Locations = append(batch.Locations, artifact.LocationEvent{Location: location, Action: artifact.LocationAdd})
	digest := sha256.New()
	changed := false
	for _, binding := range []struct {
		alias  string
		target artifact.ID
	}{
		{registeredAlias(name), version.ID},
		{CatalogAlias, catalog.ID},
	} {
		current, bound, err := repository.ResolveAlias(ctx, binding.alias)
		if err != nil {
			return artifact.Batch{}, false, err
		}
		fmt.Fprintf(digest, "%s\x00%s\x00", binding.alias, binding.target)
		if bound && current == binding.target {
			continue
		}
		changed = true
		aliasBinding := artifact.AliasBinding{Name: binding.alias, Target: binding.target}
		if bound {
			aliasBinding.Previous = artifact.IDPointer(current)
			fmt.Fprintf(digest, "previous=%s\x00", current)
		}
		batch.Aliases = append(batch.Aliases, aliasBinding)
	}
	if err := ensureRegisteredLocation(ctx, repository, version.ID, location, &changed); err != nil {
		return artifact.Batch{}, false, err
	}
	batch.Key = fmt.Sprintf("dataset/register/%s/%x", name, digest.Sum(nil))
	return batch, changed, nil
}

// ensureRegisteredLocation keeps registration idempotent while letting
// a re-registration record a location the store has not seen yet.
func ensureRegisteredLocation(
	ctx context.Context,
	reader artifact.Reader,
	dataset artifact.ID,
	location artifact.Location,
	changed *bool,
) error {
	if *changed {
		return nil
	}
	recorded, err := reader.Locations(ctx, dataset)
	if err != nil {
		// The dataset document may not exist yet on first registration.
		if _, statErr := os.Stat(location.Value); statErr != nil {
			return statErr
		}
		*changed = true
		return nil
	}
	if !slices.Contains(recorded, location) {
		*changed = true
	}
	return nil
}
