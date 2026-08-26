package evaluation

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/strictjson"
)

const (
	benchmarkCatalogMediaType = "application/vnd.overgo.benchmark-catalog+json"
	benchmarkCatalogSchema    = "overgo/benchmark-catalog/v1"
	benchmarkCatalogAlias     = "evaluation/catalogs/active"
)

var benchmarkCatalogCodec = artifact.JSONDocumentCodec(
	"benchmark catalog", artifact.KindDataset, benchmarkCatalogMediaType, benchmarkCatalogSchema,
	canonicalizeBenchmarkCatalog, func(value benchmarkCatalog) artifact.ID { return value.ID },
	func(value *benchmarkCatalog, id artifact.ID) { value.ID = id },
	func(value benchmarkCatalog) benchmarkCatalog {
		value.Entries = slices.Clone(value.Entries)
		return value
	},
)

type benchmarkDeclaration struct {
	Name string                      `json:"name"`
	Path string                      `json:"path"`
	Spec dataset.BenchmarkImportSpec `json:"spec"`
}

type benchmarkManifest struct {
	Datasets []benchmarkDeclaration `json:"datasets"`
}

type benchmarkEntry struct {
	Name    string      `json:"name"`
	Split   string      `json:"split"`
	Dataset artifact.ID `json:"dataset"`
	Profile artifact.ID `json:"profile"`
}

type benchmarkCatalog struct {
	ID      artifact.ID      `json:"-"`
	Version uint16           `json:"version"`
	Entries []benchmarkEntry `json:"entries"`
}

func CatalogLocalBenchmarks(ctx context.Context, repository artifact.Repository, manifestPath string) (artifact.ID, error) {
	if ctx == nil || repository == nil {
		return artifact.ID{}, errors.New("evaluation: benchmark catalog repository is absent")
	}
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		return artifact.ID{}, err
	}
	var manifest benchmarkManifest
	if err := strictjson.DecodeBytes(data, &manifest); err != nil {
		return artifact.ID{}, err
	}
	if len(manifest.Datasets) == 0 {
		return artifact.ID{}, errors.New("evaluation: benchmark catalog is empty")
	}
	return catalogBenchmarkDeclarations(ctx, repository, filepath.Dir(manifestPath), manifest.Datasets)
}

// catalogBenchmarkDeclarations imports every declaration and publishes
// the active catalog; an empty base requires absolute paths (the cache
// scan derives them), a manifest base anchors relative ones.
func catalogBenchmarkDeclarations(
	ctx context.Context,
	repository artifact.Repository,
	base string,
	declarations []benchmarkDeclaration,
) (artifact.ID, error) {
	catalog := benchmarkCatalog{Version: artifact.InitialDocumentVersion, Entries: make([]benchmarkEntry, 0, len(declarations))}
	// Publishing merges onto the standing catalog: a DNA corpus import
	// must not evict the lm_eval entries, and re-imports overwrite their
	// own names. Entries whose name a new declaration carries drop here
	// and re-enter from the fresh import below.
	if currentID, bound, err := artifact.ResolveAlias(ctx, repository, benchmarkCatalogAlias); err != nil {
		return artifact.ID{}, err
	} else if bound {
		current, found, err := benchmarkCatalogCodec.Read(ctx, repository, currentID)
		if err != nil || !found {
			return artifact.ID{}, errors.Join(err, errors.New("evaluation: standing benchmark catalog is unreadable"))
		}
		replaced := make(map[string]bool, len(declarations))
		for _, declaration := range declarations {
			replaced[strings.TrimSpace(declaration.Name)] = true
		}
		for _, entry := range current.Entries {
			if !replaced[entry.Name] {
				catalog.Entries = append(catalog.Entries, entry)
			}
		}
	}
	for _, declaration := range declarations {
		name := strings.TrimSpace(declaration.Name)
		path := declaration.Path
		if base != "" {
			var err error
			if path, err = localBenchmarkPath(base, declaration.Path); err != nil {
				return artifact.ID{}, errors.New("evaluation: invalid benchmark declaration")
			}
		} else if !filepath.IsAbs(path) {
			return artifact.ID{}, errors.New("evaluation: derived benchmark path must be absolute")
		}
		if name == "" {
			return artifact.ID{}, errors.New("evaluation: invalid benchmark declaration")
		}
		imported, err := dataset.ImportBenchmark(ctx, repository, path, declaration.Spec)
		if err != nil {
			return artifact.ID{}, err
		}
		catalog.Entries = append(catalog.Entries, benchmarkEntry{
			Name: name, Split: imported.Spec.Split, Dataset: imported.ID, Profile: imported.Profile,
		})
	}
	catalog, err := benchmarkCatalogCodec.New(catalog)
	if err != nil {
		return artifact.ID{}, err
	}
	previous, exists, err := artifact.ResolveAlias(ctx, repository, benchmarkCatalogAlias)
	if err != nil {
		return artifact.ID{}, err
	}
	if exists && previous == catalog.ID {
		return catalog.ID, nil
	}
	parents := make([]artifact.ID, len(catalog.Entries))
	for index := range catalog.Entries {
		parents[index] = catalog.Entries[index].Dataset
	}
	alias := artifact.AliasBinding{Name: benchmarkCatalogAlias, Target: catalog.ID}
	if exists {
		alias.Previous = artifact.IDPointer(previous)
	}
	batch, err := benchmarkCatalogCodec.Batch(
		"evaluation/catalog/"+catalog.ID.String(), catalog,
		artifact.DependencyLineage(catalog.ID, parents...), []artifact.AliasBinding{alias},
	)
	if err != nil {
		return artifact.ID{}, err
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return artifact.ID{}, err
	}
	return catalog.ID, nil
}

func canonicalizeBenchmarkCatalog(catalog *benchmarkCatalog) error {
	if catalog == nil || catalog.Version != artifact.InitialDocumentVersion || len(catalog.Entries) == 0 {
		return errors.New("evaluation: invalid benchmark catalog")
	}
	sort.Slice(catalog.Entries, func(i, j int) bool {
		if catalog.Entries[i].Name != catalog.Entries[j].Name {
			return catalog.Entries[i].Name < catalog.Entries[j].Name
		}
		return catalog.Entries[i].Split < catalog.Entries[j].Split
	})
	for index, entry := range catalog.Entries {
		if strings.TrimSpace(entry.Name) != entry.Name || entry.Name == "" || strings.TrimSpace(entry.Split) != entry.Split ||
			entry.Split == "" || entry.Dataset.Kind() != artifact.KindDataset || entry.Profile.Kind() != artifact.KindProfile ||
			index > 0 && catalog.Entries[index-1].Name == entry.Name && catalog.Entries[index-1].Split == entry.Split {
			return errors.New("evaluation: invalid benchmark catalog entry")
		}
	}
	return nil
}

func localBenchmarkPath(base, relative string) (string, error) {
	relative = filepath.Clean(strings.TrimSpace(relative))
	if relative == "." || filepath.IsAbs(relative) || !filepath.IsLocal(relative) {
		return "", errors.New("evaluation: benchmark path is not local")
	}
	return filepath.Join(base, relative), nil
}
