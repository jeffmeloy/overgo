package modelrecipe

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/model"
)

// RegisteredProfileAliasPrefix scopes current architecture profiles.
const RegisteredProfileAliasPrefix = "profile.registered."

// ProfileCatalogStatus classifies one registry binding.
type ProfileCatalogStatus string

const (
	// ProfileCatalogPublished marks an exact, valid binding.
	ProfileCatalogPublished ProfileCatalogStatus = "published"
	// ProfileCatalogMissing marks an absent binding.
	ProfileCatalogMissing ProfileCatalogStatus = "missing"
	// ProfileCatalogMismatched marks an unexpected target.
	ProfileCatalogMismatched ProfileCatalogStatus = "mismatched"
	// ProfileCatalogInvalid marks invalid bound content or lineage.
	ProfileCatalogInvalid ProfileCatalogStatus = "invalid"
)

// ProfileCatalogEntry reports one registered architecture binding.
type ProfileCatalogEntry struct {
	Architecture string               `json:"architecture"`
	Alias        string               `json:"alias"`
	Expected     artifact.ID          `json:"expected"`
	Actual       *artifact.ID         `json:"actual,omitempty"`
	Status       ProfileCatalogStatus `json:"status"`
	Detail       string               `json:"detail,omitempty"`
}

// ProfileCatalogCoverage reports registry-to-OvergoDB parity.
type ProfileCatalogCoverage struct {
	Registered int                   `json:"registered"`
	Published  int                   `json:"published"`
	Complete   bool                  `json:"complete"`
	Entries    []ProfileCatalogEntry `json:"entries"`
}

// ProfileCatalogPublication reports one atomic catalog publication.
type ProfileCatalogPublication struct {
	Commit   artifact.CommitID      `json:"commit"`
	Changed  bool                   `json:"changed"`
	Coverage ProfileCatalogCoverage `json:"coverage"`
}

func compileArchitectureProfileCatalog() ([]ProfileDocument, error) {
	names := model.SupportedArchitectures()
	documents := make([]ProfileDocument, len(names))
	for index, name := range names {
		profile, ok := model.LookupArchitecture(name)
		if !ok {
			return nil, fmt.Errorf("model recipe: registered architecture %q is absent", name)
		}
		document, err := NewProfileDocument(profile)
		if err != nil {
			return nil, fmt.Errorf("model recipe: compile architecture %q: %w", name, err)
		}
		documents[index] = document
	}
	return documents, nil
}

// InspectArchitectureProfileCatalog compares OvergoDB with the registry.
func InspectArchitectureProfileCatalog(ctx context.Context, reader artifact.Reader) (ProfileCatalogCoverage, error) {
	if ctx == nil || reader == nil {
		return ProfileCatalogCoverage{}, errors.New("model recipe: nil profile catalog context or reader")
	}
	documents, err := compileArchitectureProfileCatalog()
	if err != nil {
		return ProfileCatalogCoverage{}, err
	}
	return inspectArchitectureProfileCatalog(ctx, reader, documents)
}

// ResolveRegisteredArchitectureProfile requires the published registry profile.
func ResolveRegisteredArchitectureProfile(
	ctx context.Context,
	reader artifact.Reader,
	architecture string,
) (ProfileDocument, error) {
	if ctx == nil || reader == nil {
		return ProfileDocument{}, errors.New("model recipe: nil registered profile context or reader")
	}
	profile, ok := model.LookupArchitecture(architecture)
	if !ok {
		return ProfileDocument{}, fmt.Errorf("model recipe: architecture %q is not registered", architecture)
	}
	expected, err := NewProfileDocument(profile)
	if err != nil {
		return ProfileDocument{}, err
	}
	alias := registeredProfileAlias(architecture)
	target, found, err := reader.ResolveAlias(ctx, alias)
	if err != nil {
		return ProfileDocument{}, err
	}
	if !found {
		return ProfileDocument{}, fmt.Errorf("model recipe: registered profile alias %q is absent", alias)
	}
	if target != expected.ID {
		return ProfileDocument{}, fmt.Errorf("model recipe: registered profile alias %q is stale", alias)
	}
	document, err := loadProfile(ctx, reader, target)
	if err != nil {
		return ProfileDocument{}, err
	}
	if document.Architecture != architecture {
		return ProfileDocument{}, errors.New("model recipe: registered profile architecture differs")
	}
	return document, nil
}

// PublishArchitectureProfileCatalog atomically publishes registry authority.
func PublishArchitectureProfileCatalog(ctx context.Context, repository artifact.Repository) (ProfileCatalogPublication, error) {
	if ctx == nil || repository == nil {
		return ProfileCatalogPublication{}, errors.New("model recipe: nil profile catalog context or repository")
	}
	documents, err := compileArchitectureProfileCatalog()
	if err != nil {
		return ProfileCatalogPublication{}, err
	}
	coverage, err := inspectArchitectureProfileCatalog(ctx, repository, documents)
	if err != nil {
		return ProfileCatalogPublication{}, err
	}
	if coverage.Complete {
		commit, _ := repository.Head()
		return ProfileCatalogPublication{Commit: commit, Coverage: coverage}, nil
	}
	batch, err := profileCatalogBatch(ctx, repository, documents)
	if err != nil {
		return ProfileCatalogPublication{}, err
	}
	commit, err := artifact.CommitBatch(ctx, repository, batch)
	if err != nil {
		return ProfileCatalogPublication{}, err
	}
	coverage, err = inspectArchitectureProfileCatalog(ctx, repository, documents)
	if err != nil {
		return ProfileCatalogPublication{}, err
	}
	if !coverage.Complete {
		return ProfileCatalogPublication{}, errors.New("model recipe: profile catalog publication is incomplete")
	}
	return ProfileCatalogPublication{Commit: commit, Changed: true, Coverage: coverage}, nil
}

func inspectArchitectureProfileCatalog(
	ctx context.Context,
	reader artifact.Reader,
	documents []ProfileDocument,
) (ProfileCatalogCoverage, error) {
	coverage := ProfileCatalogCoverage{Registered: len(documents), Entries: make([]ProfileCatalogEntry, len(documents))}
	for index, document := range documents {
		entry := ProfileCatalogEntry{
			Architecture: document.Architecture,
			Alias:        registeredProfileAlias(document.Architecture),
			Expected:     document.ID,
			Status:       ProfileCatalogMissing,
		}
		actual, found, err := reader.ResolveAlias(ctx, entry.Alias)
		if err != nil {
			return ProfileCatalogCoverage{}, err
		}
		if found {
			entry.Actual = artifact.IDPointer(actual)
			switch {
			case actual != document.ID:
				entry.Status = ProfileCatalogMismatched
			case actual == document.ID:
				loaded, loadErr := loadProfile(ctx, reader, actual)
				if loadErr != nil {
					entry.Status, entry.Detail = ProfileCatalogInvalid, loadErr.Error()
				} else if loaded.Architecture != document.Architecture || loaded.Policy.Name != document.Architecture {
					entry.Status, entry.Detail = ProfileCatalogInvalid, "profile architecture differs"
				} else {
					entry.Status = ProfileCatalogPublished
					coverage.Published++
				}
			}
		}
		coverage.Entries[index] = entry
	}
	coverage.Complete = coverage.Published == coverage.Registered
	return coverage, nil
}

func profileCatalogBatch(
	ctx context.Context,
	repository artifact.Repository,
	documents []ProfileDocument,
) (artifact.Batch, error) {
	batch := artifact.Batch{}
	digest := sha256.New()
	for _, document := range documents {
		contents, lineage, err := profilePublicationFacts(document)
		if err != nil {
			return artifact.Batch{}, err
		}
		batch.Contents = append(batch.Contents, contents...)
		batch.Lineage = append(batch.Lineage, lineage...)
		alias := registeredProfileAlias(document.Architecture)
		actual, found, err := repository.ResolveAlias(ctx, alias)
		if err != nil {
			return artifact.Batch{}, err
		}
		fmt.Fprintf(digest, "%s\x00%s\x00", alias, document.ID)
		binding := artifact.AliasBinding{Name: alias, Target: document.ID}
		if found {
			fmt.Fprintf(digest, "current=%s\x00", actual)
			if actual == document.ID {
				continue
			}
			binding.Previous = &actual
		} else {
			fmt.Fprint(digest, "missing\x00")
		}
		batch.Aliases = append(batch.Aliases, binding)
	}
	batch.Contents = compactProfileContents(batch.Contents)
	batch.Lineage = compactProfileLineage(batch.Lineage)
	batch.Key = fmt.Sprintf("profile/catalog/%x", digest.Sum(nil))
	return batch, nil
}

func compactProfileContents(contents []artifact.Content) []artifact.Content {
	slices.SortFunc(contents, func(left, right artifact.Content) int {
		return artifact.CompareID(left.Descriptor.ID, right.Descriptor.ID)
	})
	return slices.CompactFunc(contents, func(left, right artifact.Content) bool {
		return left.Descriptor.ID == right.Descriptor.ID
	})
}

func compactProfileLineage(lineage []artifact.Lineage) []artifact.Lineage {
	slices.SortFunc(lineage, func(left, right artifact.Lineage) int {
		if order := artifact.CompareID(left.Child, right.Child); order != 0 {
			return order
		}
		if order := artifact.CompareID(left.Parent, right.Parent); order != 0 {
			return order
		}
		return int(left.Relation) - int(right.Relation)
	})
	return slices.Compact(lineage)
}

func registeredProfileAlias(architecture string) string {
	return RegisteredProfileAliasPrefix + architecture
}
