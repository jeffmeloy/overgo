package agenttool

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

// RegisteredAliasPrefix scopes current tool manual bindings.
const RegisteredAliasPrefix = "tool.registered."

// CatalogEntryStatus classifies one registered manual binding.
type CatalogEntryStatus string

const (
	// CatalogEntryPublished marks an exact, valid binding.
	CatalogEntryPublished CatalogEntryStatus = "published"
	// CatalogEntryMissing marks an absent binding.
	CatalogEntryMissing CatalogEntryStatus = "missing"
	// CatalogEntryMismatched marks an unexpected target.
	CatalogEntryMismatched CatalogEntryStatus = "mismatched"
	// CatalogEntryInvalid marks bound content that fails to parse back.
	CatalogEntryInvalid CatalogEntryStatus = "invalid"
)

// CatalogCoverageEntry reports one manual binding.
type CatalogCoverageEntry struct {
	Name     string             `json:"name"`
	Alias    string             `json:"alias"`
	Effect   Effect             `json:"effect"`
	Expected artifact.ID        `json:"expected"`
	Actual   *artifact.ID       `json:"actual,omitempty"`
	Status   CatalogEntryStatus `json:"status"`
	Detail   string             `json:"detail,omitzero"`
}

// CatalogCoverage reports registry-to-store parity for tool manuals.
type CatalogCoverage struct {
	Registered int                    `json:"registered"`
	Published  int                    `json:"published"`
	Complete   bool                   `json:"complete"`
	Entries    []CatalogCoverageEntry `json:"entries"`
}

// CatalogPublication reports one atomic manual catalog publication.
type CatalogPublication struct {
	Commit   artifact.CommitID `json:"commit"`
	Changed  bool              `json:"changed"`
	Coverage CatalogCoverage   `json:"coverage"`
}

// RegisteredAlias names the store binding for one tool manual.
func RegisteredAlias(name string) string {
	return RegisteredAliasPrefix + name
}

// InspectManualCatalog compares the supplied manual set with the store.
func InspectManualCatalog(ctx context.Context, reader artifact.Reader, manuals []Manual) (CatalogCoverage, error) {
	if ctx == nil || reader == nil {
		return CatalogCoverage{}, errors.New("agent tool: nil catalog context or reader")
	}
	coverage := CatalogCoverage{Registered: len(manuals), Entries: make([]CatalogCoverageEntry, len(manuals))}
	for index, manual := range manuals {
		entry := CatalogCoverageEntry{
			Name: manual.Name, Alias: RegisteredAlias(manual.Name),
			Effect: manual.Effect, Expected: manual.ID, Status: CatalogEntryMissing,
		}
		actual, found, err := reader.ResolveAlias(ctx, entry.Alias)
		if err != nil {
			return CatalogCoverage{}, err
		}
		if found {
			entry.Actual = artifact.IDPointer(actual)
			switch {
			case actual != manual.ID:
				entry.Status = CatalogEntryMismatched
			default:
				bound, loadErr := RequireManual(ctx, reader, actual)
				if loadErr != nil {
					entry.Status, entry.Detail = CatalogEntryInvalid, loadErr.Error()
				} else if bound.Name != manual.Name {
					entry.Status, entry.Detail = CatalogEntryInvalid, "bound manual names a different tool"
				} else {
					entry.Status = CatalogEntryPublished
					coverage.Published++
				}
			}
		}
		coverage.Entries[index] = entry
	}
	coverage.Complete = coverage.Published == coverage.Registered
	return coverage, nil
}

// PublishManualCatalog atomically publishes the manual set as store
// authority: contents committed, aliases compare-and-set, idempotent
// when the store already binds every manual exactly.
func PublishManualCatalog(ctx context.Context, repository artifact.Repository, manuals []Manual) (CatalogPublication, error) {
	if ctx == nil || repository == nil {
		return CatalogPublication{}, errors.New("agent tool: nil catalog context or repository")
	}
	if err := validateManualSet(manuals); err != nil {
		return CatalogPublication{}, err
	}
	for _, manual := range manuals {
		if err := CheckArgvAuthority(ctx, repository, manual); err != nil {
			return CatalogPublication{}, err
		}
		if manual.Capability.Valid() && manual.CapabilityIdentity == nil {
			if _, err := runrecord.RequireCapabilityIdentity(ctx, repository, manual.Capability); err != nil {
				return CatalogPublication{}, err
			}
		}
	}
	coverage, err := InspectManualCatalog(ctx, repository, manuals)
	if err != nil {
		return CatalogPublication{}, err
	}
	if coverage.Complete {
		commit, _ := repository.Head()
		return CatalogPublication{Commit: commit, Coverage: coverage}, nil
	}
	batch, err := manualCatalogBatch(ctx, repository, manuals)
	if err != nil {
		return CatalogPublication{}, err
	}
	commit, err := artifact.CommitBatch(ctx, repository, batch)
	if err != nil {
		return CatalogPublication{}, err
	}
	coverage, err = InspectManualCatalog(ctx, repository, manuals)
	if err != nil {
		return CatalogPublication{}, err
	}
	if !coverage.Complete {
		return CatalogPublication{}, errors.New("agent tool: manual catalog publication is incomplete")
	}
	return CatalogPublication{Commit: commit, Changed: true, Coverage: coverage}, nil
}

// LoadManual reads one committed manual by its exact immutable
// identity -- the replay path, where the mutable name alias must
// LoadManual reads one manual document by its exact identity, so a
// caller replaying a recorded step resolves the tool it actually ran
// rather than whatever the name alias points at now.
func LoadManual(ctx context.Context, reader artifact.Reader, id artifact.ID) (Manual, error) {
	if ctx == nil || reader == nil {
		return Manual{}, errors.New("agent tool: nil load context or reader")
	}
	return RequireManual(ctx, reader, id)
}

// ResolveRegisteredManual requires the store-published manual currently
// bound to name, refusing an absent or mistargeted alias.
func ResolveRegisteredManual(ctx context.Context, reader artifact.Reader, name string) (Manual, error) {
	if ctx == nil || reader == nil {
		return Manual{}, errors.New("agent tool: nil resolve context or reader")
	}
	alias := RegisteredAlias(name)
	target, found, err := reader.ResolveAlias(ctx, alias)
	if err != nil {
		return Manual{}, err
	}
	if !found {
		return Manual{}, fmt.Errorf("agent tool: registered alias %q is absent", alias)
	}
	manual, err := RequireManual(ctx, reader, target)
	if err != nil {
		return Manual{}, err
	}
	if manual.Name != name {
		return Manual{}, fmt.Errorf("agent tool: registered alias %q binds a different tool", alias)
	}
	active, activeFound, err := reader.ResolveAlias(ctx, ActiveCatalogAlias)
	if err != nil {
		return Manual{}, err
	}
	if activeFound {
		snapshot, err := RequireCatalogSnapshot(ctx, reader, active)
		if err != nil {
			return Manual{}, err
		}
		admitted := false
		for _, entry := range snapshot.Entries {
			if entry.Name == name && entry.Manual == manual.ID {
				admitted = true
				break
			}
		}
		if !admitted {
			return Manual{}, fmt.Errorf("agent tool: registered alias %q is outside the active catalog", alias)
		}
	}
	return manual, nil
}

func validateManualSet(manuals []Manual) error {
	if len(manuals) == 0 {
		return errors.New("agent tool: manual catalog is empty")
	}
	seen := map[string]bool{}
	for _, manual := range manuals {
		if err := manualCodec.ValidateIdentity(manual); err != nil {
			return err
		}
		if seen[manual.Name] {
			return fmt.Errorf("agent tool: duplicate manual %q", manual.Name)
		}
		seen[manual.Name] = true
	}
	return nil
}

func manualCatalogBatch(
	ctx context.Context,
	repository artifact.Repository,
	manuals []Manual,
) (artifact.Batch, error) {
	batch := artifact.Batch{}
	digest := sha256.New()
	capabilities := map[artifact.ID]bool{}
	for _, manual := range manuals {
		if err := appendManualDocuments(&batch.Contents, &batch.Lineage, manual, capabilities); err != nil {
			return artifact.Batch{}, err
		}
		alias := RegisteredAlias(manual.Name)
		actual, found, err := repository.ResolveAlias(ctx, alias)
		if err != nil {
			return artifact.Batch{}, err
		}
		fmt.Fprintf(digest, "%s\x00%s\x00", alias, manual.ID)
		binding := artifact.AliasBinding{Name: alias, Target: manual.ID}
		if found {
			fmt.Fprintf(digest, "current=%s\x00", actual)
			if actual == manual.ID {
				continue
			}
			binding.Previous = &actual
		} else {
			fmt.Fprint(digest, "missing\x00")
		}
		batch.Aliases = append(batch.Aliases, binding)
	}
	batch.Key = fmt.Sprintf("agent-tool/catalog/%x", digest.Sum(nil))
	return batch, nil
}
