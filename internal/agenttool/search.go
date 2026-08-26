package agenttool

import (
	"context"
	"errors"
	"slices"
	"sort"
	"strings"
	"unicode"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
)

const (
	// CatalogSnapshotMediaType identifies encoded tool-catalog snapshot documents.
	CatalogSnapshotMediaType = "application/vnd.overgo.agent-tool-catalog+json"
	// CatalogSnapshotSchema identifies the exact stored tool-catalog snapshot schema.
	CatalogSnapshotSchema = "overgo/agent-tool-catalog/v1"

	catalogSearchMaxQueryBytes = 4096
	catalogSearchMaxResults    = 256
	catalogMaxEntries          = 4096
	catalogExactNameWeight     = 64
	catalogNameTermWeight      = 8
	catalogFieldTermWeight     = 4
	catalogDescriptionWeight   = 1
)

// CatalogEntry is the exact context needed to select a manual without loading
// every full document into a model prompt. Manual remains the execution
// authority; this is a content-identified compiled projection of it.
type CatalogEntry struct {
	Name        string      `json:"name"`
	Manual      artifact.ID `json:"manual"`
	Description string      `json:"description"`
	Effect      Effect      `json:"effect"`
	Arguments   []Field     `json:"arguments,omitempty"`
}

// CatalogSnapshot is one immutable, canonically ordered manual catalog.
type CatalogSnapshot struct {
	Version uint16         `json:"version"`
	Entries []CatalogEntry `json:"entries"`
	ID      artifact.ID    `json:"-"`
}

// CatalogSearchResult identifies one bounded deterministic match.
type CatalogSearchResult struct {
	Name   string      `json:"name"`
	Manual artifact.ID `json:"manual"`
	Effect Effect      `json:"effect"`
	Score  int         `json:"score"`
}

var catalogSnapshotCodec = artifact.JSONDocumentCodec(
	"agent tool catalog", artifact.KindProfile, CatalogSnapshotMediaType, CatalogSnapshotSchema,
	canonicalizeCatalogSnapshot,
	func(value CatalogSnapshot) artifact.ID { return value.ID },
	func(value *CatalogSnapshot, id artifact.ID) { value.ID = id },
	cloneCatalogSnapshot,
)

// NewCatalogSnapshot compiles exact manuals into one searchable catalog.
func NewCatalogSnapshot(manuals []Manual) (CatalogSnapshot, error) {
	if err := validateManualSet(manuals); err != nil {
		return CatalogSnapshot{}, err
	}
	entries := make([]CatalogEntry, len(manuals))
	for index, manual := range manuals {
		entries[index] = CatalogEntry{
			Name: manual.Name, Manual: manual.ID, Description: manual.Description,
			Effect: manual.Effect, Arguments: slices.Clone(manual.Arguments),
		}
	}
	return catalogSnapshotCodec.New(CatalogSnapshot{
		Version: artifact.InitialDocumentVersion, Entries: entries,
	})
}

// RequireCatalogSnapshot loads one exact compiled catalog.
func RequireCatalogSnapshot(ctx context.Context, reader artifact.Reader, id artifact.ID) (CatalogSnapshot, error) {
	return catalogSnapshotCodec.Require(ctx, reader, id)
}

// ArtifactContent returns the snapshot's canonical repository bytes.
func (snapshot CatalogSnapshot) ArtifactContent() (artifact.Content, error) {
	return catalogSnapshotCodec.Content(snapshot)
}

// Search returns at most limit matches. Ranking is a pure function of the
// snapshot and query; ties resolve by manual name and content identity.
func (snapshot CatalogSnapshot) Search(query string, limit int) ([]CatalogSearchResult, error) {
	if err := catalogSnapshotCodec.ValidateIdentity(snapshot); err != nil {
		return nil, err
	}
	if !textcheck.Bounded(query, catalogSearchMaxQueryBytes, "\x00\r\n") || limit <= 0 || limit > catalogSearchMaxResults {
		return nil, errors.New("agent tool: catalog search request exceeds its bound")
	}
	query = strings.ToLower(strings.TrimSpace(query))
	terms := catalogTerms(query)
	if len(terms) == 0 {
		return nil, errors.New("agent tool: catalog search query has no terms")
	}
	results := make([]CatalogSearchResult, 0, min(limit, len(snapshot.Entries)))
	for _, entry := range snapshot.Entries {
		score := catalogEntryScore(entry, query, terms)
		if score == 0 {
			continue
		}
		results = append(results, CatalogSearchResult{
			Name: entry.Name, Manual: entry.Manual, Effect: entry.Effect, Score: score,
		})
	}
	sort.Slice(results, func(left, right int) bool {
		if results[left].Score != results[right].Score {
			return results[left].Score > results[right].Score
		}
		if results[left].Name != results[right].Name {
			return results[left].Name < results[right].Name
		}
		return artifact.CompareID(results[left].Manual, results[right].Manual) < 0
	})
	return slices.Clone(results[:min(limit, len(results))]), nil
}

func canonicalizeCatalogSnapshot(snapshot *CatalogSnapshot) error {
	if snapshot == nil || snapshot.Version != artifact.InitialDocumentVersion ||
		len(snapshot.Entries) == 0 || len(snapshot.Entries) > catalogMaxEntries {
		return errors.New("agent tool: invalid catalog snapshot")
	}
	for index := range snapshot.Entries {
		entry := &snapshot.Entries[index]
		if !manualNamePattern.MatchString(entry.Name) || entry.Manual.Kind() != artifact.KindRecipe ||
			entry.Description == "" || !textcheck.Bounded(entry.Description, len(entry.Description), "\x00\r\n") ||
			!entry.Effect.Valid() {
			return errors.New("agent tool: invalid catalog entry")
		}
		entry.Arguments = slices.Clone(entry.Arguments)
		for _, field := range entry.Arguments {
			if !manualNamePattern.MatchString(field.Name) || !field.Kind.Valid() ||
				field.Description != "" && !textcheck.Bounded(field.Description, len(field.Description), "\x00\r\n") {
				return errors.New("agent tool: invalid catalog field")
			}
		}
		sort.Slice(entry.Arguments, func(left, right int) bool {
			return entry.Arguments[left].Name < entry.Arguments[right].Name
		})
		if duplicateCatalogFields(entry.Arguments) {
			return errors.New("agent tool: duplicate catalog field")
		}
	}
	sort.Slice(snapshot.Entries, func(left, right int) bool {
		return snapshot.Entries[left].Name < snapshot.Entries[right].Name
	})
	seenEntries := make(map[string]bool, len(snapshot.Entries))
	for _, entry := range snapshot.Entries {
		if seenEntries[entry.Name] {
			return errors.New("agent tool: duplicate catalog entry")
		}
		seenEntries[entry.Name] = true
	}
	return nil
}

func cloneCatalogSnapshot(snapshot CatalogSnapshot) CatalogSnapshot {
	snapshot.Entries = slices.Clone(snapshot.Entries)
	for index := range snapshot.Entries {
		snapshot.Entries[index].Arguments = slices.Clone(snapshot.Entries[index].Arguments)
	}
	return snapshot
}

func duplicateCatalogFields(fields []Field) bool {
	seen := make(map[string]bool, len(fields))
	for _, field := range fields {
		if seen[field.Name] {
			return true
		}
		seen[field.Name] = true
	}
	return false
}

func catalogEntryScore(entry CatalogEntry, query string, terms []string) int {
	name := strings.ToLower(entry.Name)
	if name == query {
		return catalogExactNameWeight
	}
	nameTerms := termSet(catalogTerms(name))
	fieldTerms := map[string]bool{}
	for _, field := range entry.Arguments {
		for _, term := range catalogTerms(strings.ToLower(field.Name + " " + field.Description)) {
			fieldTerms[term] = true
		}
	}
	descriptionTerms := termSet(catalogTerms(strings.ToLower(entry.Description)))
	score := 0
	for _, term := range terms {
		switch {
		case nameTerms[term]:
			score += catalogNameTermWeight
		case fieldTerms[term]:
			score += catalogFieldTermWeight
		case descriptionTerms[term]:
			score += catalogDescriptionWeight
		}
	}
	return score
}

func catalogTerms(value string) []string {
	terms := strings.FieldsFunc(value, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	slices.Sort(terms)
	return slices.Compact(terms)
}

func termSet(terms []string) map[string]bool {
	set := make(map[string]bool, len(terms))
	for _, term := range terms {
		set[term] = true
	}
	return set
}
