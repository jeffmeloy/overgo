package overgodb

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/extent"
)

// DocumentOrder selects introduction chronology.
type DocumentOrder uint8

const (
	// DocumentOldestFirst visits introduction order.
	DocumentOldestFirst DocumentOrder = iota + 1
	// DocumentNewestFirst visits reverse introduction order.
	DocumentNewestFirst
)

// DocumentQuery selects one exact document contract.
type DocumentQuery struct {
	Contracts     []artifact.DocumentContract
	AliasPrefixes []string
	Order         DocumentOrder
	MaxResults    int
	Cursor        *QueryCursor
}

// DocumentView carries one streamed document and catalog facts.
type DocumentView struct {
	Content  artifact.Content
	Sequence uint64
	Aliases  []string
}

// DocumentPage reports streamed selection state.
type DocumentPage struct {
	Head      artifact.CommitID
	Sequence  uint64
	Truncated bool
	Next      *QueryCursor
}

// DocumentReader streams exact document projections.
type DocumentReader interface {
	VisitDocuments(context.Context, DocumentQuery, func(DocumentView) error) (DocumentPage, error)
}

// VisitDocuments streams exact-contract documents in introduction order.
func (s *Store) VisitDocuments(
	ctx context.Context,
	query DocumentQuery,
	visit func(DocumentView) error,
) (DocumentPage, error) {
	if err := validateDocumentQuery(query, visit); err != nil {
		return DocumentPage{}, err
	}
	if err := contextError(ctx); err != nil {
		return DocumentPage{}, err
	}
	s.mu.RLock()
	if err := s.ready(false); err != nil {
		s.mu.RUnlock()
		return DocumentPage{}, err
	}
	contract, err := documentQueryDigest(query)
	if err != nil {
		s.mu.RUnlock()
		return DocumentPage{}, err
	}
	if query.Cursor != nil && (query.Cursor.Head != s.head || query.Cursor.Contract != contract) {
		s.mu.RUnlock()
		return DocumentPage{}, errors.New("overgodb: document cursor is stale or belongs to another query")
	}
	page := DocumentPage{Head: s.head, Sequence: s.sequence}
	aliases := s.state.aliasesByTarget(query.AliasPrefixes)
	// The read walks only its contracts' schemas, in commit order, from the
	// cursor, and stops one match past the page: its cost follows the page,
	// not the history.
	window := s.state.artifacts.documentPositions(query.Contracts)
	newestFirst := query.Order == DocumentNewestFirst
	if query.Cursor != nil {
		record, found := s.state.artifacts.record(query.Cursor.After)
		if !found {
			s.mu.RUnlock()
			return DocumentPage{}, errors.New("overgodb: document cursor names no catalog record")
		}
		// The window holds what lies past the cursor in the read's order.
		at, on := slices.BinarySearch(window, record.position)
		switch {
		case newestFirst:
			window = window[:at]
		case on:
			// The cursor's own record was the previous page's last.
			window = window[at+extent.SingletonExtent:]
		default:
			window = window[at:]
		}
	}
	var entries []documentEntry
	var last artifact.ID
	var lastSequence uint64
	// take files one matching document and reports whether the page has
	// room for another look.
	take := func(position int) bool {
		entry, matched := s.state.documentAt(position, query, aliases)
		if !matched {
			return true
		}
		if query.MaxResults != 0 && len(entries) == query.MaxResults {
			page.Truncated = true
			return false
		}
		entries = append(entries, entry)
		last, lastSequence = entry.descriptor.ID, entry.sequence
		return true
	}
	if newestFirst {
		for _, position := range slices.Backward(window) {
			if !take(position) {
				break
			}
		}
	} else {
		for _, position := range window {
			if !take(position) {
				break
			}
		}
	}
	if page.Truncated {
		page.Next = &QueryCursor{
			Head: page.Head, Contract: contract, After: last, AfterSequence: lastSequence,
		}
	}
	s.mu.RUnlock()
	for _, entry := range entries {
		if err := contextError(ctx); err != nil {
			return DocumentPage{}, err
		}
		s.mu.RLock()
		if err := s.ready(false); err != nil {
			s.mu.RUnlock()
			return DocumentPage{}, err
		}
		data, err := s.materializeContent(entry.descriptor.ID, entry.locator)
		s.mu.RUnlock()
		if err != nil {
			return DocumentPage{}, err
		}
		if err := visit(DocumentView{
			Content:  artifact.Content{Descriptor: entry.descriptor, Data: data},
			Sequence: entry.sequence, Aliases: entry.aliases,
		}); err != nil {
			return DocumentPage{}, err
		}
	}
	return page, nil
}

type documentEntry struct {
	descriptor artifact.Descriptor
	locator    contentLocator
	sequence   uint64
	aliases    []string
}

// VisitDecodedDocuments decodes each streamed document before publication.
func VisitDecodedDocuments[T any](
	ctx context.Context,
	store DocumentReader,
	query DocumentQuery,
	decode func([]byte) (T, error),
	visit func(DocumentView, T) error,
) (DocumentPage, error) {
	if store == nil || decode == nil || visit == nil {
		return DocumentPage{}, errors.New("overgodb: invalid decoded document visitor")
	}
	return store.VisitDocuments(ctx, query, func(view DocumentView) error {
		value, err := decode(view.Content.Data)
		if err != nil {
			return err
		}
		return visit(view, value)
	})
}

// documentAt is the one match rule of a document read: the record at a
// bySequence position, when its content is held and live, a contract names
// it, and any alias prefix selects it.
func (s catalogState) documentAt(position int, query DocumentQuery, aliases map[artifact.ID][]string) (documentEntry, bool) {
	id := s.artifacts.bySequence[position]
	record, _ := s.artifacts.record(id)
	locator, hasContent := s.contents.locator(id)
	if !hasContent || locator.released != 0 || !matchesDocumentContract(record.descriptor, query.Contracts) {
		return documentEntry{}, false
	}
	names, selected := aliases[id]
	if len(query.AliasPrefixes) != 0 && !selected {
		return documentEntry{}, false
	}
	return documentEntry{descriptor: record.descriptor, locator: locator, sequence: record.sequence, aliases: slices.Clone(names)}, true
}

// CountDocuments is the explicit census of a document query: how many
// documents it matches in all, a walk of its schemas that paging never
// pays for.
func (s *Store) CountDocuments(ctx context.Context, query DocumentQuery) (int, error) {
	if err := validateDocumentQuery(query, func(DocumentView) error { return nil }); err != nil {
		return 0, err
	}
	if err := contextError(ctx); err != nil {
		return 0, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return 0, err
	}
	aliases := s.state.aliasesByTarget(query.AliasPrefixes)
	count := 0
	for _, position := range s.state.artifacts.documentPositions(query.Contracts) {
		if _, matched := s.state.documentAt(position, query, aliases); matched {
			count++
		}
	}
	return count, nil
}

// documentPositions returns the ascending bySequence positions of every
// record whose schema one of the contracts names; the caller only reads the
// slice.
func (f artifactFacet) documentPositions(contracts []artifact.DocumentContract) []int {
	var schemas []string
	for _, contract := range contracts {
		if !slices.Contains(schemas, contract.Schema) {
			schemas = append(schemas, contract.Schema)
		}
	}
	if len(schemas) == 1 {
		return f.schemaPositions[schemas[0]]
	}
	var positions []int
	for _, schema := range schemas {
		positions = append(positions, f.schemaPositions[schema]...)
	}
	slices.Sort(positions)
	return positions
}

func validateDocumentQuery(query DocumentQuery, visit func(DocumentView) error) error {
	if len(query.Contracts) == 0 {
		return errors.New("overgodb: document query requires an exact contract")
	}
	for index, contract := range query.Contracts {
		if err := contract.Validate(); err != nil {
			return err
		}
		if slices.Contains(query.Contracts[:index], contract) {
			return errors.New("overgodb: duplicate document query contract")
		}
	}
	if query.Order != DocumentOldestFirst && query.Order != DocumentNewestFirst {
		return errors.New("overgodb: invalid document order")
	}
	if query.MaxResults < 0 || visit == nil {
		return errors.New("overgodb: invalid document query bound or visitor")
	}
	for index, prefix := range query.AliasPrefixes {
		if prefix == "" || strings.TrimSpace(prefix) != prefix || strings.ContainsAny(prefix, "\r\n") ||
			slices.Contains(query.AliasPrefixes[:index], prefix) {
			return errors.New("overgodb: invalid document alias prefixes")
		}
	}
	if query.Cursor != nil && (!query.Cursor.After.Valid() || query.Cursor.AfterSequence == 0) {
		return errors.New("overgodb: invalid document cursor")
	}
	return nil
}

func documentQueryDigest(query DocumentQuery) ([sha256.Size]byte, error) {
	payload, err := json.Marshal(struct {
		Contracts     []artifact.DocumentContract
		AliasPrefixes []string
		Order         DocumentOrder
	}{query.Contracts, query.AliasPrefixes, query.Order})
	if err != nil {
		return [sha256.Size]byte{}, err
	}
	return sha256.Sum256(payload), nil
}

func matchesDocumentContract(descriptor artifact.Descriptor, contracts []artifact.DocumentContract) bool {
	return slices.ContainsFunc(contracts, func(contract artifact.DocumentContract) bool {
		return descriptor.ID.Kind() == contract.Kind && descriptor.MediaType == contract.MediaType && descriptor.Schema == contract.Schema
	})
}

func (s catalogState) aliasesByTarget(prefixes []string) map[artifact.ID][]string {
	if len(prefixes) == 0 {
		return nil
	}
	result := map[artifact.ID][]string{}
	s.aliases.each(func(name string, target artifact.ID) {
		if slices.ContainsFunc(prefixes, func(prefix string) bool { return strings.HasPrefix(name, prefix) }) {
			result[target] = append(result[target], name)
		}
	})
	for target := range result {
		slices.Sort(result[target])
	}
	return result
}

// VisitAliases streams aliases in name order.
func (s *Store) VisitAliases(ctx context.Context, prefix string, visit func(AliasView) error) error {
	if visit == nil || strings.TrimSpace(prefix) != prefix || strings.ContainsAny(prefix, "\r\n") {
		return errors.New("overgodb: invalid alias visitor")
	}
	if err := contextError(ctx); err != nil {
		return err
	}
	s.mu.RLock()
	if err := s.ready(false); err != nil {
		s.mu.RUnlock()
		return err
	}
	aliases := s.state.aliasViews(prefix)
	s.mu.RUnlock()
	for _, alias := range aliases {
		if err := visit(alias); err != nil {
			return err
		}
	}
	return nil
}

func (s catalogState) aliasViews(prefix string) []AliasView {
	names := make([]string, 0, s.aliases.count())
	s.aliases.each(func(name string, _ artifact.ID) {
		if strings.HasPrefix(name, prefix) {
			names = append(names, name)
		}
	})
	slices.Sort(names)
	aliases := make([]AliasView, len(names))
	for index, name := range names {
		target, _ := s.aliases.resolve(name)
		aliases[index] = AliasView{Name: name, Target: target}
	}
	return aliases
}
