package overgodb

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
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
	Matched   int
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
	entries := make([]documentEntry, 0, min(query.MaxResults, len(s.state.bySequence)))
	var last artifactSlotCursor
	emitted := 0
	for offset := range s.state.bySequence {
		index := offset
		if query.Order == DocumentNewestFirst {
			index = len(s.state.bySequence) - offset - 1
		}
		id := s.state.bySequence[index]
		slot := s.state.slots[id]
		if !slot.hasContent || !matchesDocumentContract(slot.descriptor, query.Contracts) {
			continue
		}
		names, selected := aliases[id]
		if len(query.AliasPrefixes) != 0 && !selected {
			continue
		}
		page.Matched++
		cursor := artifactSlotCursor{sequence: slot.sequence, id: id}
		if query.Cursor != nil && !cursor.follows(*query.Cursor, query.Order) {
			continue
		}
		if query.MaxResults != 0 && emitted >= query.MaxResults {
			page.Truncated = true
			continue
		}
		entries = append(entries, documentEntry{
			descriptor: slot.descriptor, locator: slot.content, sequence: slot.sequence, aliases: slices.Clone(names),
		})
		last = cursor
		emitted++
	}
	if page.Truncated {
		page.Next = &QueryCursor{
			Head: page.Head, Contract: contract, After: last.id, AfterSequence: last.sequence,
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
		data, err := s.materializeContent(entry.locator)
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

type artifactSlotCursor struct {
	sequence uint64
	id       artifact.ID
}

func (c artifactSlotCursor) follows(after QueryCursor, order DocumentOrder) bool {
	if order == DocumentOldestFirst {
		return c.sequence > after.AfterSequence ||
			c.sequence == after.AfterSequence && artifact.CompareID(c.id, after.After) > 0
	}
	return c.sequence < after.AfterSequence ||
		c.sequence == after.AfterSequence && artifact.CompareID(c.id, after.After) < 0
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
	for name, target := range s.aliases {
		if slices.ContainsFunc(prefixes, func(prefix string) bool { return strings.HasPrefix(name, prefix) }) {
			result[target] = append(result[target], name)
		}
	}
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
	names := make([]string, 0, len(s.aliases))
	for name := range s.aliases {
		if strings.HasPrefix(name, prefix) {
			names = append(names, name)
		}
	}
	slices.Sort(names)
	aliases := make([]AliasView, len(names))
	for index, name := range names {
		aliases[index] = AliasView{Name: name, Target: s.aliases[name]}
	}
	return aliases
}
