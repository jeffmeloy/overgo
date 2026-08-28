package overgodb

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

const causalityProjectionVersion = initialProjectionVersion

// CausalityQuery selects one bounded view of the operational causal graph.
// Filters intersect. DescendantOf traverses causal subjects, independently of
// artifact lineage; MaxDepth bounds that traversal.
type CausalityQuery struct {
	Execution    *artifact.ID
	Root         *artifact.ID
	DescendantOf *artifact.ID
	Evidence     *artifact.ID
	Trigger      string
	MaxDepth     uint32
	MaxResults   int
}

// CausalityResult names the exact journal view and projection schema used to
// answer a causal query.
type CausalityResult struct {
	Head              artifact.CommitID     `json:"head"`
	Sequence          uint64                `json:"sequence"`
	ProjectionVersion uint16                `json:"projection_version"`
	Matched           int                   `json:"matched"`
	Links             []artifact.CausalLink `json:"links,omitempty"`
	Truncated         bool                  `json:"truncated,omitempty"`
}

type causalityFacet struct {
	records    map[artifact.ID]artifact.CausalLink
	byRoot     map[artifact.ID][]artifact.ID
	children   map[artifact.ID][]artifact.ID
	byEvidence map[artifact.ID][]artifact.ID
	byTrigger  map[string][]artifact.ID
}

func newCausalityFacet() causalityFacet {
	return causalityFacet{
		records: map[artifact.ID]artifact.CausalLink{}, byRoot: map[artifact.ID][]artifact.ID{},
		children: map[artifact.ID][]artifact.ID{}, byEvidence: map[artifact.ID][]artifact.ID{},
		byTrigger: map[string][]artifact.ID{},
	}
}

func (f causalityFacet) validate(batch artifact.Batch, hasArtifact func(artifact.ID) bool) error {
	pending := make(map[artifact.ID]artifact.CausalLink, len(batch.Causality))
	for _, link := range batch.Causality {
		if err := link.Validate(); err != nil {
			return err
		}
		if current, found := f.records[link.Execution]; found && !equalCausalLink(current, link) {
			return fmt.Errorf("overgodb: causal execution conflicts: %s", link.Execution)
		}
		if current, found := pending[link.Execution]; found && !equalCausalLink(current, link) {
			return fmt.Errorf("overgodb: duplicate causal execution: %s", link.Execution)
		}
		for _, required := range causalEvidence(link) {
			if !hasArtifact(required) {
				return fmt.Errorf("overgodb: causal evidence is unknown: %s", required)
			}
		}
		pending[link.Execution] = link
	}
	lookup := func(id artifact.ID) (artifact.CausalLink, bool) {
		if link, found := pending[id]; found {
			return link, true
		}
		link, found := f.records[id]
		return link, found
	}
	for _, link := range pending {
		if !link.Subject.Valid() {
			continue
		}
		if parent, found := lookup(link.Subject); found && parent.Root != link.Root {
			return errors.New("overgodb: causal descendant changes root")
		}
		seen := map[artifact.ID]bool{link.Execution: true}
		for current := link.Subject; current.Valid(); {
			if seen[current] {
				return errors.New("overgodb: causal cycle")
			}
			seen[current] = true
			parent, found := lookup(current)
			if !found {
				break
			}
			current = parent.Subject
		}
	}
	return nil
}

func (f *causalityFacet) add(link artifact.CausalLink) {
	if _, found := f.records[link.Execution]; found {
		return
	}
	link = link.Clone()
	f.records[link.Execution] = link
	insertCausalIndex(f.byRoot, link.Root, link.Execution)
	insertCausalIndex(f.byTrigger, link.Trigger, link.Execution)
	if link.Subject.Valid() {
		insertCausalIndex(f.children, link.Subject, link.Execution)
	}
	for _, evidence := range causalEvidence(link) {
		insertCausalIndex(f.byEvidence, evidence, link.Execution)
	}
}

func (f *causalityFacet) accept(batch artifact.Batch, _ map[artifact.ID]contentLocator, sequence uint64) error {
	if len(batch.Causality) != 0 && sequence == 0 {
		return errors.New("overgodb: causal projection requires a commit sequence")
	}
	return nil
}

func (f *causalityFacet) applyCommit(batch artifact.Batch, _ map[artifact.ID]contentLocator, _ uint64) {
	for _, link := range batch.Causality {
		f.add(link)
	}
}

func (f causalityFacet) delta(batch artifact.Batch, delta *artifact.Batch) {
	for _, link := range batch.Causality {
		if current, found := f.records[link.Execution]; !found || !equalCausalLink(current, link) {
			delta.Causality = append(delta.Causality, link.Clone())
		}
	}
}

func (f *causalityFacet) checkpoint() ([]byte, error) {
	links := make([]artifact.CausalLink, 0, len(f.records))
	for _, link := range f.records {
		links = append(links, link.Clone())
	}
	sort.Slice(links, func(i, j int) bool { return artifact.CompareID(links[i].Execution, links[j].Execution) < 0 })
	return json.Marshal(links)
}

func (f *causalityFacet) restore(data []byte) error {
	var links []artifact.CausalLink
	if err := strictjson.DecodeBytes(data, &links); err != nil {
		return fmt.Errorf("causality checkpoint: %w", err)
	}
	*f = newCausalityFacet()
	for index, link := range links {
		if err := link.Validate(); err != nil || index > 0 && artifact.CompareID(links[index-1].Execution, link.Execution) >= 0 {
			return errors.New("causality checkpoint: invalid link order")
		}
		f.add(link)
	}
	return nil
}

// QueryCausality evaluates a bounded causal projection read.
func (s *Store) QueryCausality(ctx context.Context, query CausalityQuery) (CausalityResult, error) {
	if err := validateCausalityQuery(query); err != nil {
		return CausalityResult{}, err
	}
	if err := contextError(ctx); err != nil {
		return CausalityResult{}, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return CausalityResult{}, err
	}
	ids := s.state.causality.selectIDs(query)
	result := CausalityResult{
		Head: s.head, Sequence: s.sequence, ProjectionVersion: causalityProjectionVersion,
		Matched: len(ids), Truncated: query.MaxResults < len(ids),
	}
	ids = ids[:min(len(ids), query.MaxResults)]
	for _, id := range ids {
		result.Links = append(result.Links, s.state.causality.records[id].Clone())
	}
	return result, nil
}

func validateCausalityQuery(query CausalityQuery) error {
	if query.MaxResults < 0 {
		return errors.New("overgodb: causal query result bound is invalid")
	}
	for _, id := range []*artifact.ID{query.Execution, query.Root, query.DescendantOf, query.Evidence} {
		if id != nil && id.Kind() != artifact.KindEvidence {
			return errors.New("overgodb: causal query requires evidence identities")
		}
	}
	if query.Trigger != "" && (strings.TrimSpace(query.Trigger) != query.Trigger || strings.ContainsAny(query.Trigger, " /\x00\r\n\t")) {
		return errors.New("overgodb: causal query trigger is invalid")
	}
	if query.DescendantOf != nil && query.MaxDepth == 0 {
		return errors.New("overgodb: causal descendant depth is invalid")
	}
	return nil
}

func (f causalityFacet) selectIDs(query CausalityQuery) []artifact.ID {
	var candidates []artifact.ID
	switch {
	case query.Execution != nil:
		if _, found := f.records[*query.Execution]; found {
			candidates = []artifact.ID{*query.Execution}
		}
	case query.DescendantOf != nil:
		candidates = f.descendants(*query.DescendantOf, query.MaxDepth)
	case query.Root != nil:
		candidates = slices.Clone(f.byRoot[*query.Root])
	case query.Evidence != nil:
		candidates = slices.Clone(f.byEvidence[*query.Evidence])
	case query.Trigger != "":
		candidates = slices.Clone(f.byTrigger[query.Trigger])
	default:
		candidates = make([]artifact.ID, 0, len(f.records))
		for id := range f.records {
			candidates = append(candidates, id)
		}
		slices.SortFunc(candidates, artifact.CompareID)
	}
	return slices.DeleteFunc(candidates, func(id artifact.ID) bool {
		link := f.records[id]
		return query.Root != nil && link.Root != *query.Root ||
			query.Evidence != nil && !slices.Contains(causalEvidence(link), *query.Evidence) ||
			query.Trigger != "" && link.Trigger != query.Trigger
	})
}

func (f causalityFacet) descendants(root artifact.ID, maxDepth uint32) []artifact.ID {
	seen := map[artifact.ID]bool{root: true}
	frontier := []artifact.ID{root}
	var result []artifact.ID
	for depth := uint32(0); depth < maxDepth && len(frontier) != 0; depth++ {
		var next []artifact.ID
		for _, parent := range frontier {
			for _, child := range f.children[parent] {
				if !seen[child] {
					seen[child] = true
					result, next = append(result, child), append(next, child)
				}
			}
		}
		frontier = next
	}
	slices.SortFunc(result, artifact.CompareID)
	return result
}

func causalEvidence(link artifact.CausalLink) []artifact.ID {
	result := []artifact.ID{link.Root}
	if link.Subject.Valid() {
		result = append(result, link.Subject)
	}
	result = append(result, link.Motivation...)
	slices.SortFunc(result, artifact.CompareID)
	return slices.Compact(result)
}

func insertCausalIndex[K comparable](index map[K][]artifact.ID, key K, id artifact.ID) {
	values := index[key]
	at, found := slices.BinarySearchFunc(values, id, artifact.CompareID)
	if !found {
		index[key] = slices.Insert(values, at, id)
	}
}

func equalCausalLink(left, right artifact.CausalLink) bool {
	return left.Execution == right.Execution && left.Root == right.Root && left.Trigger == right.Trigger &&
		left.Subject == right.Subject && slices.Equal(left.Motivation, right.Motivation)
}
