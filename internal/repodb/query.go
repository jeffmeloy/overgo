package repodb

import (
	"context"
	"errors"
	"sort"
	"strings"

	"overgo/internal/artifact"
)

const MaxQueryResults = 100_000

type FollowDirection uint8

const (
	FollowNone FollowDirection = iota
	FollowParents
	FollowChildren
	FollowBoth
)

type Query struct {
	Kind         artifact.Kind
	MediaType    string
	Schema       string
	Artifact     *artifact.ID
	Alias        string
	Relation     artifact.Relation
	Follow       FollowDirection
	MaxDepth     uint32
	MaxResults   int
	FromSequence uint64
	ToSequence   uint64
}

type AliasView struct {
	Name   string      `json:"name"`
	Target artifact.ID `json:"target"`
}

type CommitView struct {
	Key      string            `json:"key"`
	ID       artifact.CommitID `json:"id"`
	Sequence uint64            `json:"sequence"`
}

type QueryResult struct {
	Head      artifact.CommitID     `json:"head"`
	Sequence  uint64                `json:"sequence"`
	Artifacts []artifact.Descriptor `json:"artifacts,omitempty"`
	Manifests []artifact.Manifest   `json:"manifests,omitempty"`
	Aliases   []AliasView           `json:"aliases,omitempty"`
	Lineage   []artifact.Lineage    `json:"lineage,omitempty"`
	Commits   []CommitView          `json:"commits,omitempty"`
	Truncated bool                  `json:"truncated,omitempty"`
}

func (s *Store) Query(ctx context.Context, query Query) (QueryResult, error) {
	if err := validateQuery(query); err != nil {
		return QueryResult{}, err
	}
	if err := contextError(ctx); err != nil {
		return QueryResult{}, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return QueryResult{}, err
	}
	result := QueryResult{Head: s.head, Sequence: s.sequence}
	selected, edges, truncated, err := s.state.querySelection(query)
	if err != nil {
		return QueryResult{}, err
	}
	result.Truncated = truncated
	ids := make([]artifact.ID, 0, len(selected))
	for id := range selected {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i].String() < ids[j].String() })
	for _, id := range ids {
		if descriptor, ok := s.state.artifacts[id]; ok {
			result.Artifacts = append(result.Artifacts, descriptor)
		}
		if manifest, ok := s.state.manifests[id]; ok {
			result.Manifests = append(result.Manifests, manifest.Clone())
		}
	}
	result.Aliases = s.state.queryAliases(query, selected, &result.Truncated)
	result.Lineage = sortedQueryLineage(edges, query.MaxResults, &result.Truncated)
	result.Commits = s.state.queryCommits(query, &result.Truncated)
	return result, nil
}

func validateQuery(query Query) error {
	if query.MaxResults <= 0 || query.MaxResults > MaxQueryResults {
		return errors.New("repodb: query result bound is invalid")
	}
	if query.Kind != artifact.KindInvalid {
		if _, err := artifact.ParseKind(query.Kind.String()); err != nil {
			return err
		}
	}
	if !validDescriptorFilter(query.MediaType) || !validDescriptorFilter(query.Schema) {
		return errors.New("repodb: query descriptor filter is invalid")
	}
	if query.Artifact != nil && !query.Artifact.Valid() {
		return errors.New("repodb: query artifact is invalid")
	}
	if query.Alias != "" && (strings.TrimSpace(query.Alias) != query.Alias || strings.ContainsAny(query.Alias, "\r\n")) {
		return errors.New("repodb: query alias is invalid")
	}
	if query.Relation != artifact.RelationInvalid {
		if _, err := artifact.ParseRelation(query.Relation.String()); err != nil {
			return err
		}
	}
	if query.Follow > FollowBoth || query.Follow != FollowNone && query.Artifact == nil && query.Alias == "" {
		return errors.New("repodb: lineage follow requires one artifact or alias")
	}
	if query.Follow != FollowNone && query.MaxDepth == 0 {
		return errors.New("repodb: lineage depth bound is invalid")
	}
	if query.ToSequence != 0 && query.FromSequence > query.ToSequence {
		return errors.New("repodb: commit sequence range is invalid")
	}
	return nil
}

func (s catalogState) querySelection(query Query) (map[artifact.ID]struct{}, []artifact.Lineage, bool, error) {
	seed, seeded, err := s.querySeed(query)
	if err != nil {
		return nil, nil, false, err
	}
	selected := make(map[artifact.ID]struct{})
	if !seeded {
		if query.FromSequence != 0 || query.ToSequence != 0 {
			return selected, nil, false, nil
		}
		candidates := s.descriptorCandidates(query)
		ids := make([]artifact.ID, 0, len(candidates))
		for id := range candidates {
			if s.matchesDescriptor(query, id) {
				ids = append(ids, id)
			}
		}
		sort.Slice(ids, func(i, j int) bool { return ids[i].String() < ids[j].String() })
		truncated := len(ids) > query.MaxResults
		if truncated {
			ids = ids[:query.MaxResults]
		}
		for _, id := range ids {
			selected[id] = struct{}{}
		}
		return selected, s.lineageWithin(selected, query.Relation), truncated, nil
	}
	if query.Kind != artifact.KindInvalid && seed.Kind() != query.Kind {
		return selected, nil, false, nil
	}
	type queuedID struct {
		id    artifact.ID
		depth uint32
	}
	queue := []queuedID{{id: seed}}
	selected[seed] = struct{}{}
	var edges []artifact.Lineage
	truncated := false
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		if query.Follow == FollowNone || current.depth >= query.MaxDepth {
			continue
		}
		for _, edge := range s.followEdges(current.id, query.Follow, query.Relation) {
			edges = append(edges, edge)
			next := edge.Parent
			if edge.Parent == current.id {
				next = edge.Child
			}
			if _, seen := selected[next]; seen {
				continue
			}
			if len(selected) >= query.MaxResults {
				truncated = true
				continue
			}
			selected[next] = struct{}{}
			queue = append(queue, queuedID{id: next, depth: current.depth + 1})
		}
	}
	if query.MediaType != "" || query.Schema != "" {
		for id := range selected {
			if !s.matchesDescriptor(query, id) {
				delete(selected, id)
			}
		}
	}
	return selected, edges, truncated, nil
}

func validDescriptorFilter(value string) bool {
	return strings.TrimSpace(value) == value && !strings.ContainsAny(value, "\r\n")
}

func (s catalogState) descriptorCandidates(query Query) map[artifact.ID]struct{} {
	if query.MediaType != "" {
		return s.artifactsByMedia[query.MediaType]
	}
	if query.Schema != "" {
		return s.artifactsBySchema[query.Schema]
	}
	all := make(map[artifact.ID]struct{}, len(s.artifacts))
	for id := range s.artifacts {
		all[id] = struct{}{}
	}
	return all
}

func (s catalogState) matchesDescriptor(query Query, id artifact.ID) bool {
	descriptor, ok := s.artifacts[id]
	return ok && (query.Kind == artifact.KindInvalid || id.Kind() == query.Kind) &&
		(query.MediaType == "" || descriptor.MediaType == query.MediaType) &&
		(query.Schema == "" || descriptor.Schema == query.Schema)
}

func (s catalogState) querySeed(query Query) (artifact.ID, bool, error) {
	var seed artifact.ID
	seeded := false
	if query.Alias != "" {
		var ok bool
		seed, ok = s.aliases[query.Alias]
		if !ok {
			return artifact.ID{}, false, errors.New("repodb: query alias is not bound")
		}
		seeded = true
	}
	if query.Artifact != nil {
		if seeded && seed != *query.Artifact {
			return artifact.ID{}, false, errors.New("repodb: query alias and artifact differ")
		}
		seed, seeded = *query.Artifact, true
	}
	if seeded {
		if _, ok := s.artifacts[seed]; !ok {
			return artifact.ID{}, false, errors.New("repodb: query artifact is unknown")
		}
	}
	return seed, seeded, nil
}

func (s catalogState) followEdges(id artifact.ID, direction FollowDirection, relation artifact.Relation) []artifact.Lineage {
	var edges []artifact.Lineage
	appendIndex := func(index map[artifact.ID]map[relationKey]artifact.Lineage) {
		for _, edge := range index[id] {
			if relation == artifact.RelationInvalid || edge.Relation == relation {
				edges = append(edges, edge)
			}
		}
	}
	if direction == FollowParents || direction == FollowBoth {
		appendIndex(s.parentEdges)
	}
	if direction == FollowChildren || direction == FollowBoth {
		appendIndex(s.childEdges)
	}
	sort.Slice(edges, func(i, j int) bool {
		left, right := edges[i], edges[j]
		if left.Relation != right.Relation {
			return left.Relation < right.Relation
		}
		if left.Child != right.Child {
			return left.Child.String() < right.Child.String()
		}
		return left.Parent.String() < right.Parent.String()
	})
	return edges
}

func (s catalogState) lineageWithin(selected map[artifact.ID]struct{}, relation artifact.Relation) []artifact.Lineage {
	edges := make([]artifact.Lineage, 0)
	for _, edge := range s.lineage {
		_, child := selected[edge.Child]
		_, parent := selected[edge.Parent]
		if child && parent && (relation == artifact.RelationInvalid || edge.Relation == relation) {
			edges = append(edges, edge)
		}
	}
	return edges
}

func (s catalogState) queryAliases(query Query, selected map[artifact.ID]struct{}, truncated *bool) []AliasView {
	aliases := make([]AliasView, 0)
	for name, target := range s.aliases {
		if query.Alias != "" && name != query.Alias {
			continue
		}
		if _, ok := selected[target]; !ok {
			continue
		}
		aliases = append(aliases, AliasView{Name: name, Target: target})
	}
	sort.Slice(aliases, func(i, j int) bool { return aliases[i].Name < aliases[j].Name })
	return truncateQuery(aliases, query.MaxResults, truncated)
}

func sortedQueryLineage(edges []artifact.Lineage, limit int, truncated *bool) []artifact.Lineage {
	sort.Slice(edges, func(i, j int) bool {
		left, right := edges[i], edges[j]
		if left.Relation != right.Relation {
			return left.Relation < right.Relation
		}
		if left.Child != right.Child {
			return left.Child.String() < right.Child.String()
		}
		return left.Parent.String() < right.Parent.String()
	})
	edges = compactLineage(edges)
	return truncateQuery(edges, limit, truncated)
}

func compactLineage(edges []artifact.Lineage) []artifact.Lineage {
	if len(edges) < 2 {
		return edges
	}
	result := edges[:1]
	for _, edge := range edges[1:] {
		if edge != result[len(result)-1] {
			result = append(result, edge)
		}
	}
	return result
}

func (s catalogState) queryCommits(query Query, truncated *bool) []CommitView {
	if query.FromSequence == 0 && query.ToSequence == 0 {
		return nil
	}
	to := query.ToSequence
	if to == 0 {
		to = ^uint64(0)
	}
	commits := make([]CommitView, 0)
	for key, commit := range s.commits {
		if commit.sequence >= query.FromSequence && commit.sequence <= to {
			commits = append(commits, CommitView{Key: key, ID: commit.id, Sequence: commit.sequence})
		}
	}
	sort.Slice(commits, func(i, j int) bool { return commits[i].Sequence < commits[j].Sequence })
	return truncateQuery(commits, query.MaxResults, truncated)
}

func truncateQuery[T any](values []T, limit int, truncated *bool) []T {
	if len(values) <= limit {
		return values
	}
	*truncated = true
	return values[:limit]
}
