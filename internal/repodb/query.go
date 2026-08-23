package repodb

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

type FollowDirection uint8

const (
	FollowNone FollowDirection = iota
	FollowParents
	FollowChildren
	FollowBoth
)

// QueryProjection selects returned catalog facts.
type QueryProjection uint16

const (
	// ProjectArtifacts returns descriptors.
	ProjectArtifacts QueryProjection = 1 << iota
	// ProjectContentPresence returns payload identities.
	ProjectContentPresence
	// ProjectContentData returns payload identities and bytes.
	ProjectContentData
	// ProjectManifests returns selected manifests.
	ProjectManifests
	// ProjectAliases returns selected alias bindings.
	ProjectAliases
	projectLineage
	// ProjectParents returns selected parent edges.
	ProjectParents
	// ProjectCommits returns selected commit facts.
	ProjectCommits

	// ProjectCatalog returns the original catalog query surface.
	ProjectCatalog = ProjectArtifacts | ProjectManifests | ProjectAliases | projectLineage | ProjectCommits
	projectAll     = ProjectCatalog | ProjectContentPresence | ProjectContentData | ProjectParents
)

func (p QueryProjection) includes(field QueryProjection) bool {
	return p&field != 0
}

// QueryCursor binds continuation to a catalog head and query contract.
type QueryCursor struct {
	Head     artifact.CommitID `json:"head"`
	Contract [sha256.Size]byte `json:"contract"`
	After    artifact.ID       `json:"after"`
}

// EncodeQueryCursor encodes a URL-safe continuation.
func EncodeQueryCursor(cursor QueryCursor) (string, error) {
	payload, err := json.Marshal(cursor)
	if err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(payload), nil
}

// ParseQueryCursor decodes a URL-safe continuation.
func ParseQueryCursor(value string) (QueryCursor, error) {
	payload, err := base64.RawURLEncoding.DecodeString(value)
	if err != nil {
		return QueryCursor{}, errors.New("repodb: invalid query cursor encoding")
	}
	var cursor QueryCursor
	if err := strictjson.DecodeBytes(payload, &cursor); err != nil {
		return QueryCursor{}, errors.New("repodb: invalid query cursor document")
	}
	return cursor, nil
}

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
	Projection   QueryProjection
	Cursor       *QueryCursor
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

// ContentView carries projected payload facts.
type ContentView struct {
	Artifact artifact.ID `json:"artifact"`
	Data     []byte      `json:"data,omitempty"`
}

type QueryResult struct {
	Head      artifact.CommitID     `json:"head"`
	Sequence  uint64                `json:"sequence"`
	Matched   int                   `json:"matched"`
	Artifacts []artifact.Descriptor `json:"artifacts,omitempty"`
	Contents  []ContentView         `json:"contents,omitempty"`
	Manifests []artifact.Manifest   `json:"manifests,omitempty"`
	Aliases   []AliasView           `json:"aliases,omitempty"`
	Lineage   []artifact.Lineage    `json:"lineage,omitempty"`
	Commits   []CommitView          `json:"commits,omitempty"`
	Truncated bool                  `json:"truncated,omitempty"`
	Next      *QueryCursor          `json:"next,omitempty"`
}

// Content returns projected bytes without another store read.
func (r QueryResult) Content(id artifact.ID) ([]byte, bool) {
	index, found := slices.BinarySearchFunc(r.Contents, id, func(view ContentView, target artifact.ID) int {
		return artifact.CompareID(view.Artifact, target)
	})
	if !found {
		return nil, false
	}
	return r.Contents[index].Data, true
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
	if query.MaxResults == 0 && s.state.queryExtent() != 0 {
		return QueryResult{}, errors.New("repodb: zero query bound requires an empty catalog")
	}
	contract, err := queryContractDigest(query)
	if err != nil {
		return QueryResult{}, err
	}
	if query.Cursor != nil && (query.Cursor.Head != s.head || query.Cursor.Contract != contract) {
		return QueryResult{}, errors.New("repodb: query cursor is stale or belongs to another query")
	}
	result := QueryResult{Head: s.head, Sequence: s.sequence}
	selected, ids, edges, matched, truncated, err := s.state.querySelection(query)
	if err != nil {
		return QueryResult{}, err
	}
	result.Truncated = truncated
	result.Matched = matched
	legacyContent := map[int64]map[artifact.ID][]byte{}
	for _, id := range ids {
		slot := s.state.slots[id]
		if slot == nil {
			continue
		}
		if query.Projection.includes(ProjectArtifacts) {
			result.Artifacts = append(result.Artifacts, slot.descriptor)
		}
		if slot.hasContent && query.Projection.includes(ProjectContentPresence|ProjectContentData) {
			view := ContentView{Artifact: id}
			if query.Projection.includes(ProjectContentData) {
				view.Data, err = s.materializeQueryContent(slot.content, id, legacyContent)
				if err != nil {
					return QueryResult{}, err
				}
			}
			result.Contents = append(result.Contents, view)
		}
		if slot.hasManifest && query.Projection.includes(ProjectManifests) {
			result.Manifests = append(result.Manifests, slot.manifest.Clone())
		}
	}
	if query.Projection.includes(ProjectAliases) {
		result.Aliases = s.state.queryAliases(query, selected, &result.Truncated)
	}
	if query.Projection.includes(ProjectParents) {
		edges = append(edges, s.state.indexedLineage(selected, true, query.Relation)...)
	}
	if query.Projection.includes(projectLineage | ProjectParents) {
		result.Lineage = sortedQueryLineage(edges, query.MaxResults, &result.Truncated)
	}
	if query.Projection.includes(ProjectCommits) {
		result.Commits = s.state.queryCommits(query, &result.Truncated)
	}
	if truncated && len(ids) > 0 && query.Artifact == nil && query.Alias == "" &&
		query.FromSequence == 0 && query.ToSequence == 0 {
		result.Next = &QueryCursor{Head: s.head, Contract: contract, After: ids[len(ids)-1]}
	}
	return result, nil
}

// QueryExtent returns the exact current fact-class bound.
func (s *Store) QueryExtent() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.state.queryExtent()
}

func (s catalogState) queryExtent() int {
	return max(len(s.slots), len(s.aliases), s.edgeCount, len(s.commits))
}

func validateQuery(query Query) error {
	if query.MaxResults < 0 {
		return errors.New("repodb: query result bound is invalid")
	}
	if query.Projection == 0 || query.Projection&^projectAll != 0 {
		return errors.New("repodb: query projection is invalid")
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
	if query.Cursor != nil {
		if !query.Cursor.After.Valid() || query.Artifact != nil || query.Alias != "" || query.Follow != FollowNone ||
			query.FromSequence != 0 || query.ToSequence != 0 {
			return errors.New("repodb: query cursor is incompatible with query shape")
		}
	}
	return nil
}

func queryContractDigest(query Query) ([sha256.Size]byte, error) {
	contract := struct {
		Kind       uint8
		MediaType  string
		Schema     string
		Artifact   string
		Alias      string
		Relation   uint8
		Follow     uint8
		MaxDepth   uint32
		From, To   uint64
		Projection uint16
	}{
		Kind: uint8(query.Kind), MediaType: query.MediaType, Schema: query.Schema,
		Alias: query.Alias, Relation: uint8(query.Relation), Follow: uint8(query.Follow), MaxDepth: query.MaxDepth,
		From: query.FromSequence, To: query.ToSequence, Projection: uint16(query.Projection),
	}
	if query.Artifact != nil {
		contract.Artifact = query.Artifact.String()
	}
	payload, err := json.Marshal(contract)
	if err != nil {
		return [sha256.Size]byte{}, err
	}
	return sha256.Sum256(payload), nil
}

func (s catalogState) querySelection(query Query) (map[artifact.ID]struct{}, []artifact.ID, []artifact.Lineage, int, bool, error) {
	seed, seeded, err := s.querySeed(query)
	if err != nil {
		return nil, nil, nil, 0, false, err
	}
	selected := make(map[artifact.ID]struct{})
	if !seeded {
		if query.FromSequence != 0 || query.ToSequence != 0 {
			return selected, nil, nil, 0, false, nil
		}
		candidateCount := s.descriptorCandidateCount(query)
		bounded := query.Cursor != nil || query.MaxResults < candidateCount
		ids := make([]artifact.ID, 0, min(candidateCount, query.MaxResults))
		matched := 0
		truncated := false
		s.visitDescriptorCandidates(query, func(id artifact.ID) {
			if !s.matchesDescriptor(query, id) {
				return
			}
			matched++
			if query.Cursor != nil && artifact.CompareID(id, query.Cursor.After) <= 0 {
				return
			}
			if !bounded {
				ids = append(ids, id)
				return
			}
			index, _ := slices.BinarySearchFunc(ids, id, artifact.CompareID)
			if len(ids) < query.MaxResults {
				ids = append(ids, artifact.ID{})
				copy(ids[index+1:], ids[index:])
				ids[index] = id
				return
			}
			truncated = true
			if index < len(ids) {
				copy(ids[index+1:], ids[index:len(ids)-1])
				ids[index] = id
			}
		})
		if !bounded {
			slices.SortFunc(ids, artifact.CompareID)
		}
		for _, id := range ids {
			selected[id] = struct{}{}
		}
		var edges []artifact.Lineage
		if query.Projection.includes(projectLineage) {
			edges = s.lineageWithin(selected, query.Relation)
		}
		return selected, ids, edges, matched, truncated, nil
	}
	if query.Kind != artifact.KindInvalid && seed.Kind() != query.Kind {
		return selected, nil, nil, 0, false, nil
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
	ids := make([]artifact.ID, 0, len(selected))
	for id := range selected {
		ids = append(ids, id)
	}
	slices.SortFunc(ids, artifact.CompareID)
	return selected, ids, edges, len(selected), truncated, nil
}

func validDescriptorFilter(value string) bool {
	return strings.TrimSpace(value) == value && !strings.ContainsAny(value, "\r\n")
}

func (s catalogState) visitDescriptorCandidates(query Query, visit func(artifact.ID)) {
	if query.MediaType != "" {
		for _, id := range s.byMedia[query.MediaType] {
			visit(id)
		}
		return
	}
	if query.Schema != "" {
		for _, id := range s.bySchema[query.Schema] {
			visit(id)
		}
		return
	}
	for id := range s.slots {
		visit(id)
	}
}

func (s catalogState) descriptorCandidateCount(query Query) int {
	if query.MediaType != "" {
		return len(s.byMedia[query.MediaType])
	}
	if query.Schema != "" {
		return len(s.bySchema[query.Schema])
	}
	return len(s.slots)
}

func (s catalogState) matchesDescriptor(query Query, id artifact.ID) bool {
	slot, ok := s.slots[id]
	return ok && (query.Kind == artifact.KindInvalid || id.Kind() == query.Kind) &&
		(query.MediaType == "" || slot.descriptor.MediaType == query.MediaType) &&
		(query.Schema == "" || slot.descriptor.Schema == query.Schema)
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
		if _, ok := s.slots[seed]; !ok {
			return artifact.ID{}, false, errors.New("repodb: query artifact is unknown")
		}
	}
	return seed, seeded, nil
}

func (s catalogState) followEdges(id artifact.ID, direction FollowDirection, relation artifact.Relation) []artifact.Lineage {
	var edges []artifact.Lineage
	appendIndex := func(index []relationKey) {
		for _, key := range index {
			edge := artifact.Lineage{Child: key.child, Parent: key.parent, Relation: key.relation}
			if relation == artifact.RelationInvalid || edge.Relation == relation {
				edges = append(edges, edge)
			}
		}
	}
	slot := s.slots[id]
	if slot == nil {
		return nil
	}
	if direction == FollowParents || direction == FollowBoth {
		appendIndex(slot.parents)
	}
	if direction == FollowChildren || direction == FollowBoth {
		appendIndex(slot.children)
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
	for id := range selected {
		for _, key := range s.slots[id].parents {
			if _, parent := selected[key.parent]; parent && (relation == artifact.RelationInvalid || key.relation == relation) {
				edges = append(edges, artifact.Lineage{Child: key.child, Parent: key.parent, Relation: key.relation})
			}
		}
	}
	return edges
}

func (s catalogState) indexedLineage(
	selected map[artifact.ID]struct{},
	parents bool,
	relation artifact.Relation,
) []artifact.Lineage {
	edges := make([]artifact.Lineage, 0)
	for id := range selected {
		index := s.slots[id].children
		if parents {
			index = s.slots[id].parents
		}
		for _, key := range index {
			edge := artifact.Lineage{Child: key.child, Parent: key.parent, Relation: key.relation}
			if relation == artifact.RelationInvalid || edge.Relation == relation {
				edges = append(edges, edge)
			}
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
	for _, commit := range s.commits {
		if commit.sequence >= query.FromSequence && commit.sequence <= to {
			commits = append(commits, CommitView{Key: commit.key, ID: commit.id, Sequence: commit.sequence})
		}
	}
	return truncateQuery(commits, query.MaxResults, truncated)
}

func truncateQuery[T any](values []T, limit int, truncated *bool) []T {
	if len(values) <= limit {
		return values
	}
	*truncated = true
	return values[:limit]
}
