package overgodb

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"maps"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

// FollowDirection selects which lineage edges a query traverses.
type FollowDirection uint8

const (
	// FollowNone traverses no lineage edges.
	FollowNone FollowDirection = iota
	// FollowParents traverses edges toward ancestors.
	FollowParents
	// FollowChildren traverses edges toward descendants.
	FollowChildren
	// FollowBoth traverses edges in both directions.
	FollowBoth
)

// QueryProjection selects returned catalog facts.
type QueryProjection uint16

const (
	// ProjectArtifacts returns descriptors.
	ProjectArtifacts QueryProjection = 1 << iota
	// ProjectContentPresence returns payload identities.
	ProjectContentPresence
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
	projectAll     = ProjectCatalog | ProjectContentPresence | ProjectParents
)

func (p QueryProjection) includes(field QueryProjection) bool {
	return p&field != 0
}

// QueryCursor binds continuation to a catalog head and query contract.
type QueryCursor struct {
	Head          artifact.CommitID `json:"head"`
	Contract      [sha256.Size]byte `json:"contract"`
	After         artifact.ID       `json:"after"`
	AfterSequence uint64            `json:"after_sequence,omitzero"`
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
		return QueryCursor{}, errors.New("overgodb: invalid query cursor encoding")
	}
	var cursor QueryCursor
	if err := strictjson.DecodeBytes(payload, &cursor); err != nil {
		return QueryCursor{}, errors.New("overgodb: invalid query cursor document")
	}
	return cursor, nil
}

// Query filters the catalog by identity, alias, lineage, and commit
// range, bounding results and selecting projections through one value.
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
	// RequireIndex refuses a query no projection index owns instead of
	// silently scanning the complete catalog.
	RequireIndex bool
}

// QueryPlanReport states which owning projection served a query and what it
// cost: candidates inspected, descriptors matched, rows returned, and
// projected facts loaded. The plan is deterministic for one query contract
// and head, so equal queries at equal heads report equal plans.
type QueryPlanReport struct {
	Index     string `json:"index"`
	Inspected int    `json:"inspected"`
	Matched   int    `json:"matched"`
	Returned  int    `json:"returned"`
	Loaded    int    `json:"loaded"`
}

// AliasView carries one projected alias binding.
type AliasView struct {
	Name   string      `json:"name"`
	Target artifact.ID `json:"target"`
}

// CommitView carries one projected commit fact.
type CommitView struct {
	Key      string            `json:"key"`
	ID       artifact.CommitID `json:"id"`
	Sequence uint64            `json:"sequence"`
}

// ArtifactIntroduction identifies the exact commit that first made an
// artifact's content bytes durable. Descriptor declarations do not establish
// this authority, and repeated content never moves it.
type ArtifactIntroduction struct {
	Artifact artifact.ID       `json:"artifact"`
	Commit   artifact.CommitID `json:"commit"`
	Sequence uint64            `json:"sequence"`
}

// ContentView carries projected payload identity.
type ContentView struct {
	Artifact artifact.ID `json:"artifact"`
}

// QueryResult carries the matched projections plus the head coordinates
// they were derived from and a cursor when the result was truncated.
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
	Truncated bool                  `json:"truncated,omitzero"`
	Next      *QueryCursor          `json:"next,omitempty"`
	Plan      QueryPlanReport       `json:"plan"`
}

// ArtifactIntroduction returns the immutable first-durable-content commit
// authority for one artifact. It is a constant-time lookup over the content
// and commit projections rather than an unbounded document scan.
func (s *Store) ArtifactIntroduction(
	ctx context.Context,
	id artifact.ID,
) (ArtifactIntroduction, bool, error) {
	if !id.Valid() {
		return ArtifactIntroduction{}, false, errors.New("overgodb: invalid artifact introduction identity")
	}
	if err := contextError(ctx); err != nil {
		return ArtifactIntroduction{}, false, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.ready(false); err != nil {
		return ArtifactIntroduction{}, false, err
	}
	locator, found := s.state.contents.locator(id)
	if !found {
		return ArtifactIntroduction{}, false, nil
	}
	record, described := s.state.artifacts.record(id)
	if !described || locator.sequence < record.sequence || locator.sequence > uint64(s.state.commits.count()) {
		return ArtifactIntroduction{}, false, errors.New("overgodb: content introduction sequence is corrupt")
	}
	commit := s.state.commits.at(int(locator.sequence - 1))
	if commit.sequence != locator.sequence || !commit.id.Valid() {
		return ArtifactIntroduction{}, false, errors.New("overgodb: content introduction commit is corrupt")
	}
	return ArtifactIntroduction{Artifact: id, Commit: commit.id, Sequence: locator.sequence}, true, nil
}

// Query evaluates one filtered, bounded catalog read against current state.
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
	if query.MaxResults == 0 && (s.state.artifacts.count() != 0 || s.state.aliases.count() != 0 ||
		s.state.lineage.count() != 0 || s.state.commits.count() != 0) {
		return QueryResult{}, errors.New("overgodb: zero query bound requires an empty catalog")
	}
	contract, err := queryContractDigest(query)
	if err != nil {
		return QueryResult{}, err
	}
	if query.Cursor != nil && (query.Cursor.Head != s.head || query.Cursor.Contract != contract) {
		return QueryResult{}, errors.New("overgodb: query cursor is stale or belongs to another query")
	}
	planIndex := "seed"
	if query.Artifact == nil && query.Alias == "" {
		if query.FromSequence != 0 || query.ToSequence != 0 {
			planIndex = "sequence"
		} else {
			planIndex, _ = s.state.queryIndexPlan(query)
		}
	}
	if query.RequireIndex && planIndex == "catalog" {
		return QueryResult{}, errors.New(
			"overgodb: no projection index owns this query contract; narrow the contract or drop the index requirement",
		)
	}
	result := QueryResult{Head: s.head, Sequence: s.sequence}
	selected, ids, edges, inspected, matched, truncated, err := s.state.querySelection(query)
	if err != nil {
		return QueryResult{}, err
	}
	result.Truncated = truncated
	result.Matched = matched
	for _, id := range ids {
		record, ok := s.state.artifacts.record(id)
		if !ok {
			continue
		}
		if query.Projection.includes(ProjectArtifacts) {
			result.Artifacts = append(result.Artifacts, record.descriptor)
		}
		if s.state.contents.has(id) && query.Projection.includes(ProjectContentPresence) {
			result.Contents = append(result.Contents, ContentView{Artifact: id})
		}
		if record.hasManifest && query.Projection.includes(ProjectManifests) {
			result.Manifests = append(result.Manifests, record.manifest.Clone())
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
	result.Plan = QueryPlanReport{
		Index: planIndex, Inspected: inspected, Matched: matched, Returned: len(ids),
		Loaded: len(result.Artifacts) + len(result.Contents) + len(result.Manifests) +
			len(result.Aliases) + len(result.Lineage) + len(result.Commits),
	}
	return result, nil
}

func validateQuery(query Query) error {
	if query.MaxResults < 0 {
		return errors.New("overgodb: query result bound is invalid")
	}
	if query.Projection == 0 || query.Projection&^projectAll != 0 {
		return errors.New("overgodb: query projection is invalid")
	}
	if query.Kind != artifact.KindInvalid {
		if _, err := artifact.ParseKind(query.Kind.String()); err != nil {
			return err
		}
	}
	if !validDescriptorFilter(query.MediaType) || !validDescriptorFilter(query.Schema) {
		return errors.New("overgodb: query descriptor filter is invalid")
	}
	if query.Artifact != nil && !query.Artifact.Valid() {
		return errors.New("overgodb: query artifact is invalid")
	}
	if query.Alias != "" && (strings.TrimSpace(query.Alias) != query.Alias || strings.ContainsAny(query.Alias, "\r\n")) {
		return errors.New("overgodb: query alias is invalid")
	}
	if query.Relation != artifact.RelationInvalid {
		if _, err := artifact.ParseRelation(query.Relation.String()); err != nil {
			return err
		}
	}
	if query.Follow > FollowBoth || query.Follow != FollowNone && query.Artifact == nil && query.Alias == "" {
		return errors.New("overgodb: lineage follow requires one artifact or alias")
	}
	if query.Follow != FollowNone && query.MaxDepth == 0 {
		return errors.New("overgodb: lineage depth bound is invalid")
	}
	if query.ToSequence != 0 && query.FromSequence > query.ToSequence {
		return errors.New("overgodb: commit sequence range is invalid")
	}
	if query.Cursor != nil {
		if !query.Cursor.After.Valid() || query.Artifact != nil || query.Alias != "" || query.Follow != FollowNone ||
			query.FromSequence != 0 || query.ToSequence != 0 {
			return errors.New("overgodb: query cursor is incompatible with query shape")
		}
	}
	return nil
}

func queryContractDigest(query Query) ([sha256.Size]byte, error) {
	contract := struct {
		Kind         uint8
		MediaType    string
		Schema       string
		Artifact     string
		Alias        string
		Relation     uint8
		Follow       uint8
		MaxDepth     uint32
		From, To     uint64
		Projection   uint16
		RequireIndex bool
	}{
		Kind: uint8(query.Kind), MediaType: query.MediaType, Schema: query.Schema,
		Alias: query.Alias, Relation: uint8(query.Relation), Follow: uint8(query.Follow), MaxDepth: query.MaxDepth,
		From: query.FromSequence, To: query.ToSequence, Projection: uint16(query.Projection),
		RequireIndex: query.RequireIndex,
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

func (s catalogState) querySelection(query Query) (map[artifact.ID]struct{}, []artifact.ID, []artifact.Lineage, int, int, bool, error) {
	seed, seeded, err := s.querySeed(query)
	if err != nil {
		return nil, nil, nil, 0, 0, false, err
	}
	selected := make(map[artifact.ID]struct{})
	if !seeded {
		if query.FromSequence != 0 || query.ToSequence != 0 {
			return selected, nil, nil, 0, 0, false, nil
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
		return selected, ids, edges, candidateCount, matched, truncated, nil
	}
	if query.Kind != artifact.KindInvalid && seed.Kind() != query.Kind {
		return selected, nil, nil, len(selected), 0, false, nil
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
	inspected := len(selected)
	if query.MediaType != "" || query.Schema != "" {
		maps.DeleteFunc(selected, func(id artifact.ID, _ struct{}) bool {
			return !s.matchesDescriptor(query, id)
		})
	}
	ids := slices.SortedFunc(maps.Keys(selected), artifact.CompareID)
	switch {
	case ids == nil:
		ids = []artifact.ID{}
	}
	return selected, ids, edges, inspected, len(selected), truncated, nil
}

func validDescriptorFilter(value string) bool {
	return strings.TrimSpace(value) == value && !strings.ContainsAny(value, "\r\n")
}

// queryIndexPlan names the owning projection index for one descriptor query,
// deterministically for the contract and head: the narrowest declared filter
// wins, and a media/schema pair tie-breaks by measured candidate width at
// this exact head.
func (s catalogState) queryIndexPlan(query Query) (string, int) {
	media, schema := query.MediaType != "", query.Schema != ""
	switch {
	case media && schema:
		if len(s.artifacts.byMedia[query.MediaType]) <= len(s.artifacts.bySchema[query.Schema]) {
			return "media", len(s.artifacts.byMedia[query.MediaType])
		}
		return "schema", len(s.artifacts.bySchema[query.Schema])
	case media:
		return "media", len(s.artifacts.byMedia[query.MediaType])
	case schema:
		return "schema", len(s.artifacts.bySchema[query.Schema])
	case query.Kind != artifact.KindInvalid:
		return "kind", len(s.artifacts.byKind[query.Kind])
	default:
		return "catalog", s.artifacts.count()
	}
}

func (s catalogState) visitDescriptorCandidates(query Query, visit func(artifact.ID)) {
	index, _ := s.queryIndexPlan(query)
	switch index {
	case "media":
		for _, id := range s.artifacts.byMedia[query.MediaType] {
			visit(id)
		}
	case "schema":
		for _, id := range s.artifacts.bySchema[query.Schema] {
			visit(id)
		}
	case "kind":
		for _, id := range s.artifacts.byKind[query.Kind] {
			visit(id)
		}
	default:
		for id := range s.artifacts.records {
			visit(id)
		}
	}
}

func (s catalogState) descriptorCandidateCount(query Query) int {
	_, count := s.queryIndexPlan(query)
	return count
}

func (s catalogState) matchesDescriptor(query Query, id artifact.ID) bool {
	record, ok := s.artifacts.record(id)
	return ok && (query.Kind == artifact.KindInvalid || id.Kind() == query.Kind) &&
		(query.MediaType == "" || record.descriptor.MediaType == query.MediaType) &&
		(query.Schema == "" || record.descriptor.Schema == query.Schema)
}

func (s catalogState) querySeed(query Query) (artifact.ID, bool, error) {
	var seed artifact.ID
	seeded := false
	if query.Alias != "" {
		var ok bool
		seed, ok = s.aliases.resolve(query.Alias)
		if !ok {
			return artifact.ID{}, false, errors.New("overgodb: query alias is not bound")
		}
		seeded = true
	}
	if query.Artifact != nil {
		if seeded && seed != *query.Artifact {
			return artifact.ID{}, false, errors.New("overgodb: query alias and artifact differ")
		}
		seed, seeded = *query.Artifact, true
	}
	if seeded {
		if !s.artifacts.has(seed) {
			return artifact.ID{}, false, errors.New("overgodb: query artifact is unknown")
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
	if direction == FollowParents || direction == FollowBoth {
		appendIndex(s.lineage.parentsOf(id))
	}
	if direction == FollowChildren || direction == FollowBoth {
		appendIndex(s.lineage.childrenOf(id))
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
		for _, key := range s.lineage.parentsOf(id) {
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
		index := s.lineage.childrenOf(id)
		if parents {
			index = s.lineage.parentsOf(id)
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
	s.aliases.each(func(name string, target artifact.ID) {
		if query.Alias != "" && name != query.Alias {
			return
		}
		if _, ok := selected[target]; !ok {
			return
		}
		aliases = append(aliases, AliasView{Name: name, Target: target})
	})
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
	return slices.Compact(edges)
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
	for _, commit := range s.commits.all() {
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
