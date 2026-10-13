package overgodb

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// keptContract is the schema of the fixture's documents that are of no
// retired family.
var keptContract = artifact.DocumentContract{Kind: artifact.KindEvidence, MediaType: "application/json", Schema: "test/kept/v1"}

// familyBatch builds, from the text of a table entry alone, a batch that
// writes one record of the family the entry names: for a family of artifacts
// the document of its schema, or the bare identity or the schemaless bytes
// its name states, in a batch whose key opens with the family's key; for a
// family of commits a batch of that key; for a family of aliases an alias of
// that name. A batch that carries no artifact of the family carries a kept
// document for its record to stand on. It returns the artifact the batch
// introduces and the producer the store admits its schema from.
func familyBatch(t *testing.T, family retiredFamily, ordinal int) (artifact.Batch, artifact.ID, Producer) {
	t.Helper()
	body := fmt.Appendf(nil, `{"family":%q,"keyed":%q,"ordinal":%d}`, family.name, family.keyed, ordinal)
	document := func(contract artifact.DocumentContract) artifact.Content {
		t.Helper()
		content, err := contract.ContentBytes(body)
		if err != nil {
			t.Fatal(err)
		}
		return content
	}
	kept := document(keptContract)
	switch family.keyed {
	case commitKeys:
		return artifact.Batch{Key: fmt.Sprintf("%s/%d", family.name, ordinal), Contents: []artifact.Content{kept}}, kept.Descriptor.ID, Producer{}
	case aliasNames:
		return artifact.Batch{
			Key: fmt.Sprintf("kept/alias/%d", ordinal), Contents: []artifact.Content{kept},
			Aliases: []artifact.AliasBinding{{Name: fmt.Sprintf("%s/%d", family.name, ordinal), Target: kept.Descriptor.ID}},
		}, kept.Descriptor.ID, Producer{}
	}
	stated, key, _ := strings.Cut(family.name, " by ")
	batch := artifact.Batch{Key: fmt.Sprintf("%s/%d", key, ordinal)}
	schemaless, unstated := strings.CutPrefix(stated, "no schema: ")
	if !unstated {
		content := document(artifact.DocumentContract{Kind: artifact.KindEvidence, MediaType: "application/json", Schema: stated})
		batch.Contents = []artifact.Content{content}
		var producer Producer
		for _, guarded := range producerKinds {
			if strings.HasPrefix(stated, guarded.prefix) {
				producer = NewProducer(guarded.producer)
			}
		}
		return batch, content.Descriptor.ID, producer
	}
	named, media, _ := strings.Cut(schemaless, " ")
	kind, err := artifact.ParseKind(named)
	if err != nil {
		t.Fatalf("family %q states the kind %q: %v", family.name, named, err)
	}
	id := testutil.ArtifactBytesID(t, kind, body)
	if media == "bare" {
		batch.Artifacts = []artifact.Descriptor{{ID: id}}
		return batch, id, Producer{}
	}
	if media == "untyped" {
		media = ""
	}
	batch.Contents = []artifact.Content{{Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(body)), MediaType: media}, Data: body}}
	return batch, id, Producer{}
}

// TestRetiredFamiliesAreFamiliesTheCensusCounts holds the table of retired
// families to the census and to the record it was read from.
//
// A store written from the table's text alone, one record for each entry,
// is counted by the census under exactly the table's names: every entry is a
// family the census produces under the entry's key, and the census marks
// those lines and no other as retired, whatever floor folds the rest. An
// entry that is not a family the census could produce, a key that names one
// record, a schema with no key, a name longer than its leading part, is
// found by the same reading, so it could not stand in the table.
//
// Every entry is a family the record of the installed store's census classes
// as the development process's with no writer in the tree, under the key the
// record counts it in, and the commit the entry gives is the one the record
// says deleted the writer. Every family the record classes so is in the
// table, but for the ones a reader outside the record's four still reads,
// which the test names with their reader.
func TestRetiredFamiliesAreFamiliesTheCensusCounts(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	// written builds a store of one record for each family and returns its root.
	written := func(families []retiredFamily) string {
		t.Helper()
		root := filepath.Join(t.TempDir(), "store")
		store, err := Open(root)
		if err != nil {
			t.Fatal(err)
		}
		for ordinal, family := range families {
			batch, _, producer := familyBatch(t, family, ordinal+1)
			if _, err := store.CommitAs(ctx, producer, batch); err != nil {
				t.Fatalf("family %q: %v", family.name, err)
			}
		}
		if _, err := store.Snapshot(ctx); err != nil {
			t.Fatal(err)
		}
		if err := store.Close(); err != nil {
			t.Fatal(err)
		}
		return root
	}
	// unproduced gives the families the census of a store written from them
	// does not count under their own key and name.
	unproduced := func(families []retiredFamily) []retiredFamily {
		t.Helper()
		census, err := censusCheckpoint(written(families), 1)
		if err != nil {
			t.Fatal(err)
		}
		return slices.DeleteFunc(slices.Clone(families), func(family retiredFamily) bool {
			return slices.ContainsFunc(census.members, func(member memberCensus) bool {
				return keyedLike(member.name) == family.keyed &&
					slices.ContainsFunc(member.families, func(counted familyCount) bool { return counted.name == family.name })
			})
		})
	}
	if missing := unproduced(retiredFamilies); len(missing) != 0 {
		t.Fatalf("the table retires families the census does not produce: %+v", missing)
	}
	malformed := []retiredFamily{
		{artifactKeys, "overgo/closure-ledger/v3 by closure-scan/000001", closureLedgerDeleted, "a key that names one record"},
		{artifactKeys, "overgo/closure-ledger/v3", closureLedgerDeleted, "a schema under no key"},
		{artifactKeys, "no schema: evidence by gate/package", oldGateDeleted, "a schemaless family that states no media type"},
		{aliasNames, "gate/package/evidence", oldGateDeleted, "a name longer than its leading part"},
		{commitKeys, "rebuild/000003", oldGateDeleted, "a key that names one batch"},
	}
	if missing := unproduced(malformed); len(missing) != len(malformed) {
		t.Fatalf("of %d entries the census could not produce, the reading refuses %d: %+v", len(malformed), len(missing), missing)
	}

	// The census marks the table's families and no other.
	// A family's line gives its member, bytes, records, share and date
	// before its name.
	const everything, nameAt = 101, 6
	root := written(retiredFamilies)
	for _, floor := range []float64{0, everything} {
		var output bytes.Buffer
		if err := WriteCheckpointCensus(&output, root, 1, floor); err != nil {
			t.Fatal(err)
		}
		marked := map[retiredFamily]bool{}
		for line := range strings.Lines(output.String()) {
			fields := strings.Fields(line)
			name := strings.Join(fields[min(len(fields), nameAt):], " ")
			switch fields[0] {
			case "retired":
				marked[retiredFamily{keyed: keyedLike(fields[1]), name: name}] = true
			case "family":
				if retiredAt(keyedLike(fields[1]), name) >= 0 {
					t.Fatalf("census at floor %v leaves a retired family unmarked: %s", floor, line)
				}
			}
		}
		for _, family := range retiredFamilies {
			if !marked[retiredFamily{keyed: family.keyed, name: family.name}] {
				t.Fatalf("census at floor %v does not mark %s %q as retired:\n%s", floor, family.keyed, family.name, output.String())
			}
			delete(marked, retiredFamily{keyed: family.keyed, name: family.name})
		}
		if len(marked) != 0 {
			t.Fatalf("census at floor %v marks families the table does not hold: %v", floor, marked)
		}
	}

	// The table against the record it was read from.
	raw, err := os.ReadFile(filepath.Join(testutil.RepoRoot(t), "docs", "verification", "checkpoint-families.txt"))
	if err != nil {
		t.Fatal(err)
	}
	// read are the families the record classes as unwritten and unread by
	// its four readers that a reader outside them still reads.
	read := map[string]string{
		"overgo/finding/v1 by gate/final":                                        "overgodb query -findings reads the register of findings",
		"evaluation/instruction-generations":                                     "internal/evaluation/qwen9_ifeval_scores_test.go resolves these aliases",
		"overgo/instruction-generation/v1 by evaluation/instruction-generations": "internal/evaluation/qwen9_ifeval_scores_test.go loads these documents",
	}
	entries := map[string]map[string]string{}
	var entry map[string]string
	for line := range strings.Lines(string(raw)) {
		key, value, found := strings.Cut(strings.TrimRight(line, "\r\n"), ": ")
		switch {
		case !found || strings.Contains(key, " "):
		case key == "family":
			entry = map[string]string{}
			entries[value] = entry
		case entry != nil:
			entry[key] = value
		}
	}
	const process, absent = "development process", "none in the tree"
	seen := map[string]bool{}
	for _, family := range retiredFamilies {
		recorded := entries[family.name]
		counted := slices.ContainsFunc(strings.Split(recorded["in"], ", "), func(member string) bool { return keyedLike(member) == family.keyed })
		if recorded["class"] != process || !strings.HasPrefix(recorded["writer"], absent) || !strings.Contains(recorded["writer"], family.deleted) || !counted {
			t.Fatalf("the table retires %s %q as deleted by %s, and the record gives it class %q, writer %q, in %q",
				family.keyed, family.name, family.deleted, recorded["class"], recorded["writer"], recorded["in"])
		}
		if family.why == "" || seen[family.keyed+family.name] {
			t.Fatalf("the table's entry %s %q says nothing of its records, or stands twice", family.keyed, family.name)
		}
		seen[family.keyed+family.name] = true
	}
	for name, recorded := range entries {
		if recorded["class"] != process || !strings.HasPrefix(recorded["writer"], absent) || read[name] != "" {
			continue
		}
		for member := range strings.SplitSeq(recorded["in"], ", ") {
			if retiredAt(keyedLike(member), name) < 0 {
				t.Fatalf("the record classes %q in %s as unwritten and unread, and the table neither retires it nor does the test name its reader", name, member)
			}
		}
	}
}

// retiredFixture is a store that holds every family the table retires beside
// records of no retired family, with what a rebuild without the retired
// families is to make of it.
//
// Of every retired family of artifacts it holds one artifact nothing names,
// lone, which declares an edge to the kept anchor and has a location, and
// one a kept document names as its lineage parent, named. Every retired
// family of aliases has an alias to a lone artifact and one to the anchor,
// and every retired family of commits a batch of its key. Three more
// artifacts of retired families stay for other reasons: aliased is the
// target of a kept alias, deep is the lineage parent of aliased and of
// nothing kept otherwise, and execution and root are named by a causal link.
type retiredFixture struct {
	root                                  string
	anchor                                artifact.ID
	lone, stays                           []artifact.ID
	retiredAliases                        []string
	keptAlias                             string
	aliased, deep, execution, causalRoot  artifact.ID
	want                                  []retiredCount
	artifacts, edges, aliases, causalLink int
}

func newRetiredFixture(t *testing.T) retiredFixture {
	t.Helper()
	ctx := t.Context()
	fixture := retiredFixture{root: filepath.Join(t.TempDir(), "store"), keptAlias: "kept/aliased", want: make([]retiredCount, len(retiredFamilies))}
	store, err := Open(fixture.root)
	if err != nil {
		t.Fatal(err)
	}
	ordinal := 0
	// commit writes a batch and counts it under its key when that is a
	// retired family of commits.
	commit := func(producer Producer, batch artifact.Batch) {
		t.Helper()
		if _, err := store.CommitAs(ctx, producer, batch); err != nil {
			t.Fatalf("batch %s: %v", batch.Key, err)
		}
		if at := retiredAt(commitKeys, leadingPart(batch.Key)); at >= 0 {
			fixture.want[at].leftOut++
		}
	}
	// one writes an artifact of the family at a place in the table, with the
	// edges it declares, and returns it with its bytes.
	one := func(at int, parents ...artifact.ID) (artifact.ID, uint64) {
		t.Helper()
		ordinal++
		batch, id, producer := familyBatch(t, retiredFamilies[at], ordinal)
		for _, parent := range parents {
			batch.Lineage = append(batch.Lineage, artifact.Lineage{Child: id, Parent: parent, Relation: artifact.RelationDependsOn})
		}
		var size uint64
		for _, content := range batch.Contents {
			size += content.Descriptor.Size
		}
		commit(producer, batch)
		return id, size
	}
	anchorBatch, anchor, _ := familyBatch(t, retiredFamily{keyed: commitKeys, name: "kept/anchor"}, ordinal)
	commit(Producer{}, anchorBatch)
	fixture.anchor = anchor
	var named []artifact.Lineage
	documents := -1
	for at, family := range retiredFamilies {
		if family.keyed != artifactKeys {
			continue
		}
		lone, size := one(at, anchor)
		commit(Producer{}, artifact.Batch{Key: fmt.Sprintf("kept/location/%d", at), Locations: []artifact.LocationEvent{{
			Location: artifact.Location{Artifact: lone, Kind: artifact.LocationFile, Value: fmt.Sprintf("C:/retired/%d", at)}, Action: artifact.LocationAdd,
		}}})
		count := &fixture.want[at]
		count.leftOut, count.edges, count.locations = 1, 1, 1
		if size != 0 {
			count.contents, count.bytes = 1, size
			documents = at
		}
		fixture.lone = append(fixture.lone, lone)
		stays, _ := one(at)
		count.kept++
		fixture.stays = append(fixture.stays, stays)
		named = append(named, artifact.Lineage{Parent: stays, Relation: artifact.RelationDependsOn})
	}
	if documents < 0 {
		t.Fatal("the table retires no family of documents for the fixture's causal link to name")
	}
	for at, family := range retiredFamilies {
		ordinal++
		batch, target, producer := familyBatch(t, family, ordinal)
		switch family.keyed {
		case commitKeys:
			commit(producer, batch)
			fixture.stays = append(fixture.stays, target)
		case aliasNames:
			batch.Aliases = append(batch.Aliases, artifact.AliasBinding{Name: family.name + ".lone", Target: fixture.lone[0]})
			commit(producer, batch)
			fixture.stays = append(fixture.stays, target)
			fixture.want[at].leftOut = len(batch.Aliases)
			for _, alias := range batch.Aliases {
				fixture.retiredAliases = append(fixture.retiredAliases, alias.Name)
			}
		}
	}
	// A kept document names one artifact of every retired family.
	ordinal++
	namer, namerID, _ := familyBatch(t, retiredFamily{keyed: commitKeys, name: "kept/namer"}, ordinal)
	for index := range named {
		named[index].Child = namerID
	}
	namer.Lineage = named
	commit(Producer{}, namer)
	// A kept alias names one, which names another; a causal link names two.
	fixture.deep, _ = one(documents)
	fixture.aliased, _ = one(documents, fixture.deep)
	fixture.execution, _ = one(documents)
	fixture.causalRoot, _ = one(documents)
	fixture.want[documents].kept += len([]artifact.ID{fixture.deep, fixture.aliased, fixture.execution, fixture.causalRoot})
	commit(Producer{}, artifact.Batch{
		Key:       "kept/cause",
		Aliases:   []artifact.AliasBinding{{Name: fixture.keptAlias, Target: fixture.aliased}},
		Causality: []artifact.CausalLink{{Execution: fixture.execution, Root: fixture.causalRoot, Trigger: "manual"}},
	})
	fixture.stays = append(fixture.stays, anchor, namerID, fixture.deep, fixture.aliased, fixture.execution, fixture.causalRoot)
	for at, family := range retiredFamilies {
		fixture.want[at].retiredFamily = family
	}
	fixture.artifacts, fixture.edges, fixture.aliases = store.state.artifacts.count(), store.state.lineage.count(), store.state.aliases.count()
	fixture.causalLink = len(store.state.causality.records)
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	return fixture
}

// rebuilt rebuilds the fixture's store, read-only, and opens the copy.
func (fixture retiredFixture) rebuilt(t *testing.T, withoutRetired bool) (*Store, *Store, RebuildReport) {
	t.Helper()
	source, err := OpenReadOnly(fixture.root)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = source.Close() })
	destination := filepath.Join(t.TempDir(), "rebuilt")
	report, err := Rebuild(t.Context(), source, destination, nil, nil, withoutRetired)
	if err != nil {
		t.Fatal(err)
	}
	copied, err := OpenReadOnly(destination)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = copied.Close() })
	return source, copied, report
}

// registered is the order a store registered the fixture's kept documents in.
func registered(t *testing.T, store *Store) []artifact.ID {
	t.Helper()
	var order []artifact.ID
	if _, err := store.VisitDocuments(t.Context(), DocumentQuery{Contracts: []artifact.DocumentContract{keptContract}, Order: DocumentOldestFirst}, func(view DocumentView) error {
		order = append(order, view.Content.Descriptor.ID)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	return order
}

// TestRebuildWithoutRetiredLeavesOutWhatNoKeptRecordNames rebuilds a store
// that holds every retired family, without them.
//
// An artifact of a retired family that nothing kept names is gone from the
// copy with its content, its location and the edge it declared, and so is
// every alias named for a retired family, whether its target stays or not.
// An artifact of a retired family stays, as the source held it, when a kept
// document names it as a lineage parent, when a kept alias targets it, when
// an artifact that stayed for either reason names it, and when a causal link
// names it; the alias and the link are carried. Every record of no retired
// family is carried as the source held it.
//
// The report counts each family as the table orders them: what was left out
// of it, what went with those artifacts, and what stayed because it is named;
// its totals are the source's less what the families' counts say was left
// out. The copy registers the kept documents in the order the source did,
// which is not the order of their identities.
func TestRebuildWithoutRetiredLeavesOutWhatNoKeptRecordNames(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	fixture := newRetiredFixture(t)
	source, copied, report := fixture.rebuilt(t, true)

	for _, lone := range fixture.lone {
		_, held, err := copied.Artifact(ctx, lone)
		content, contentErr := copied.HasContent(ctx, lone)
		locations, locationErr := copied.Locations(ctx, lone)
		if err != nil || contentErr != nil || locationErr != nil || held || content || len(locations) != 0 {
			t.Fatalf("the copy holds retired artifact %s that nothing names: artifact %t content %t locations %v (%v, %v, %v)", lone, held, content, locations, err, contentErr, locationErr)
		}
	}
	if children, err := copied.Children(ctx, fixture.anchor); err != nil || len(children) != 0 {
		t.Fatalf("the copy holds edges the artifacts left out declared: %v, %v", children, err)
	}
	for _, name := range fixture.retiredAliases {
		if target, found, err := copied.ResolveAlias(ctx, name); err != nil || found {
			t.Fatalf("the copy resolves retired alias %q to %s (%v)", name, target, err)
		}
	}
	for _, id := range fixture.stays {
		before, _, err := source.Artifact(ctx, id)
		after, held, afterErr := copied.Artifact(ctx, id)
		if err != nil || afterErr != nil || !held || before != after {
			t.Fatalf("the copy holds %s as %+v (%t), and the source as %+v (%v, %v)", id, after, held, before, err, afterErr)
		}
		contentBefore, err := source.HasContent(ctx, id)
		contentAfter, afterErr := copied.HasContent(ctx, id)
		if err != nil || afterErr != nil || contentBefore != contentAfter {
			t.Fatalf("the copy holds the content of %s %t, and the source %t (%v, %v)", id, contentAfter, contentBefore, err, afterErr)
		}
		parentsBefore, err := source.Parents(ctx, id)
		parentsAfter, afterErr := copied.Parents(ctx, id)
		if err != nil || afterErr != nil || !slices.Equal(parentsBefore, parentsAfter) {
			t.Fatalf("the copy gives %s the parents %v, and the source %v (%v, %v)", id, parentsAfter, parentsBefore, err, afterErr)
		}
	}
	if target, found, err := copied.ResolveAlias(ctx, fixture.keptAlias); err != nil || !found || target != fixture.aliased {
		t.Fatalf("the copy resolves the kept alias to %s (%t, %v), want %s", target, found, err, fixture.aliased)
	}
	if linked, err := copied.QueryCausality(ctx, CausalityQuery{Execution: &fixture.execution}); err != nil || linked.Matched != fixture.causalLink {
		t.Fatalf("the copy holds %d causal links of the kept execution (%v), want %d", linked.Matched, err, fixture.causalLink)
	}

	if !slices.Equal(report.retired, fixture.want) {
		t.Fatalf("the rebuild counts the retired families as\n%+v\nwant\n%+v", report.retired, fixture.want)
	}
	lines := report.Retired()
	if len(lines) != len(retiredFamilies) {
		t.Fatalf("the report has %d lines for %d retired families", len(lines), len(retiredFamilies))
	}
	var artifacts, edges, aliases int
	for at, count := range fixture.want {
		if !strings.HasPrefix(lines[at], fmt.Sprintf("retired %s left-out %d ", count.keyed, count.leftOut)) || !strings.HasSuffix(lines[at], fmt.Sprintf(" kept %d %s", count.kept, count.name)) {
			t.Fatalf("the report's line %d is %q, want it to count %+v", at, lines[at], count)
		}
		switch count.keyed {
		case artifactKeys:
			artifacts, edges = artifacts+count.leftOut, edges+count.edges
		case aliasNames:
			aliases += count.leftOut
		}
	}
	if report.Artifacts != fixture.artifacts-artifacts || report.LineageEdges != fixture.edges-edges || report.Aliases != fixture.aliases-aliases || report.CausalLinks != fixture.causalLink {
		t.Fatalf("the rebuild carried %d artifacts, %d edges, %d aliases and %d causal links of the source's %d, %d, %d and %d, having left out %d, %d and %d",
			report.Artifacts, report.LineageEdges, report.Aliases, report.CausalLinks, fixture.artifacts, fixture.edges, fixture.aliases, fixture.causalLink, artifacts, edges, aliases)
	}

	order := registered(t, source)
	if slices.IsSortedFunc(order, func(left, right artifact.ID) int { return strings.Compare(left.String(), right.String()) }) {
		t.Fatal("the fixture registers its kept documents in the order of their identities, which no rebuild could lose")
	}
	if got := registered(t, copied); !slices.Equal(got, order) {
		t.Fatalf("the copy registers the kept documents as %v, and the source as %v", got, order)
	}
}

// TestRebuildNotAskedLeavesNothingOut rebuilds the same store without being
// asked to leave the retired families out. The copy holds every artifact,
// edge, alias and causal link of the source, the retired ones among them
// with their content and their locations; the report has no line for a
// retired family; and the copy registers the documents in the order the
// source registered them, as a rebuild that leaves the retired families out
// does.
func TestRebuildNotAskedLeavesNothingOut(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	fixture := newRetiredFixture(t)
	source, copied, report := fixture.rebuilt(t, false)
	if lines := report.Retired(); len(lines) != 0 {
		t.Fatalf("a rebuild not asked reports retired families: %v", lines)
	}
	if report.Artifacts != fixture.artifacts || report.LineageEdges != fixture.edges || report.Aliases != fixture.aliases || report.CausalLinks != fixture.causalLink {
		t.Fatalf("a rebuild not asked carried %+v of %d artifacts, %d edges, %d aliases and %d causal links", report, fixture.artifacts, fixture.edges, fixture.aliases, fixture.causalLink)
	}
	if order, got := registered(t, source), registered(t, copied); !slices.Equal(got, order) {
		t.Fatalf("a rebuild not asked registers the documents as %v, and the source as %v", got, order)
	}
	for _, lone := range fixture.lone {
		before, _, err := source.Artifact(ctx, lone)
		after, held, afterErr := copied.Artifact(ctx, lone)
		content, contentErr := copied.HasContent(ctx, lone)
		locations, locationErr := copied.Locations(ctx, lone)
		if err != nil || afterErr != nil || contentErr != nil || locationErr != nil || !held || before != after || content != (before.Size != 0) || len(locations) != 1 {
			t.Fatalf("a rebuild not asked holds %s as %+v (%t) with content %t and locations %v, and the source as %+v (%v, %v, %v, %v)",
				lone, after, held, content, locations, before, err, afterErr, contentErr, locationErr)
		}
	}
	if children, err := copied.Children(ctx, fixture.anchor); err != nil || len(children) != len(fixture.lone) {
		t.Fatalf("a rebuild not asked holds %d of the %d edges to the anchor (%v)", len(children), len(fixture.lone), err)
	}
	for _, name := range fixture.retiredAliases {
		before, _, err := source.ResolveAlias(ctx, name)
		after, found, afterErr := copied.ResolveAlias(ctx, name)
		if err != nil || afterErr != nil || !found || before != after {
			t.Fatalf("a rebuild not asked resolves %q to %s (%t), and the source to %s (%v, %v)", name, after, found, before, err, afterErr)
		}
	}
}
