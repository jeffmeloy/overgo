package overgodb

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// censusFixture is a store of three commits with a checkpoint at its head.
// The first, keyed alpha/one, carries one document of the alpha schema, the
// bare identity it depends on and an alias to it; the second, keyed
// beta/two, two documents of the beta schema, one depending on the other,
// and an alias; the third, keyed gamma/3, one more alpha document.
type censusFixture struct {
	root                       string
	alpha, later, first, other artifact.Content
	declared                   artifact.Descriptor
}

func newCensusFixture(t *testing.T) censusFixture {
	t.Helper()
	ctx := t.Context()
	fixture := censusFixture{root: filepath.Join(t.TempDir(), "store")}
	mint := func(schema, body string) artifact.Content {
		t.Helper()
		contract := artifact.DocumentContract{Kind: artifact.KindOutput, MediaType: "application/json", Schema: schema}
		content, err := contract.ContentBytes([]byte(body))
		if err != nil {
			t.Fatal(err)
		}
		return content
	}
	fixture.alpha, fixture.later = mint("test/alpha/v1", `{"alpha":1}`), mint("test/alpha/v1", `{"alpha":2,"later":true}`)
	fixture.first, fixture.other = mint("test/beta/v1", `{"beta":1}`), mint("test/beta/v1", `{"beta":"the second of two"}`)
	fixture.declared = artifact.Descriptor{ID: testutil.ArtifactID(t, artifact.KindEvidence, "declared and not carried")}
	store, err := Open(fixture.root)
	if err != nil {
		t.Fatal(err)
	}
	for _, batch := range []artifact.Batch{
		{
			Key: "alpha/one/" + fixture.alpha.Descriptor.ID.String(), Contents: []artifact.Content{fixture.alpha},
			Artifacts: []artifact.Descriptor{fixture.declared},
			Lineage:   []artifact.Lineage{{Child: fixture.alpha.Descriptor.ID, Parent: fixture.declared.ID, Relation: artifact.RelationDependsOn}},
			Aliases:   []artifact.AliasBinding{{Name: "alpha/head/1", Target: fixture.alpha.Descriptor.ID}},
		},
		{
			Key: "beta/two", Contents: []artifact.Content{fixture.first, fixture.other},
			Lineage: []artifact.Lineage{{Child: fixture.other.Descriptor.ID, Parent: fixture.first.Descriptor.ID, Relation: artifact.RelationDependsOn}},
			Aliases: []artifact.AliasBinding{{Name: "beta.head", Target: fixture.first.Descriptor.ID}},
		},
		{Key: "gamma/3", Contents: []artifact.Content{fixture.later}},
	} {
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := store.Snapshot(ctx); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	return fixture
}

// TestCheckpointCensusAttributesEveryByte builds a store whose two schemas
// have three writers between them and counts its checkpoint.
//
// Every member of the generation is counted, in registration order, and its
// header, its framing and its families' bytes are its file, to the byte. A
// record is attributed to its writer: an artifact to its schema and the
// leading part of the key of the commit that introduced it, so the alpha
// schema's two writers are two families and each holds the bytes its record
// was written in; a bare identity to its kind and that key; a content
// locator and a lineage edge to the document they belong to; a commit and an
// alias to the leading part of their key or name. Each family is dated by
// the last commit that wrote to it, and every decode is timed as often as
// asked.
func TestCheckpointCensusAttributesEveryByte(t *testing.T) {
	t.Parallel()
	fixture := newCensusFixture(t)
	const repeats = 2
	census, err := censusCheckpoint(fixture.root, repeats)
	if err != nil {
		t.Fatal(err)
	}
	const head = 3
	if census.sequence != head || census.generation != checkpointGeneration(head) || len(census.loads) != repeats {
		t.Fatalf("census of generation %q at sequence %d with %d loads, want the one at the head, %d, loaded %d times", census.generation, census.sequence, len(census.loads), head, repeats)
	}
	registered := projections(new(catalogState))
	if len(census.members) != len(registered) {
		t.Fatalf("census counts %d members, the store registers %d projections", len(census.members), len(registered))
	}
	type held struct {
		records, bytes int
		last           uint64
	}
	families := map[string]map[string]held{}
	for index, member := range census.members {
		if member.name != registered[index].name || len(member.restores) != repeats || !slices.IsSorted(member.restores) {
			t.Fatalf("member %d is %s with restores %v, want %s timed %d times, shortest first", index, member.name, member.restores, registered[index].name, repeats)
		}
		file, err := os.Stat(filepath.Join(newestCheckpoints(t, fixture.root), member.name+checkpointExtension))
		if err != nil {
			t.Fatal(err)
		}
		attributed, records := member.header+member.framing, 0
		families[member.name] = map[string]held{}
		for _, family := range member.families {
			attributed, records = attributed+family.bytes, records+family.records
			families[member.name][family.name] = held{family.records, family.bytes, family.last}
		}
		if int64(member.bytes) != file.Size() || attributed != member.bytes || records != member.records {
			t.Fatalf("%s: file of %d bytes counted as %d, of which header, framing and families hold %d; %d records of %d in families", member.name, file.Size(), member.bytes, attributed, records, member.records)
		}
	}
	// encoded is the bytes a record holds in its member's body: its own and
	// the separator after it.
	encoded := func(record any) int {
		t.Helper()
		raw, err := json.Marshal(record)
		if err != nil {
			t.Fatal(err)
		}
		return len(raw) + 1
	}
	artifactBytes := func(descriptor artifact.Descriptor, sequence uint64) int {
		return encoded(artifactCheckpointEntry{Descriptor: descriptor, Sequence: sequence})
	}
	edge := func(child, parent artifact.Content) int {
		return encoded(artifact.Lineage{Child: child.Descriptor.ID, Parent: parent.Descriptor.ID, Relation: artifact.RelationDependsOn})
	}
	want := map[string]map[string]held{
		"artifacts": {
			"test/alpha/v1 by alpha/one":            {1, artifactBytes(fixture.alpha.Descriptor, 1), 1},
			"no schema: evidence bare by alpha/one": {1, artifactBytes(fixture.declared, 1), 1},
			"test/beta/v1 by beta/two":              {2, artifactBytes(fixture.first.Descriptor, 2) + artifactBytes(fixture.other.Descriptor, 2), 2},
			"test/alpha/v1 by gamma":                {1, artifactBytes(fixture.later.Descriptor, head), head},
		},
		"lineage": {
			"test/alpha/v1 by alpha/one": {1, encoded(artifact.Lineage{Child: fixture.alpha.Descriptor.ID, Parent: fixture.declared.ID, Relation: artifact.RelationDependsOn}), 1},
			"test/beta/v1 by beta/two":   {1, edge(fixture.other, fixture.first), 2},
		},
		"aliases": {
			"alpha/head": {1, encoded(snapshotAlias{Name: "alpha/head/1", Target: fixture.alpha.Descriptor.ID}), 1},
			"beta.head":  {1, encoded(snapshotAlias{Name: "beta.head", Target: fixture.first.Descriptor.ID}), 2},
		},
		"causality": {}, "locations": {},
	}
	for member, expected := range want {
		if len(families[member]) != len(expected) {
			t.Fatalf("%s holds families %v, want %v", member, families[member], expected)
		}
		for name, family := range expected {
			if families[member][name] != family {
				t.Fatalf("%s family %q holds %+v, want %+v", member, name, families[member][name], family)
			}
		}
	}
	// A locator and a commit carry coordinates the test does not restate, so
	// they are held by their records and their dates.
	for member, expected := range map[string]map[string]held{
		"contents": {"test/alpha/v1 by alpha/one": {1, 0, 1}, "test/beta/v1 by beta/two": {2, 0, 2}, "test/alpha/v1 by gamma": {1, 0, head}},
		"commits":  {"alpha/one": {1, 0, 1}, "beta/two": {1, 0, 2}, "gamma": {1, 0, head}},
	} {
		if len(families[member]) != len(expected) {
			t.Fatalf("%s holds families %v, want %v", member, families[member], expected)
		}
		for name, family := range expected {
			if got := families[member][name]; got.records != family.records || got.last != family.last || got.bytes == 0 {
				t.Fatalf("%s family %q holds %+v, want %d records dated %d", member, name, got, family.records, family.last)
			}
		}
	}
	if _, err := censusCheckpoint(fixture.root, 0); err == nil {
		t.Fatal("a census that times nothing was not refused")
	}
	if _, err := censusCheckpoint(filepath.Join(t.TempDir(), "absent"), repeats); err == nil || !strings.Contains(err.Error(), "no checkpoint generation") {
		t.Fatalf("census of a store with no checkpoint = %v, want it refused for having none", err)
	}
}

// TestCheckpointCensusFoldsFamiliesUnderTheFloor holds what the census
// writes. With no floor every family has a line and no member a rest. Under
// a floor nothing reaches, each member's families are one rest line that
// holds their bytes and their records and says how many they are. Either
// way a member's lines hold what its header and framing leave of its bytes,
// and the members' bytes are the generation's.
func TestCheckpointCensusFoldsFamiliesUnderTheFloor(t *testing.T) {
	t.Parallel()
	fixture := newCensusFixture(t)
	const everything = 101
	for _, floor := range []float64{0, everything} {
		var output bytes.Buffer
		if err := WriteCheckpointCensus(&output, fixture.root, 1, floor); err != nil {
			t.Fatal(err)
		}
		number := func(text string) int {
			t.Helper()
			value, err := strconv.Atoi(text)
			if err != nil {
				t.Fatalf("census at floor %v prints %q for a count:\n%s", floor, text, output.String())
			}
			return value
		}
		var generation, members, left, families, folded int
		for line := range strings.Lines(output.String()) {
			fields := strings.Fields(line)
			switch fields[0] {
			case "generation":
				generation = number(fields[5])
			case "projection":
				if left != 0 || families != folded {
					t.Fatalf("census at floor %v leaves %d bytes and %d families of a member unprinted:\n%s", floor, left, families-folded, output.String())
				}
				members += number(fields[3])
				left, families, folded = number(fields[3])-number(fields[5])-number(fields[7]), number(fields[11]), 0
			case "family":
				left, folded = left-number(fields[2]), folded+1
				if floor != 0 {
					t.Fatalf("census at floor %v prints a family:\n%s", floor, output.String())
				}
			case "rest":
				left, folded = left-number(fields[2]), folded+number(fields[6])
				if floor == 0 {
					t.Fatalf("census with no floor folds families:\n%s", output.String())
				}
			}
		}
		if left != 0 || families != folded || generation == 0 || generation != members {
			t.Fatalf("census at floor %v: %d bytes of the last member unprinted, %d of its %d families printed, generation of %d bytes against members of %d:\n%s",
				floor, left, folded, families, generation, members, output.String())
		}
	}
}

// TestLeadingPartNamesTheWriter holds the leading part of a key or a name
// to its text before the second separator, and before the first when what
// follows it is empty or holds a digit, since that names a record.
func TestLeadingPartNamesTheWriter(t *testing.T) {
	t.Parallel()
	for name, want := range map[string]string{
		"gate/package/evidence:sha256:05a0":                  "gate/package",
		"stage-receipt/evidence:sha256:4e8d/integrate":       "stage-receipt/evidence",
		"recipe.status.recipe:sha256:3ad7":                   "recipe.status",
		"characterize:tensor-inventory:sha256:12f8":          "characterize:tensor-inventory",
		"rebuild/000003":                                     "rebuild",
		"evaluation/gemma12-native-ifeval-4487fe76/cases/17": "evaluation",
		"qwen4/ifeval/raw":                                   "qwen4/ifeval",
		"closure/":                                           "closure",
		"loop":                                               "loop",
		"":                                                   "",
	} {
		if got := leadingPart(name); got != want {
			t.Errorf("leadingPart(%q) = %q, want %q", name, got, want)
		}
	}
}
