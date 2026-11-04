package overgodb

import (
	"bytes"
	"cmp"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode"

	"overgo/internal/artifact"
	"overgo/internal/processmeasure"
	"overgo/internal/strictjson"
)

// A checkpoint census counts the checkpoint generation an open loads, the
// bytes every open decodes, by record family. A family is the key that names
// the code writing a record. A record of the artifacts projection is keyed by
// its descriptor's schema and the leading part of the key of the commit that
// introduced it, since one schema has several writers and the key tells them
// apart. A descriptor that states no schema is keyed in its place by its kind
// and its media type, "untyped" for none and "bare" for an identity declared
// and nothing else. A content locator and a location are keyed by the family
// of the artifact they belong to, and a lineage edge by its child's, the
// document that declares the edge. A commit is keyed by the leading part of
// its batch key, an alias by the leading part of its name, and a causal link
// by its trigger.
//
// A family is dated by the latest commit sequence among its records: a
// commit's own, and for every other record the sequence that introduced the
// artifact it belongs to, an alias's target and a causal link's execution
// among them. A family dated far behind the generation is one nothing writes
// any more.
//
// A checkpoint body is a compact JSON array, so the attribution is exact: a
// record owns its own bytes and the separator after it, and what is left of
// a member's file is its header line and the array's opening bracket.

// familySeparators are the characters batch keys and alias names are built
// with.
const familySeparators = "/:."

// halves divides a count of sorted readings to index their middle one.
const halves = 2

// percent scales a share for printing.
const percent = 100

// decimal is the base JSON writes a number in.
const decimal = 10

// leadingPart is what a batch key or an alias name says of its writer: its
// text before the second separator. A second part that is empty or holds a
// digit names one record and not the code that wrote it, so the first part
// stands alone.
func leadingPart(name string) string {
	first := strings.IndexAny(name, familySeparators)
	if first < 0 {
		return name
	}
	second := name[first+1:]
	if end := strings.IndexAny(second, familySeparators); end >= 0 {
		second = second[:end]
	}
	if second == "" || strings.ContainsFunc(second, unicode.IsDigit) {
		return name[:first]
	}
	return name[:first+1+len(second)]
}

// familyCount is what one member's body holds of one family, and the latest
// sequence among its records.
type familyCount struct {
	name           string
	records, bytes int
	last           uint64
}

// memberCensus is one member of the generation: its file's bytes, which are
// its header line, its body's framing and its families' records, and the
// timed restores of its body, shortest first.
type memberCensus struct {
	name                            string
	version                         uint16
	bytes, header, framing, records int
	restores                        []time.Duration
	families                        []familyCount
}

// checkpointCensus is one generation counted: its members in registration
// order, and the timed loads of the whole set, shortest first.
type checkpointCensus struct {
	generation string
	sequence   uint64
	loads      []time.Duration
	members    []memberCensus
}

// familyKey names the family of one record of a member's body and the
// sequence that dates it.
type familyKey func(record json.RawMessage) (string, uint64, error)

// censusCheckpoint counts the newest generation under root that verifies, the
// one an open loads. It times the whole set loaded as an open loads it, every
// member read, verified and restored at once, and each member's restore
// alone, each repeats times, and attributes every member's body to families.
// It reads the generation's files and writes nothing.
func censusCheckpoint(root string, repeats int) (checkpointCensus, error) {
	if repeats < 1 {
		return checkpointCensus{}, fmt.Errorf("overgodb: a census times each decode at least once, not %d times", repeats)
	}
	var refused string
	for _, generation := range slices.Backward(checkpointGenerations(root)) {
		read := checkpointMembers(root, generation)
		_, anchor, loaded, reason := loadCheckpointSet(read)
		if !loaded {
			refused = cmp.Or(refused, reason)
			continue
		}
		load := func() (catalogState, error) {
			state, _, loaded, reason := loadCheckpointSet(read)
			if !loaded {
				return catalogState{}, fmt.Errorf("overgodb: %s", reason)
			}
			return state, nil
		}
		census := checkpointCensus{generation: generation, sequence: anchor.sequence}
		// The timings come first, while the process holds nothing of the
		// generation, since a restore costs by what the collector has to walk.
		var err error
		census.loads, err = timed(repeats, func() error {
			_, err := load()
			return err
		})
		if err != nil {
			return checkpointCensus{}, err
		}
		for index, registered := range projections(new(catalogState)) {
			member, err := timeMember(registered.name, index, read, repeats)
			if err != nil {
				return checkpointCensus{}, err
			}
			census.members = append(census.members, member)
		}
		state, err := load()
		if err != nil {
			return checkpointCensus{}, err
		}
		keys, owner := familyKeys(&state), artifactFamily(&state)
		for index := range census.members {
			if err := census.members[index].count(read, keys, owner); err != nil {
				return checkpointCensus{}, err
			}
		}
		return census, nil
	}
	return checkpointCensus{}, fmt.Errorf("overgodb: no checkpoint generation under %s verifies: %s", root, refused)
}

// timed runs work repeats times and returns its walls, shortest first. Each
// run follows a collection, so it starts as an open does, on a heap that
// holds little.
func timed(repeats int, work func() error) ([]time.Duration, error) {
	walls := make([]time.Duration, 0, repeats)
	for range repeats {
		runtime.GC()
		started := processmeasure.NewStopwatch()
		if err := work(); err != nil {
			return nil, err
		}
		elapsed, err := started.Elapsed()
		if err != nil {
			return nil, err
		}
		walls = append(walls, time.Duration(elapsed))
	}
	slices.Sort(walls)
	return walls, nil
}

// timeMember reads one member and times the restore of its body into a facet
// of its own.
func timeMember(name string, index int, read func(string) ([]byte, error), repeats int) (memberCensus, error) {
	document, err := read(name)
	if err != nil {
		return memberCensus{}, err
	}
	line, body, _ := bytes.Cut(document, []byte{'\n'})
	var header checkpointHeader
	if err := strictjson.DecodeBytes(line, &header); err != nil {
		return memberCensus{}, fmt.Errorf("overgodb: checkpoint %s header: %w", name, err)
	}
	restore, reads := projections(new(catalogState))[index].restorer(header.Version)
	if !reads {
		return memberCensus{}, fmt.Errorf("overgodb: checkpoint %s at version %d is not one this reader reads", name, header.Version)
	}
	member := memberCensus{name: name, version: header.Version, bytes: len(document), header: len(line) + 1}
	member.restores, err = timed(repeats, func() error { return restore(body) })
	return member, err
}

// count attributes the member's body to families by the member's key; lineage's compact body by countLineage.
func (m *memberCensus) count(read func(string) ([]byte, error), keys map[string]familyKey, owner func(artifact.ID) (string, uint64)) error {
	key, keyed := keys[m.name]
	if !keyed {
		return fmt.Errorf("overgodb: the census has no family key for projection %s", m.name)
	}
	document, err := read(m.name)
	if err != nil {
		return err
	}
	if m.name == "lineage" && m.version == lineageProjectionVersion {
		return m.countLineage(document[m.header:], owner)
	}
	var records []json.RawMessage
	if err := json.Unmarshal(document[m.header:], &records); err != nil {
		return fmt.Errorf("overgodb: checkpoint %s: %w", m.name, err)
	}
	counts := familyCounts{}
	attributed := 0
	for _, record := range records {
		name, sequence, err := key(record)
		if err != nil {
			return fmt.Errorf("overgodb: checkpoint %s: %w", m.name, err)
		}
		// A record owns the separator after it: a comma, or the closing bracket.
		counts.add(name, sequence, len(record)+1)
		attributed += len(record) + 1
	}
	// What is left is the opening bracket, or both brackets of an empty array.
	m.records, m.framing = len(records), m.bytes-m.header-attributed
	if m.framing != len("[]")-min(m.records, 1) {
		return fmt.Errorf("overgodb: checkpoint %s is not the compact array its writer makes: %d bytes frame %d records", m.name, m.framing, m.records)
	}
	m.families = counts.sorted()
	return nil
}

// countLineage attributes lineage's compact body: an identity owns its text and the separator after it, by the
// artifact it names, and an edge its three indexes and the separator after each, by its child's; what is left is the
// body's frame and its relations.
func (m *memberCensus) countLineage(body []byte, owner func(artifact.ID) (string, uint64)) error {
	var lineage lineageCheckpointBody
	if err := json.Unmarshal(body, &lineage); err != nil {
		return fmt.Errorf("overgodb: checkpoint %s: %w", m.name, err)
	}
	counts := familyCounts{}
	attributed := 0
	for _, id := range lineage.IDs {
		name, sequence := owner(id)
		bytes := len(id.String()) + len(`"",`)
		counts.add(name, sequence, bytes)
		attributed += bytes
	}
	for at := 0; at+lineageEdgeIndexes <= len(lineage.Edges); at += lineageEdgeIndexes {
		if int64(lineage.Edges[at]) >= int64(len(lineage.IDs)) {
			return fmt.Errorf("overgodb: checkpoint %s: an edge names no identity the body holds", m.name)
		}
		bytes := 0
		for _, index := range lineage.Edges[at : at+lineageEdgeIndexes] {
			bytes += len(strconv.FormatUint(uint64(index), decimal)) + len(",")
		}
		name, sequence := owner(lineage.IDs[lineage.Edges[at]])
		counts.add(name, sequence, bytes)
		attributed += bytes
	}
	// The frame is the body with no identity and no edge, less the closing bracket of each array that holds one, which
	// its last element owns.
	frame, err := json.Marshal(lineageCheckpointBody{IDs: lineage.IDs[:0], Relations: lineage.Relations, Edges: lineage.Edges[:0]})
	if err != nil {
		return err
	}
	m.records, m.framing = len(lineage.IDs)+len(lineage.Edges)/lineageEdgeIndexes, len(body)-attributed
	if m.framing != len(frame)-min(len(lineage.IDs), 1)-min(len(lineage.Edges), 1) {
		return fmt.Errorf("overgodb: checkpoint %s is not the compact body its writer makes: %d bytes frame %d records", m.name, m.framing, m.records)
	}
	m.families = counts.sorted()
	return nil
}

// familyCounts is a member's families by name as its records are counted.
type familyCounts map[string]*familyCount

// add counts one record of the family name, dated sequence, of bytes bytes.
func (c familyCounts) add(name string, sequence uint64, bytes int) {
	name = cmp.Or(name, "unnamed")
	count := c[name]
	if count == nil {
		count = &familyCount{name: name}
		c[name] = count
	}
	count.records++
	count.bytes += bytes
	count.last = max(count.last, sequence)
}

// sorted is the families, the most bytes first, then by name.
func (c familyCounts) sorted() []familyCount {
	families := make([]familyCount, 0, len(c))
	for _, count := range slices.SortedFunc(maps.Values(c), func(left, right *familyCount) int {
		return cmp.Or(cmp.Compare(right.bytes, left.bytes), cmp.Compare(left.name, right.name))
	}) {
		families = append(families, *count)
	}
	return families
}

// artifactFamily gives, read against state, the family of the artifact an
// identity names and the sequence that introduced it.
func artifactFamily(state *catalogState) func(artifact.ID) (string, uint64) {
	introducedBy := make(map[uint64]string, state.commits.count())
	for _, commit := range state.commits.all() {
		introducedBy[commit.sequence] = leadingPart(commit.key)
	}
	return func(id artifact.ID) (string, uint64) {
		record, held := state.artifacts.record(id)
		if !held {
			return "an artifact the generation does not hold", 0
		}
		stated := record.descriptor.Schema
		switch {
		case bare(record.descriptor):
			stated = fmt.Sprintf("no schema: %s bare", id.Kind())
		case stated == "":
			stated = fmt.Sprintf("no schema: %s %s", id.Kind(), cmp.Or(record.descriptor.MediaType, "untyped"))
		}
		return stated + " by " + cmp.Or(introducedBy[record.sequence], "an unkeyed commit"), record.sequence
	}
}

// familyKeys gives each projection the key of one record of its body, read
// against the generation's restored state.
func familyKeys(state *catalogState) map[string]familyKey {
	owner := artifactFamily(state)
	return map[string]familyKey{
		"artifacts": keyedBy(func(entry artifactCheckpointEntry) (string, uint64) { return owner(entry.Descriptor.ID) }),
		"contents":  keyedBy(func(entry contentCheckpointEntry) (string, uint64) { return owner(entry.Artifact) }),
		"lineage":   keyedBy(func(edge artifact.Lineage) (string, uint64) { return owner(edge.Child) }),
		"causality": keyedBy(func(link artifact.CausalLink) (string, uint64) {
			_, sequence := owner(link.Execution)
			return link.Trigger, sequence
		}),
		"locations": keyedBy(func(location artifact.Location) (string, uint64) { return owner(location.Artifact) }),
		"aliases": keyedBy(func(alias snapshotAlias) (string, uint64) {
			_, sequence := owner(alias.Target)
			return leadingPart(alias.Name), sequence
		}),
		"commits": keyedBy(func(commit commitCheckpointEntry) (string, uint64) { return leadingPart(commit.Key), commit.Sequence }),
	}
}

// keyedBy decodes one record of a body as its writer encoded it and gives
// its family and its sequence.
func keyedBy[T any](family func(T) (string, uint64)) familyKey {
	return func(record json.RawMessage) (string, uint64, error) {
		var entry T
		if err := json.Unmarshal(record, &entry); err != nil {
			return "", 0, err
		}
		name, sequence := family(entry)
		return name, sequence, nil
	}
}

// middle is the median of readings sorted shortest first, the later of the
// two middle ones when their count is even.
func middle(sorted []time.Duration) time.Duration { return sorted[len(sorted)/halves] }

// WriteCheckpointCensus writes the census of the checkpoint generation an
// open of the store under root loads, timing each decode repeats times. One
// line gives the generation and its bytes, one the load of the whole set as
// an open makes it, one each member with its restore alone, and under each
// member one line a family: bytes, records, its share of the generation's
// bytes, the latest sequence among its records, and its name. A member's
// families that each hold less than floor percent of the generation are
// counted together in one line, the rest, with how many they are. A retired
// family's line opens with "retired" where another's opens with "family",
// and it is printed whatever it holds, so a census with no such line is of a
// store that holds no record of a retired family under its key. The times
// are the middle reading in microseconds beside the least and the most. It
// writes nothing to the store.
func WriteCheckpointCensus(output io.Writer, root string, repeats int, floor float64) error {
	census, err := censusCheckpoint(root, repeats)
	if err != nil {
		return err
	}
	total := 0
	for _, member := range census.members {
		total += member.bytes
	}
	share := func(bytes int) float64 { return percent * float64(bytes) / float64(total) }
	var text bytes.Buffer
	fmt.Fprintf(&text, "generation %s sequence %d bytes %d repeats %d\n", cmp.Or(census.generation, "flat"), census.sequence, total, repeats)
	fmt.Fprintf(&text, "set load-microseconds %d least %d most %d\n",
		middle(census.loads).Microseconds(), census.loads[0].Microseconds(), census.loads[len(census.loads)-1].Microseconds())
	for _, member := range census.members {
		fmt.Fprintf(&text, "projection %s bytes %d header %d framing %d records %d families %d decode-microseconds %d least %d most %d\n",
			member.name, member.bytes, member.header, member.framing, member.records, len(member.families),
			middle(member.restores).Microseconds(), member.restores[0].Microseconds(), member.restores[len(member.restores)-1].Microseconds())
		var rest familyCount
		small := 0
		for _, family := range member.families {
			line := "family"
			if retiredAt(keyedLike(member.name), family.name) >= 0 {
				line = "retired"
			}
			if share(family.bytes) < floor && line == "family" {
				rest.bytes += family.bytes
				rest.records += family.records
				rest.last = max(rest.last, family.last)
				small++
				continue
			}
			fmt.Fprintf(&text, "%s %s %d %d %.2f%% %d %s\n", line, member.name, family.bytes, family.records, share(family.bytes), family.last, family.name)
		}
		if small > 0 {
			fmt.Fprintf(&text, "rest %s %d %d %.2f%% %d %d families\n", member.name, rest.bytes, rest.records, share(rest.bytes), rest.last, small)
		}
	}
	_, err = output.Write(text.Bytes())
	return err
}
