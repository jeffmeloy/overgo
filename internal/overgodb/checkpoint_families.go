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
	"strings"
	"time"
	"unicode"

	"overgo/internal/artifact"
	"overgo/internal/processmeasure"
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
		keys := familyKeys(&state)
		for index := range census.members {
			if err := census.members[index].count(read, keys); err != nil {
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
	member := memberCensus{name: name, bytes: len(document), header: len(line) + 1}
	member.restores, err = timed(repeats, func() error {
		return projections(new(catalogState))[index].view.restore(body)
	})
	return member, err
}

// count attributes the member's body to families by the member's key.
func (m *memberCensus) count(read func(string) ([]byte, error), keys map[string]familyKey) error {
	key, keyed := keys[m.name]
	if !keyed {
		return fmt.Errorf("overgodb: the census has no family key for projection %s", m.name)
	}
	document, err := read(m.name)
	if err != nil {
		return err
	}
	var records []json.RawMessage
	if err := json.Unmarshal(document[m.header:], &records); err != nil {
		return fmt.Errorf("overgodb: checkpoint %s: %w", m.name, err)
	}
	counts := map[string]*familyCount{}
	attributed := 0
	for _, record := range records {
		name, sequence, err := key(record)
		if err != nil {
			return fmt.Errorf("overgodb: checkpoint %s: %w", m.name, err)
		}
		name = cmp.Or(name, "unnamed")
		count := counts[name]
		if count == nil {
			count = &familyCount{name: name}
			counts[name] = count
		}
		// A record owns the separator after it: a comma, or the closing bracket.
		count.records++
		count.bytes += len(record) + 1
		count.last = max(count.last, sequence)
		attributed += len(record) + 1
	}
	// What is left is the opening bracket, or both brackets of an empty array.
	m.records, m.framing = len(records), m.bytes-m.header-attributed
	if m.framing != len("[]")-min(m.records, 1) {
		return fmt.Errorf("overgodb: checkpoint %s is not the compact array its writer makes: %d bytes frame %d records", m.name, m.framing, m.records)
	}
	for _, count := range slices.SortedFunc(maps.Values(counts), func(left, right *familyCount) int {
		return cmp.Or(cmp.Compare(right.bytes, left.bytes), cmp.Compare(left.name, right.name))
	}) {
		m.families = append(m.families, *count)
	}
	return nil
}

// familyKeys gives each projection the key of one record of its body, read
// against the generation's restored state.
func familyKeys(state *catalogState) map[string]familyKey {
	introducedBy := make(map[uint64]string, state.commits.count())
	for _, commit := range state.commits.all() {
		introducedBy[commit.sequence] = leadingPart(commit.key)
	}
	owner := func(id artifact.ID) (string, uint64) {
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
// counted together in one line, the rest, with how many they are. The times
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
			if share(family.bytes) < floor {
				rest.bytes += family.bytes
				rest.records += family.records
				rest.last = max(rest.last, family.last)
				small++
				continue
			}
			fmt.Fprintf(&text, "family %s %d %d %.2f%% %d %s\n", member.name, family.bytes, family.records, share(family.bytes), family.last, family.name)
		}
		if small > 0 {
			fmt.Fprintf(&text, "rest %s %d %d %.2f%% %d %d families\n", member.name, rest.bytes, rest.records, share(rest.bytes), rest.last, small)
		}
	}
	_, err = output.Write(text.Bytes())
	return err
}
