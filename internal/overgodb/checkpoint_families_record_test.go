package overgodb

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/testutil"
)

// TestCheckpointFamiliesRecordIsWhole reads the record of the installed
// store's checkpoint counted by record family and holds it to being whole.
//
// The census in it is the command's: it names the command and the store's
// sequence, and every projection the store registers is there, in order,
// with its bytes and its decode time. A projection's header, framing,
// family lines and rest are its bytes, to the byte, its family lines and
// rest are its families and its records, the projections' bytes are the
// generation's, and each line's share is what its bytes give.
//
// The table names a writer and the four readers for every family whose
// lines hold a hundredth of the generation or more. Each entry is a family
// of the census with the bytes and the date its lines give; its writer is a
// place in the tree or none in it, each reader a place or none found; and
// it is the development process's exactly when no reader was found.
//
// The closing figures add up from the lines above: the bytes of the
// development process's families, with and without a writer, of the
// families that are read and of what the table leaves unclassed; for each
// projection the development process's bytes and their share of its decode
// time; the sum of those; and the estimate of what an open would save,
// which is the load of the set scaled by how much the slowest projection's
// restore would shorten, since an open restores them at once.
func TestCheckpointFamiliesRecordIsWhole(t *testing.T) {
	t.Parallel()
	raw, err := os.ReadFile(filepath.Join(testutil.RepoRoot(t), "docs", "verification", "checkpoint-families.txt"))
	if err != nil {
		t.Fatal(err)
	}
	number := func(what, text string) int64 {
		t.Helper()
		value, err := strconv.ParseInt(text, 10, 64)
		if err != nil || value < 0 {
			t.Fatalf("the record's %s is %q (%v), want a count", what, text, err)
		}
		return value
	}

	// The census: the generation, the set's load, and each projection with
	// its families.
	type projectionLines struct {
		name                                             string
		bytes, left, records, families, decode, estimate int64
	}
	type familyLines struct {
		bytes, last  int64
		byProjection map[string]int64
	}
	var generation, generationSequence, storeSequence, setLoad int64
	var command bool
	var members []*projectionLines
	census := map[string]*familyLines{}
	fields := map[string]string{}
	var entries []map[string]string
	// The record's three parts each open with a heading of their own.
	const censusPart, tablePart, closingPart = "== census ==", "== writers and readers ==", "== closing =="
	var part string
	for line := range strings.Lines(string(raw)) {
		line = strings.TrimRight(line, "\r\n")
		if strings.HasPrefix(line, "== ") {
			part = line
			continue
		}
		words := strings.Fields(line)
		if key, value, found := strings.Cut(line, ": "); part != censusPart && found && !strings.Contains(key, " ") {
			switch {
			case part == closingPart:
				fields[key] = value
			case part == tablePart && key == "family":
				entries = append(entries, map[string]string{key: value})
			case part == tablePart && len(entries) > 0:
				entries[len(entries)-1][key] = value
			}
			continue
		}
		if part != censusPart || len(words) == 0 {
			continue
		}
		share := func(bytes int64) string { return fmt.Sprintf("%.2f%%", 100*float64(bytes)/float64(generation)) }
		switch {
		case strings.HasPrefix(line, "overgodb checkpoint-families -repo "):
			command = true
		case words[0] == "store" && len(words) == 5 && words[1] == "sequence":
			storeSequence = number("store sequence", words[2])
		case words[0] == "generation" && len(words) == 8:
			generationSequence, generation = number("generation sequence", words[3]), number("generation bytes", words[5])
		case words[0] == "set" && len(words) == 7:
			setLoad = number("set load", words[2])
		case words[0] == "projection" && len(words) == 18:
			member := &projectionLines{
				name: words[1], bytes: number("projection bytes", words[3]), records: number("projection records", words[9]),
				families: number("projection families", words[11]), decode: number("decode time", words[13]),
			}
			member.left = member.bytes - number("projection header", words[5]) - number("projection framing", words[7])
			members = append(members, member)
		case (words[0] == "family" || words[0] == "rest") && len(words) >= 7 && len(members) > 0:
			member := members[len(members)-1]
			bytes, last := number("family bytes", words[2]), number("family date", words[5])
			if words[1] != member.name || words[4] != share(bytes) || last > generationSequence {
				t.Fatalf("the record's line %q is not a line of projection %s with the share %s and a date within the generation", line, member.name, share(bytes))
			}
			member.left, member.records = member.left-bytes, member.records-number("family records", words[3])
			if words[0] == "rest" {
				member.families -= number("rest families", words[6])
				continue
			}
			member.families--
			name := strings.Join(words[6:], " ")
			family := census[name]
			if family == nil {
				family = &familyLines{byProjection: map[string]int64{}}
				census[name] = family
			}
			if family.byProjection[member.name] != 0 {
				t.Fatalf("the record counts family %q twice in projection %s", name, member.name)
			}
			family.bytes, family.last = family.bytes+bytes, max(family.last, last)
			family.byProjection[member.name] = bytes
		}
	}
	registered := projections(new(catalogState))
	if !command || generation == 0 || setLoad == 0 || storeSequence < generationSequence || generationSequence == 0 || len(members) != len(registered) {
		t.Fatalf("the record's census names its command %t, a generation of %d bytes at sequence %d of a store at %d, a set loaded in %d microseconds and %d projections: want all of them, and the store's %d projections",
			command, generation, generationSequence, storeSequence, setLoad, len(members), len(registered))
	}
	var counted int64
	for index, member := range members {
		if member.name != registered[index].name || member.bytes == 0 || member.left != 0 || member.records != 0 || member.families != 0 {
			t.Fatalf("the record's projection %d is %s of %d bytes with %d bytes, %d records and %d families its lines do not hold: want %s, whole",
				index, member.name, member.bytes, member.left, member.records, member.families, registered[index].name)
		}
		counted += member.bytes
	}
	if counted != generation {
		t.Fatalf("the record's projections hold %d bytes and its generation %d", counted, generation)
	}

	// The table: a writer and the four readers for every family of a
	// hundredth of the generation or more.
	const hundredth = 100
	place := regexp.MustCompile(`[A-Za-z0-9_./-]+\.go:[0-9]+`)
	const process, read, absent, unread = "development process", "read", "none in the tree", "none found"
	classed := map[string]map[string]string{}
	totals := map[string]int64{}
	for _, entry := range entries {
		name := entry["family"]
		family, known := census[name]
		if !known || classed[name] != nil {
			t.Fatalf("the record's table names %q, which is not one family of its census named once", name)
		}
		classed[name] = entry
		if number("bytes of "+name, entry["bytes"]) != family.bytes || number("date of "+name, entry["last"]) != family.last {
			t.Fatalf("the record's table gives %q %s bytes dated %s, and its census lines give %d dated %d", name, entry["bytes"], entry["last"], family.bytes, family.last)
		}
		if writer := entry["writer"]; !strings.HasPrefix(writer, absent) && !place.MatchString(writer) {
			t.Fatalf("the record's table gives %q the writer %q: want a place in the tree, or %q", name, writer, absent)
		}
		class := process
		for _, reader := range []string{"load", "evaluation", "training", "scorecard"} {
			switch found := entry[reader]; {
			case strings.HasPrefix(found, unread):
			case place.MatchString(found):
				class = read
			default:
				t.Fatalf("the record's table gives %q the %s reader %q: want a place in the tree, or %q", name, reader, found, unread)
			}
		}
		if entry["class"] != class {
			t.Fatalf("the record's table classes %q as %q, and its readers say %q", name, entry["class"], class)
		}
		kind := class
		if class == process {
			kind = "development-process-with-writer"
			if strings.HasPrefix(entry["writer"], absent) {
				kind = "development-process-without-writer"
			}
			totals["development-process"] += family.bytes
			for _, member := range members {
				totals["development-process-"+member.name] += family.byProjection[member.name]
			}
		}
		totals[kind] += family.bytes
	}
	for name, family := range census {
		if family.bytes*hundredth >= generation && classed[name] == nil {
			t.Fatalf("the record's table leaves out %q, which holds %d of the generation's %d bytes", name, family.bytes, generation)
		}
	}

	// The closing: what the table's classes hold, and the estimate.
	figure := func(name string) int64 { t.Helper(); return number(name, fields[name]) }
	totals["checkpoint"] = generation
	totals["unclassed"] = generation - totals["development-process"] - totals[read]
	for _, name := range []string{"checkpoint", "development-process", "development-process-without-writer", "development-process-with-writer", read, "unclassed"} {
		if figure(name+"-bytes") != totals[name] {
			t.Fatalf("the record closes with %s-bytes %s, and its lines give %d", name, fields[name+"-bytes"], totals[name])
		}
	}
	if want := fmt.Sprintf("%.2f%%", 100*float64(totals["development-process"])/float64(generation)); fields["development-process-share"] != want {
		t.Fatalf("the record closes with the development process's share %q, and its bytes give %s", fields["development-process-share"], want)
	}
	var processor, slowest, slowestAfter int64
	for _, member := range members {
		held := totals["development-process-"+member.name]
		member.estimate = held * member.decode / member.bytes
		if figure("development-process-"+member.name+"-bytes") != held || figure("development-process-"+member.name+"-estimate-microseconds") != member.estimate {
			t.Fatalf("the record closes with %s bytes of the development process in %s and an estimate of %s microseconds, and its lines give %d bytes and %d of the projection's %d",
				fields["development-process-"+member.name+"-bytes"], member.name, fields["development-process-"+member.name+"-estimate-microseconds"], held, member.estimate, member.decode)
		}
		processor += member.estimate
		slowest, slowestAfter = max(slowest, member.decode), max(slowestAfter, member.decode-member.estimate)
	}
	open := setLoad * (slowest - slowestAfter) / slowest
	for name, want := range map[string]int64{
		"estimate-processor-microseconds": processor, "open-load-microseconds": setLoad,
		"slowest-restore-microseconds": slowest, "slowest-restore-after-microseconds": slowestAfter,
		"estimate-open-microseconds": open,
	} {
		if figure(name) != want {
			t.Fatalf("the record closes with %s %s, and its lines give %d", name, fields[name], want)
		}
	}
}
