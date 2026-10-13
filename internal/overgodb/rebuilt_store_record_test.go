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

// TestRebuiltStoreRecordIsWhole reads the record of the installed store
// rebuilt without the retired families and holds it to being whole.
//
// The rebuild's report is the command's: it names the command, the source
// and its sequence, and it has one line for each family of the table of
// retired families, in the table's order and under the table's key.
//
// The two censuses are the command's, of the source and of the copy the
// rebuild wrote: every projection the store registers, in order, whose bytes
// are the generation's. The copy's records are the installed store's less
// what the report says was left out: artifacts, lineage edges, locations and
// aliases to the record, content locators less the ones the rebuild dropped
// as well, and its commits are the rebuild's batches. The copy's census has
// no line for a retired family.
//
// Both stores pass overgodb check with the same published chains and none
// broken. The comparison finds every alias that is not of a retired family
// at the target it had, none of a retired family, every artifact of the copy
// as the installed store describes it with the parents it has there, and the
// two scorecards the same line for line.
//
// The closing figures are the censuses': the bytes and the load of each, what
// the copy saves of both and its share, the slowest restore of each, and the
// estimate as the record of the census closes with it.
func TestRebuiltStoreRecordIsWhole(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	raw, err := os.ReadFile(filepath.Join(root, "docs", "verification", "rebuilt-store.txt"))
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
	// The record's parts each open with a heading; a part's fields are its
	// lines of a key with no space, a colon and a value.
	const rebuildPart, checksPart, comparisonPart, installedPart, copyPart, scorecardPart, closingPart = "== rebuild ==", "== checks ==",
		"== copy against the installed store ==", "== census of the installed store ==", "== census of the copy ==", "== scorecard ==", "== closing =="
	lines, fields := map[string][]string{}, map[string]map[string]string{}
	var part string
	for line := range strings.Lines(string(raw)) {
		line = strings.TrimRight(line, "\r\n")
		if strings.HasPrefix(line, "== ") {
			part = line
			fields[part] = map[string]string{}
			continue
		}
		if key, value, found := strings.Cut(line, ": "); found && part != "" && !strings.Contains(key, " ") {
			fields[part][key] = value
			continue
		}
		if part != "" && line != "" {
			lines[part] = append(lines[part], line)
		}
	}
	figure := func(part, name string) int64 { t.Helper(); return number(name, fields[part][name]) }

	// The rebuild: its command, its source, its summary and the retired families.
	type family struct{ leftOut, contents, edges, locations, kept int64 }
	var source, destination string
	var sourceSequence int64
	summary := map[string]int64{}
	totals := map[string]family{}
	reported := 0
	const retiredWords, nameAt = "retired keyed left-out n contents n content-bytes n lineage n locations n kept n", 14
	for _, line := range lines[rebuildPart] {
		words := strings.Fields(line)
		switch {
		case strings.HasPrefix(line, "overgodb rebuild -repo ") && strings.HasSuffix(line, " -without-retired") && len(words) == len(strings.Fields("overgodb rebuild -repo r -destination d -without-retired")):
			source, destination = words[3], words[5]
		case strings.HasPrefix(line, "source sequence "):
			sourceSequence = number("source sequence", words[2])
		case strings.HasPrefix(line, "rebuilt "+destination+": "):
			for _, counted := range regexp.MustCompile(`([a-z_]+)=([0-9]+)`).FindAllStringSubmatch(line, -1) {
				summary[counted[1]] = number(counted[1], counted[2])
			}
		case words[0] == "retired" && len(words) > nameAt:
			if reported == len(retiredFamilies) {
				t.Fatalf("the record reports more retired families than the table's %d: %q", len(retiredFamilies), line)
			}
			want, name := retiredFamilies[reported], strings.Join(words[nameAt:], " ")
			if words[1] != want.keyed || name != want.name || strings.Join(words[:nameAt], " ") != fmt.Sprintf("retired %s left-out %s contents %s content-bytes %s lineage %s locations %s kept %s", words[1], words[3], words[5], words[7], words[9], words[11], words[13]) {
				t.Fatalf("the record's retired line %d is %q, want the table's %s %q in the form %q", reported, line, want.keyed, want.name, retiredWords)
			}
			total := totals[words[1]]
			total.leftOut += number("left out", words[3])
			total.contents += number("contents", words[5])
			total.edges += number("lineage", words[9])
			total.locations += number("locations", words[11])
			total.kept += number("kept", words[13])
			totals[words[1]] = total
			reported++
		}
	}
	if source == "" || destination == "" || sourceSequence == 0 || reported != len(retiredFamilies) || summary["batches"] == 0 {
		t.Fatalf("the record's rebuild names source %q at sequence %d and destination %q, summarizes %v and reports %d of the table's %d retired families: want all of them",
			source, sourceSequence, destination, summary, reported, len(retiredFamilies))
	}

	// The censuses: the generation, the set's load and each projection.
	type projectionLine struct{ bytes, records, decode int64 }
	type census struct {
		sequence, bytes, load, slowest int64
		members                        map[string]projectionLine
	}
	registered := projections(new(catalogState))
	read := func(part, store string) census {
		t.Helper()
		counted := census{members: map[string]projectionLine{}}
		var command bool
		var held int64
		ordered := func(least, middle, most string) bool {
			return number("least", least) <= number("middle", middle) && number("middle", middle) <= number("most", most)
		}
		for _, line := range lines[part] {
			words := strings.Fields(line)
			switch {
			case strings.HasPrefix(line, "overgodb checkpoint-families -repo "+store+" -repeats "):
				command = true
			case words[0] == "store" && len(words) == len(strings.Fields("store sequence n head h")):
				counted.sequence = number("store sequence", words[2])
			case words[0] == "generation" && len(words) == len(strings.Fields("generation g sequence n bytes n repeats n")):
				counted.bytes = number("generation bytes", words[5])
				if number("generation sequence", words[3]) != counted.sequence {
					t.Fatalf("the record's %s is of a generation at sequence %s of a store at %d: want the one at its head", part, words[3], counted.sequence)
				}
			case words[0] == "set" && len(words) == len(strings.Fields("set load-microseconds n least n most n")) && ordered(words[4], words[2], words[6]):
				counted.load = number("set load", words[2])
			case words[0] == "projection" && len(words) == len(strings.Fields("projection p bytes n header n framing n records n families n decode-microseconds n least n most n")) && ordered(words[15], words[13], words[17]):
				if len(counted.members) == len(registered) || words[1] != registered[len(counted.members)].name {
					t.Fatalf("the record's %s counts projection %s out of the order the store registers them in", part, words[1])
				}
				member := projectionLine{bytes: number("projection bytes", words[3]), records: number("projection records", words[9]), decode: number("decode time", words[13])}
				counted.members[words[1]] = member
				held, counted.slowest = held+member.bytes, max(counted.slowest, member.decode)
			}
		}
		if !command || counted.load == 0 || counted.bytes == 0 || len(counted.members) != len(registered) || held != counted.bytes {
			t.Fatalf("the record's %s names its command on %s %t, a set loaded in %d microseconds and %d projections of %d bytes in a generation of %d: want the store's %d projections, whole",
				part, store, command, counted.load, len(counted.members), held, counted.bytes, len(registered))
		}
		return counted
	}
	installed, copied := read(installedPart, source), read(copyPart, destination)
	for name, want := range map[string]int64{
		"artifacts": installed.members["artifacts"].records - totals[artifactKeys].leftOut,
		"contents":  installed.members["contents"].records - totals[artifactKeys].contents - summary["contents_dropped"],
		"lineage":   installed.members["lineage"].records - totals[artifactKeys].edges,
		"locations": installed.members["locations"].records - totals[artifactKeys].locations,
		"aliases":   installed.members["aliases"].records - totals[aliasNames].leftOut,
		"commits":   summary["batches"],
	} {
		reportedAs := map[string]string{"contents": "contents_kept", "commits": "batches"}[name]
		if reportedAs == "" {
			reportedAs = name
		}
		if got := copied.members[name].records; got != want || summary[reportedAs] != want {
			t.Fatalf("the record's copy holds %d records of %s and its rebuild reports %d, and the installed store's less what was left out is %d", got, name, summary[reportedAs], want)
		}
	}
	if installed.sequence != sourceSequence || copied.sequence != summary["batches"] || totals[commitKeys].leftOut > installed.members["commits"].records ||
		figure(copyPart, "retired-lines") != 0 || figure(copyPart, "retired-bytes") != 0 || figure(installedPart, "retired-lines") == 0 || figure(installedPart, "retired-bytes") >= installed.bytes {
		t.Fatalf("the record's censuses are of stores at %d and %d with %s and %s lines of retired families: want the source at %d, the copy at its %d batches, and no retired family in the copy",
			installed.sequence, copied.sequence, fields[installedPart]["retired-lines"], fields[copyPart]["retired-lines"], sourceSequence, summary["batches"])
	}

	// The checks: both stores pass, with the same chains.
	chains := map[string]string{}
	var checked string
	for _, line := range lines[checksPart] {
		if store, found := strings.CutPrefix(line, "overgodb check -repo "); found {
			checked = strings.Fields(store)[0]
		} else if passed := regexp.MustCompile(`^overgodb check:chains=([0-9]+) broken=0$`).FindStringSubmatch(line); passed != nil {
			chains[checked] = passed[1]
		}
	}
	if chains[source] == "" || chains[source] != chains[destination] || number("chains", chains[source]) == 0 {
		t.Fatalf("the record's checks pass %q chains of the installed store and %q of the copy: want both, the same, none broken", chains[source], chains[destination])
	}

	// The comparison: a line's counts are named by its first word and the
	// word before each.
	compared := map[string]int64{}
	for _, line := range lines[comparisonPart] {
		words := strings.Fields(line)
		for at := 2; at < len(words); at++ {
			if value, err := strconv.ParseInt(words[at], 10, 64); err == nil {
				compared[words[0]+" "+words[at-1]] = value
			}
		}
	}
	for name, want := range map[string]int64{
		"installed sequence": sourceSequence, "copy sequence": summary["batches"],
		"aliases installed": installed.members["aliases"].records, "aliases not-of-a-retired-family": summary["aliases"], "aliases same-target-in-copy": summary["aliases"],
		"aliases missing-in-copy": 0, "aliases different-target": 0, "aliases of-a-retired-family": totals[aliasNames].leftOut, "aliases present-in-copy": 0,
		"aliases copy": summary["aliases"], "aliases not-in-installed": 0,
		"artifacts copy": summary["artifacts"], "artifacts descriptor-differs-from-installed": 0,
		"lineage edges-declared-by-copy-artifacts-in-installed": summary["lineage"], "lineage artifacts-whose-parents-differ": 0, "lineage parents-absent-from-copy": 0,
		"contents copy-holds": summary["contents_kept"], "contents held-by-both": summary["contents_kept"], "contents held-by-copy-only": 0,
	} {
		if got, stated := compared[name]; !stated || got != want {
			t.Fatalf("the record's comparison gives %q as %d (stated %t), and its rebuild and censuses give %d", name, got, stated, want)
		}
	}
	scorecards := 0
	for _, line := range lines[scorecardPart] {
		if line == "validate -scorecard -repo "+source || line == "validate -scorecard -repo "+destination {
			scorecards++
		}
	}
	if scorecards != len([]string{source, destination}) || figure(scorecardPart, "scorecard-lines-that-differ") != 0 || figure(scorecardPart, "scorecard-lines-installed") == 0 ||
		figure(scorecardPart, "scorecard-lines-installed") != figure(scorecardPart, "scorecard-lines-copy") ||
		figure(scorecardPart, "scorecard-table-lines-installed") != figure(scorecardPart, "scorecard-table-lines-copy") {
		t.Fatalf("the record's scorecards are %d commands with fields %v: want one of each store, of the same lines, none different", scorecards, fields[scorecardPart])
	}

	// The closing: the censuses' figures, and the estimate as its record closes.
	earlier, err := os.ReadFile(filepath.Join(root, "docs", "verification", "checkpoint-families.txt"))
	if err != nil {
		t.Fatal(err)
	}
	estimated := func(name string) int64 {
		t.Helper()
		found := regexp.MustCompile(`(?m)^` + name + `: ([0-9]+)\r?$`).FindSubmatch(earlier)
		if found == nil {
			t.Fatalf("the record of the census closes with no %s", name)
		}
		return number(name, string(found[1]))
	}
	share := func(part, whole int64) string { return fmt.Sprintf("%.2f%%", 100*float64(part)/float64(whole)) }
	for name, want := range map[string]int64{
		"checkpoint-bytes-installed": installed.bytes, "checkpoint-bytes-copy": copied.bytes, "checkpoint-bytes-saved": installed.bytes - copied.bytes,
		"open-load-microseconds-installed": installed.load, "open-load-microseconds-copy": copied.load, "open-load-microseconds-saved": installed.load - copied.load,
		"slowest-restore-microseconds-installed": installed.slowest, "slowest-restore-microseconds-copy": copied.slowest,
		"estimate-open-load-microseconds": estimated("open-load-microseconds"), "estimate-open-microseconds-saved": estimated("estimate-open-microseconds"),
	} {
		if figure(closingPart, name) != want {
			t.Fatalf("the record closes with %s %s, and its censuses and the estimate's record give %d", name, fields[closingPart][name], want)
		}
	}
	for name, want := range map[string]string{
		"checkpoint-bytes-saved-share": share(installed.bytes-copied.bytes, installed.bytes),
		"open-load-saved-share":        share(installed.load-copied.load, installed.load),
		"estimate-open-saved-share":    share(estimated("estimate-open-microseconds"), estimated("open-load-microseconds")),
	} {
		if fields[closingPart][name] != want {
			t.Fatalf("the record closes with %s %q, and its figures give %s", name, fields[closingPart][name], want)
		}
	}
	if copied.bytes >= installed.bytes || figure(closingPart, "rebuild-seconds") == 0 || figure(closingPart, "store-bytes-copy") >= figure(closingPart, "store-bytes-installed") {
		t.Fatalf("the record closes with a copy of %d bytes beside %d, rebuilt in %s seconds, of %s bytes on disk beside %s: want the copy the smaller of both",
			copied.bytes, installed.bytes, fields[closingPart]["rebuild-seconds"], fields[closingPart]["store-bytes-copy"], fields[closingPart]["store-bytes-installed"])
	}
}
