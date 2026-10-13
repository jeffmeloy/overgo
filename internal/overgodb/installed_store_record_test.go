package overgodb

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/testutil"
)

// TestInstalledStoreRecordIsWhole reads the record of the rebuilt store put
// into service and holds it to being whole.
//
// The install passed both of the store's consumer checks and swapped, from a
// sequence above the one it ended at. The store it put into service is the
// copy the record of the rebuilt store measured: the same head and sequence,
// and a checkpoint of the same bytes.
//
// The store in service passes overgodb check with no chain broken, stands one
// commit past its checkpoint, the commit of the install's proof, and its
// census names every projection the store registers, in order, with bytes
// that sum to the generation's and no line for a retired family.
//
// Each of the two models' loads is shorter after than before by the figure
// the record gives, and the store's open is the part that shortened.
func TestInstalledStoreRecordIsWhole(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	read := func(name string) (lines map[string][]string, fields map[string]map[string]string) {
		t.Helper()
		raw, err := os.ReadFile(filepath.Join(root, "docs", "verification", name))
		if err != nil {
			t.Fatal(err)
		}
		lines, fields = map[string][]string{}, map[string]map[string]string{}
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
		return lines, fields
	}
	lines, fields := read("installed-store.txt")
	const installPart, servicePart, loadsPart = "== install ==", "== store in service ==", "== loads =="
	number := func(part, name string) int64 {
		t.Helper()
		value, err := strconv.ParseInt(fields[part][name], 10, 64)
		if err != nil || value < 0 {
			t.Fatalf("the record's %s is %q (%v), want a count", name, fields[part][name], err)
		}
		return value
	}
	// first returns the submatches of the first of a part's lines that pattern matches.
	first := func(from map[string][]string, part, pattern string) []string {
		t.Helper()
		expression := regexp.MustCompile(pattern)
		for _, line := range from[part] {
			if found := expression.FindStringSubmatch(line); found != nil {
				return found
			}
		}
		t.Fatalf("the record's part %q has no line matching %s", part, pattern)
		return nil
	}

	install := fields[installPart]
	if install["check-blob-identity"] != "passed" || install["check-published-chains"] != "passed" || install["swapped"] != "true" || install["superseded"] == "" {
		t.Errorf("the install = %v, want both consumer checks passed, the swap made and the superseded store named", install)
	}
	before, after := number(installPart, "sequence-before"), number(installPart, "sequence-after")
	if after >= before {
		t.Errorf("the install went from sequence %d to %d, want a store of fewer commits", before, after)
	}
	first(lines, installPart, `^backup published: \S+ head=`+regexp.QuoteMeta(install["head-before"])+` sequence=`+strconv.FormatInt(before, 10)+`$`)

	// The copy the earlier record measured.
	measured, _ := read("rebuilt-store.txt")
	copied := first(measured, "== copy against the installed store ==", `^copy sequence (\d+) head ([0-9a-f]+) `)
	if copied[1] != strconv.FormatInt(after, 10) || copied[2] != install["head-after"] {
		t.Errorf("the install ended at sequence %d head %s, and the copy the rebuilt store's record measured at %s head %s", after, install["head-after"], copied[1], copied[2])
	}
	copiedBytes := first(measured, "== census of the copy ==", `^generation \d+ sequence \d+ bytes (\d+) `)[1]

	first(lines, servicePart, `^overgodb check:blobs=\d+ verified$`)
	first(lines, servicePart, `^overgodb check:chains=\d+ broken=0$`)
	if sequence := first(lines, servicePart, `^store sequence (\d+) head [0-9a-f]+$`)[1]; sequence != strconv.FormatInt(after+1, 10) {
		t.Errorf("the store in service is at sequence %s, want %d, one commit past the rebuild for the install's proof", sequence, after+1)
	}
	generation := first(lines, servicePart, `^generation \d+ sequence (\d+) bytes (\d+) `)
	if generation[1] != strconv.FormatInt(after, 10) || generation[2] != copiedBytes {
		t.Errorf("the checkpoint in service is at sequence %s with %s bytes, want the copy's %d and %s", generation[1], generation[2], after, copiedBytes)
	}
	var sum int64
	for _, registered := range projections(new(catalogState)) {
		found := first(lines, servicePart, `^projection `+regexp.QuoteMeta(registered.name)+` bytes (\d+) .* decode-microseconds (\d+) `)
		size, err := strconv.ParseInt(found[1], 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		sum += size
	}
	if strconv.FormatInt(sum, 10) != generation[2] {
		t.Errorf("the projections' bytes sum to %d, and the generation's are %s", sum, generation[2])
	}
	if number(servicePart, "retired-lines") != 0 {
		t.Errorf("the census of the store in service has %s lines for retired families, want none", fields[servicePart]["retired-lines"])
	}

	for _, model := range []string{"minicpm5-1b", "qwen35-9b"} {
		loadBefore, loadAfter, saved := number(loadsPart, model+"-load-milliseconds-before"), number(loadsPart, model+"-load-milliseconds-after"), number(loadsPart, model+"-load-milliseconds-saved")
		openBefore, openAfter := number(loadsPart, model+"-store-open-milliseconds-before"), number(loadsPart, model+"-store-open-milliseconds-after")
		if saved <= 0 || loadBefore-loadAfter != saved {
			t.Errorf("%s loads in %d ms after %d before and the record says %d saved, want the difference and a load that shortened", model, loadAfter, loadBefore, saved)
		}
		if openAfter >= openBefore || openBefore > loadBefore || openAfter > loadAfter {
			t.Errorf("%s: the store's open took %d ms before and %d after, of loads of %d and %d, want an open that shortened and is a part of its load", model, openBefore, openAfter, loadBefore, loadAfter)
		}
	}
}
