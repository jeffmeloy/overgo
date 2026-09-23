package runrecord

import (
	"bytes"
	"testing"

	"overgo/internal/artifact"
)

// TestSelectionCauseRecordIsASummary holds the gate's selection record to
// saying each opaque reader's reason once. Each package used to carry every
// reader's reason itself, so the same text repeated for every package a gate
// selected -- 87 percent of each record, about 140 KB of a 160 KB record on a
// 29-package landing. The histogram still names each reader and its reason
// for every package that reaches it, and a record written the old way still
// reads.
func TestSelectionCauseRecordIsASummary(t *testing.T) {
	t.Parallel()
	result, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("gate result"))
	if err != nil {
		t.Fatal(err)
	}
	const reason = "reads a path the source does not name, so the input graph binds it to every root"
	readers := map[string]string{"overgo/internal/server": reason, "overgo/internal/gate": reason}
	var packages []SelectionPackage
	for _, name := range []string{"overgo/cmd/a", "overgo/cmd/b", "overgo/cmd/c", "overgo/cmd/d"} {
		packages = append(packages, SelectionPackage{Package: name, Step: "test", RuntimeReaders: readers})
	}
	record, err := NewSelectionCauses(SelectionCauseRecord{Result: result, Changed: []string{"docs/plan.json"}, Packages: packages, Limitations: "fixture"})
	if err != nil {
		t.Fatal(err)
	}
	content, err := SelectionCauseCodec.Content(record)
	if err != nil {
		t.Fatal(err)
	}
	if count := bytes.Count(content.Data, []byte(reason)); count != len(readers) {
		t.Fatalf("each reason appears %d times in the record, want once per reader (%d)", count, len(readers))
	}
	parsed, err := SelectionCauseCodec.Parse(content.Data)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range parsed.Packages {
		causes := SelectionCauses(entry, parsed.ReaderReasons)
		if len(causes) != len(readers) || causes[0].Kind != SelectionCauseReader || causes[0].Detail != "overgo/internal/gate: "+reason {
			t.Fatalf("%s causes = %+v", entry.Package, causes)
		}
	}
	legacy, err := SelectionCauseCodec.NewInitial(SelectionCauseRecord{Result: result, Packages: packages[:1], Limitations: "fixture"})
	if err != nil {
		t.Fatal(err)
	}
	if causes := SelectionCauses(legacy.Packages[0], legacy.ReaderReasons); len(causes) != len(readers) {
		t.Fatalf("a record written with reasons per package lost them: %+v", causes)
	}
}
