package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// TestCheckpointFamiliesPrintsTheCensusAndWritesNothing runs the subcommand
// on a store of one commit with a checkpoint at its head. It prints its own
// command line, the store's sequence, the generation at that sequence and
// the commit's records under their families; a second commit the checkpoint
// does not hold moves the store's sequence and not the generation's; and
// the store's files are afterwards what they were. A store with no
// checkpoint is refused.
func TestCheckpointFamiliesPrintsTheCensusAndWritesNothing(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	root := filepath.Join(t.TempDir(), "store")
	commit := func(key, body string, snapshot bool) {
		t.Helper()
		store, err := overgodb.Open(root)
		if err != nil {
			t.Fatal(err)
		}
		contract := artifact.DocumentContract{Kind: artifact.KindOutput, MediaType: "application/json", Schema: "test/counted/v1"}
		content, err := contract.ContentBytes([]byte(body))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, artifact.Batch{Key: key, Contents: []artifact.Content{content}}); err != nil {
			t.Fatal(err)
		}
		if snapshot {
			if _, err := store.Snapshot(ctx); err != nil {
				t.Fatal(err)
			}
		}
		if err := store.Close(); err != nil {
			t.Fatal(err)
		}
	}
	run := func() (string, error) {
		t.Helper()
		var output bytes.Buffer
		flags := flag.NewFlagSet("overgodb checkpoint-families", flag.ContinueOnError)
		flags.SetOutput(io.Discard)
		err := checkpointFamilies(flags, []string{"-repo", root, "-repeats", "1"}, &output)
		return output.String(), err
	}
	// files is every file under the store with its size and modification time.
	files := func() string {
		t.Helper()
		var listing strings.Builder
		if err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
			if err == nil && !info.IsDir() {
				fmt.Fprintf(&listing, "%s %d %s\n", path, info.Size(), info.ModTime())
			}
			return err
		}); err != nil {
			t.Fatal(err)
		}
		return listing.String()
	}

	commit("counted/first", `{"n":1}`, false)
	if output, err := run(); err == nil || !strings.Contains(err.Error(), "no checkpoint generation") {
		t.Fatalf("census of a store with no checkpoint = %v, want it refused; it printed:\n%s", err, output)
	}
	commit("counted/second", `{"n":2}`, true)
	commit("counted/third", `{"n":3}`, false)
	before := files()
	output, err := run()
	if err != nil {
		t.Fatal(err)
	}
	if after := files(); after != before {
		t.Fatalf("the census changed the store's files:\n%s\nwere\n%s", after, before)
	}
	for _, want := range []string{
		"overgodb checkpoint-families -repo " + root + " -repeats 1 -floor 0\nstore sequence 3 head ",
		"\ngeneration 00000000000000000002 sequence 2 bytes ",
		"\nset load-microseconds ",
		"\nprojection artifacts bytes ",
		" 1 test/counted/v1 by counted/first\n",
		" 2 test/counted/v1 by counted/second\n",
		"\nprojection commits bytes ",
		" 1 counted/first\n",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("census output lacks %q:\n%s", want, output)
		}
	}
	if strings.Contains(output, "counted/third") {
		t.Fatalf("census counts a commit its generation does not hold:\n%s", output)
	}
}
