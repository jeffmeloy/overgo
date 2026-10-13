package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/finding"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
)

// TestCensusAcceptanceIsOneCommand holds that accepting a changed census is
// one call on a binding file that names its own base census, so no identity
// is typed. Over a temporary store: a call with nothing changed is refused
// and writes nothing; a call with no reason prints what the registered models
// add or change against the census accepted before, not against the base,
// writes nothing, makes no store where there is none, and is what the
// command's flag reaches from a checkout with changes in it; a call with a
// reason publishes the live census, the finding that names the base, the
// census accepted before and the new one, and that finding's closing with the
// reason, all with their edges in one store commit, keeps the digests it
// computed, and writes a binding the check the acceptance tests use passes
// on. That check refuses a binding whose parts do not belong together.
func TestCensusAcceptanceIsOneCommand(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	storePath := t.TempDir()
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Error(err)
		}
	})
	commit := strings.Repeat("b", 40)
	registerCensusModel(t, store, "frozen", true, true)
	base, err := buildCapabilityCensus(ctx, store, commit, catalogLimit, discovery.NewMemo())
	if err != nil {
		t.Fatal(err)
	}
	baseBatch, err := capabilityCensusBatch(base)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, baseBatch); err != nil {
		t.Fatal(err)
	}
	// A binding starts as its base alone: the census an acceptance froze.
	path := filepath.Join(t.TempDir(), "binding.json")
	if err := os.WriteFile(path, []byte(`{"Base": "`+base.ID.String()+`"}`), 0o600); err != nil {
		t.Fatal(err)
	}

	// written is everything a call may write: the binding file and the store.
	type written struct {
		file string
		head artifact.CommitID
	}
	read := func() written {
		t.Helper()
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		// The command commits through its own handle; this one catches up.
		if err := store.Refresh(ctx); err != nil {
			t.Fatal(err)
		}
		head, _ := store.Head()
		return written{string(data), head}
	}
	accept := func(reason string) (string, error) {
		var output bytes.Buffer
		err := acceptCensus(storePath, commit, path, reason, &output)
		t.Logf("reason %q: error %v, printed:\n%s", reason, err, output.String())
		return output.String(), err
	}
	// writesNothing runs one call that must leave the file and the store alone.
	writesNothing := func(reason string) (string, error) {
		t.Helper()
		before := read()
		output, err := accept(reason)
		if after := read(); after != before {
			t.Fatalf("a call that accepts nothing wrote: before %+v, after %+v", before, after)
		}
		return output, err
	}
	// accepted reads the binding a call with a reason wrote and holds it, and
	// what it names in the store, to the acceptance over previous.
	accepted := func(reason string, previous artifact.ID) acceptedCensus {
		t.Helper()
		before := read()
		output, err := accept(reason)
		if err != nil {
			t.Fatalf("accepting a changed census: %v\n%s", err, output)
		}
		if after := read(); after.file == before.file || after.head == before.head {
			t.Fatal("an acceptance left the binding file or the store as it was")
		}
		var binding acceptedCensus
		if err := jsonfile.DecodeStrict(path, &binding); err != nil {
			t.Fatal(err)
		}
		if binding.Base != base.ID || binding.Current == previous || binding.Current == base.ID {
			t.Fatalf("binding = %+v, want the base kept and a new current census", binding)
		}
		if err := checkCensusDisposition(ctx, store, binding); err != nil {
			t.Fatalf("the written binding fails the acceptance tests' check: %v", err)
		}
		current, err := capabilityCensusCodec.Require(ctx, store, binding.Current)
		if err != nil {
			t.Fatal(err)
		}
		if current.ProducerCommit != commit {
			t.Fatalf("the census names producer %q, want the commit given", current.ProducerCommit)
		}
		if err := checkCapabilityCensus(ctx, store, current, catalogLimit); err != nil {
			t.Fatalf("the accepted census is not the live one: %v", err)
		}
		content, found, err := artifact.ReadContent(ctx, store, binding.Finding)
		if err != nil || !found {
			t.Fatalf("the finding is not in the store: found=%t: %v", found, err)
		}
		filed, err := finding.Parse(content.Data)
		if err != nil {
			t.Fatal(err)
		}
		want := []artifact.ID{base.ID, binding.Current}
		if previous != base.ID {
			want = append(want, previous)
		}
		if len(filed.Evidence) != len(want) || slices.ContainsFunc(want, func(id artifact.ID) bool { return !slices.Contains(filed.Evidence, id) }) {
			t.Fatalf("the finding names %v, want exactly the base, the census accepted before and the new one: %v", filed.Evidence, want)
		}
		// The command supplies the rest of the finding itself: the binding file
		// as its owner surface, the one command as its closure path, and the
		// check of the new census as the check that fails while it stands.
		owner, err := artifact.IdentifyBytes(artifact.KindFile, []byte(path))
		if err != nil {
			t.Fatal(err)
		}
		if filed.Status != finding.StatusOpen || filed.Severity != finding.SeverityLow || !slices.Equal(filed.OwnerSurfaces, []artifact.ID{owner}) ||
			!strings.Contains(filed.ClosurePath, "-accept-census "+path) || !strings.HasSuffix(filed.FailableCheck, "-check-census "+binding.Current.String()) {
			t.Fatalf("the command left a part of the finding to its caller, or named another file or census: %+v", filed)
		}
		closed, found, err := finding.Disposition(ctx, store, binding.Finding)
		if err != nil || !found {
			t.Fatalf("the finding has no disposition: found=%t: %v", found, err)
		}
		resolution, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte(reason))
		if err != nil {
			t.Fatal(err)
		}
		if closed.ID != binding.Disposition || closed.Status != finding.StatusClosed || !slices.Contains(closed.Evidence, resolution) {
			t.Fatalf("the disposition %+v is not the bound closing with the reason given", closed)
		}
		if text, found, err := artifact.ReadContent(ctx, store, resolution); err != nil || !found || string(text.Data) != reason {
			t.Fatalf("the store holds the reason as %q, found=%t: %v", text.Data, found, err)
		}
		// One store commit carries the three, so a failure between them cannot
		// leave a census nothing accepts or a finding nothing closes; and it
		// carries their edges, so the store can walk from the closing to the
		// finding and from the finding to each census it names.
		commits := map[artifact.CommitID]bool{}
		for child, parents := range map[artifact.ID][]artifact.ID{binding.Current: nil, binding.Finding: want, binding.Disposition: {binding.Finding}} {
			introduced, found, err := store.ArtifactIntroduction(ctx, child)
			if err != nil || !found {
				t.Fatalf("%s has no introduction: found=%t: %v", child, found, err)
			}
			commits[introduced.Commit] = true
			edges, err := store.Parents(ctx, child)
			if err != nil {
				t.Fatal(err)
			}
			for _, parent := range parents {
				if !slices.ContainsFunc(edges, func(edge artifact.Lineage) bool { return edge.Parent == parent }) {
					t.Fatalf("the store holds no edge from %s to %s", child, parent)
				}
			}
		}
		if len(commits) != 1 {
			t.Fatalf("the census, the finding and its closing came in %d store commits, want one", len(commits))
		}
		return binding
	}

	if output, err := writesNothing("nothing changed"); err == nil {
		t.Fatalf("a census that did not change was accepted:\n%s", output)
	}

	addedID, addedPath := registerCensusModel(t, store, "added", true, false)
	output, err := writesNothing("")
	if err != nil || !strings.Contains(output, "+ "+addedID.String()) {
		t.Fatalf("a call with no reason did not print the added identity: %v\n%s", err, output)
	}
	// The command's flag reaches the same call, and with no -reason it runs
	// from any checkout: it publishes nothing, so it needs no clean code.
	before := read()
	var printed bytes.Buffer
	if err := runArgs(ctx, []string{"-accept-census", path, "-repo", storePath}, &printed); err != nil || !strings.Contains(printed.String(), "+ "+addedID.String()) {
		t.Fatalf("validate -accept-census with no -reason did not print the added identity: %v\n%s", err, printed.String())
	}
	if after := read(); after != before {
		t.Fatalf("validate -accept-census with no -reason wrote: before %+v, after %+v", before, after)
	}
	// With no reason the store is opened for reading alone: one that is not
	// there is refused, not made.
	absent := t.TempDir()
	if err := acceptCensus(filepath.Join(absent, "store"), commit, path, "", io.Discard); err == nil {
		t.Fatal("a call with no reason read a store that is not there")
	}
	if entries, err := os.ReadDir(absent); err != nil || len(entries) != 0 {
		t.Fatalf("a call with no reason made %v beside a store that is not there: %v", entries, err)
	}
	first := accepted("the added model was registered on purpose", base.ID)
	// An acceptance stands in for -publish-census, so it keeps what that keeps:
	// the digests its pass over the catalog computed, for the checks that follow.
	if _, _, kept := discovery.LoadMemo(ctx, store).Claim(addedPath); !kept {
		t.Fatal("an acceptance did not keep the digest of the weights it hashed")
	}

	if output, err := writesNothing("nothing changed since"); err == nil {
		t.Fatalf("the census just accepted was accepted again:\n%s", output)
	}

	laterID, laterPath := registerCensusModel(t, store, "later", true, true)
	output, err = writesNothing("")
	if err != nil || !strings.Contains(output, "+ "+laterID.String()) || strings.Contains(output, addedID.String()) {
		t.Fatalf("a call with no reason did not print what differs from the census accepted before, and that alone: %v\n%s", err, output)
	}
	second := accepted("the later model was activated for forecasting", first.Current)

	// A model whose file is gone reads as its old entry removed and its new one added.
	if err := os.Remove(laterPath); err != nil {
		t.Fatal(err)
	}
	output, err = writesNothing("")
	entry := laterID.String() + " " + laterPath + " present="
	if err != nil || !strings.Contains(output, "- "+entry+"true") || !strings.Contains(output, "+ "+entry+"false") {
		t.Fatalf("a call with no reason did not print the entry that changed, as it was and as it is: %v\n%s", err, output)
	}

	for name, invalid := range map[string]acceptedCensus{
		"the base as its own replacement":  {Base: base.ID, Current: base.ID, Finding: second.Finding, Disposition: second.Disposition},
		"a base the finding does not name": {Base: second.Current, Current: first.Current, Finding: first.Finding, Disposition: first.Disposition},
		"a finding as its own closing":     {Base: base.ID, Current: second.Current, Finding: second.Finding, Disposition: second.Finding},
		"another finding's closing":        {Base: base.ID, Current: second.Current, Finding: second.Finding, Disposition: first.Disposition},
	} {
		if err := checkCensusDisposition(ctx, store, invalid); err == nil {
			t.Fatalf("the check admitted %s", name)
		}
	}
}
