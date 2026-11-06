package server

import (
	"encoding/json"
	"net/http"
	"strconv"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestADeltaEnvelopeNamesTheDeltaItCarries pins the endpoint a read-only peer or the embedded UI reconciles against: a
// stateless consumer receives a resync directive naming the current snapshot coordinate, a consumer at an exact head
// receives only the ordered coalesced change since under an envelope naming that delta's coordinate, so a window of one
// of two commits names the first and the consumer resuming from it receives the second, and a foreign projection
// contract forces resync instead of an unbridgeable delta.
func TestADeltaEnvelopeNamesTheDeltaItCarries(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	store := overgodbtest.Create(t, root)
	ctx := t.Context()
	first := artifact.Descriptor{ID: testutil.ArtifactID(t, artifact.KindEvidence, "delta-first"), Size: 1}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "delta/base", Artifacts: []artifact.Descriptor{first},
		Aliases: []artifact.AliasBinding{{Name: "view/current", Target: first.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	handler := testcheck.Must(New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, OvergoDBPath: root}, &fakeGenerator{}))(t)
	t.Cleanup(func() { _ = handler.Close() })

	read := func(path string) storeDeltasResponse {
		t.Helper()
		response := serveTestRequest(handler, http.MethodGet, path, "")
		if response.Code != http.StatusOK {
			t.Fatalf("%s: status=%d body=%s", path, response.Code, response.Body.String())
		}
		var decoded storeDeltasResponse
		testcheck.NoError(t, json.Unmarshal(response.Body.Bytes(), &decoded))
		return decoded
	}
	bootstrap := read("/store/deltas")
	if !bootstrap.Resync || bootstrap.Sequence == 0 || bootstrap.ProjectionVersion == "" || bootstrap.Delta != nil {
		t.Fatalf("stateless consumer = %+v, want resync directive", bootstrap)
	}

	second := artifact.Descriptor{ID: testutil.ArtifactID(t, artifact.KindEvidence, "delta-second"), Size: 1}
	third := artifact.Descriptor{ID: testutil.ArtifactID(t, artifact.KindEvidence, "delta-third"), Size: 1}
	firstTarget := first.ID
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "delta/second", Artifacts: []artifact.Descriptor{second},
		Aliases: []artifact.AliasBinding{{Name: "view/current", Target: second.ID, Previous: &firstTarget}},
	}); err != nil {
		t.Fatal(err)
	}
	secondTarget := second.ID
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "delta/third", Artifacts: []artifact.Descriptor{third},
		Aliases: []artifact.AliasBinding{{Name: "view/current", Target: third.ID, Previous: &secondTarget}},
	}); err != nil {
		t.Fatal(err)
	}

	cursor := func(head string, sequence uint64, projection string) string {
		return "/store/deltas?head=" + head + "&sequence=" + strconv.FormatUint(sequence, 10) + "&projection=" + projection
	}
	reconciled := read(cursor(bootstrap.Head, bootstrap.Sequence, bootstrap.ProjectionVersion))
	if reconciled.Resync || reconciled.Delta == nil {
		t.Fatalf("exact-head consumer = %+v, want delta", reconciled)
	}
	delta := reconciled.Delta
	if len(delta.Commits) != 2 || len(delta.Artifacts) != 2 || delta.Truncated {
		t.Fatalf("delta window = %+v", delta)
	}
	if len(delta.Aliases) != 1 || delta.Aliases[0].Name != "view/current" || delta.Aliases[0].Target != third.ID {
		t.Fatalf("alias churn did not coalesce to the final binding: %+v", delta.Aliases)
	}
	if delta.Head.String() != reconciled.Head || delta.Sequence != reconciled.Sequence {
		t.Fatalf("delta head %s@%d differs from authority %s@%d", delta.Head, delta.Sequence, reconciled.Head, reconciled.Sequence)
	}

	window := read(cursor(bootstrap.Head, bootstrap.Sequence, bootstrap.ProjectionVersion) + "&limit=1")
	if window.Delta == nil || !window.Delta.Truncated || len(window.Delta.Commits) != 1 ||
		window.Head != window.Delta.Head.String() || window.Sequence != window.Delta.Sequence {
		t.Fatalf("a window of one commit = %+v, want the first commit under an envelope naming it", window)
	}
	rest := read(cursor(window.Head, window.Sequence, window.ProjectionVersion))
	if rest.Delta == nil || rest.Delta.Truncated || len(rest.Delta.Commits) != 1 || rest.Head != reconciled.Head || rest.Sequence != reconciled.Sequence {
		t.Fatalf("resuming from the window = %+v, want the second commit", rest)
	}

	foreign := read(cursor(bootstrap.Head, bootstrap.Sequence, "foreign"))
	if !foreign.Resync || foreign.Delta != nil {
		t.Fatalf("foreign projection contract = %+v, want resync", foreign)
	}
}
