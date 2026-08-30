package server

import (
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// TestHeadBoundDeltaReconciliation pins the endpoint a read-only peer or the
// embedded UI reconciles against: a stateless consumer receives a resync
// directive naming the current snapshot coordinate, a consumer at an exact
// head receives only the ordered coalesced change since, and a foreign
// projection contract forces resync instead of an unbridgeable delta.
func TestHeadBoundDeltaReconciliation(t *testing.T) {
	root := t.TempDir()
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	first := artifact.Descriptor{ID: testutil.ArtifactID(t, artifact.KindEvidence, "delta-first"), Size: 1}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "delta/base", Artifacts: []artifact.Descriptor{first},
		Aliases: []artifact.AliasBinding{{Name: "view/current", Target: first.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	handler, err := New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, OvergoDBPath: root}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handler.Close() })

	response := serveTestRequest(handler, http.MethodGet, "/store/deltas", "")
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var bootstrap storeDeltasResponse
	if err := json.Unmarshal(response.Body.Bytes(), &bootstrap); err != nil {
		t.Fatal(err)
	}
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

	cursor := "/store/deltas?head=" + bootstrap.Head +
		"&sequence=" + strconv.FormatUint(bootstrap.Sequence, 10) +
		"&projection=" + bootstrap.ProjectionVersion
	response = serveTestRequest(handler, http.MethodGet, cursor, "")
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var reconciled storeDeltasResponse
	if err := json.Unmarshal(response.Body.Bytes(), &reconciled); err != nil {
		t.Fatal(err)
	}
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

	response = serveTestRequest(handler, http.MethodGet,
		"/store/deltas?head="+bootstrap.Head+"&sequence="+strconv.FormatUint(bootstrap.Sequence, 10)+"&projection=foreign", "")
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var foreign storeDeltasResponse
	if err := json.Unmarshal(response.Body.Bytes(), &foreign); err != nil {
		t.Fatal(err)
	}
	if !foreign.Resync || foreign.Delta != nil {
		t.Fatalf("foreign projection contract = %+v, want resync", foreign)
	}
}
