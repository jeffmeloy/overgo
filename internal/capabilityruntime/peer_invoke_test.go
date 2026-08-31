package capabilityruntime

import (
	"go/parser"
	"go/token"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func peerInvocationSelection(t *testing.T, endpoint string) modelrecipe.CapabilityEvidenceSelection {
	t.Helper()
	id := func(kind artifact.Kind, label string) artifact.ID { return testutil.ArtifactID(t, kind, label) }
	modelID := id(artifact.KindModel, "peer-invoke-model")
	recipeID := id(artifact.KindRecipe, "peer-invoke-recipe")
	resourcesID := id(artifact.KindProfile, "peer-invoke-resources")
	capabilityID := id(artifact.KindEvidence, "peer-invoke-capability")
	return modelrecipe.CapabilityEvidenceSelection{
		Session:    modelrecipe.SessionSpillover,
		Activation: modelrecipe.Activation{Definition: recipe.Definition{Model: modelID, ID: recipeID}},
		Resources:  modelrecipe.ComponentSessionPlan{Identity: resourcesID},
		Peer: modelrecipe.RemotePeerCompatibility{
			ID: id(artifact.KindEvidence, "peer-invoke-compatibility"), Model: modelID, Recipe: recipeID,
			Resources: resourcesID, PeerCapability: capabilityID,
			Capability: modelrecipe.RemotePeerCapability{
				ID: capabilityID, Endpoint: endpoint, Tasks: []recipe.Task{recipe.TaskGeneration},
			},
		},
	}
}

// TestPeerCapabilityManualAndPlacement pins the exact UTCP boundary of a
// peer capability: the derived manual binds the http-json-stream transport
// to the published endpoint and capability identity, refuses a task the
// capability does not publish, and refuses a selection whose placement
// facts are inconsistent.
func TestPeerCapabilityManualAndPlacement(t *testing.T) {
	selection := peerInvocationSelection(t, "https://peer.example/generate")
	manual, err := PeerCapabilityManual(selection, recipe.TaskGeneration)
	if err != nil {
		t.Fatal(err)
	}
	if manual.Name != "peer.generation" || manual.Effect != agenttool.EffectInspection ||
		manual.Transport.Kind != agenttool.TransportHTTPJSONStream ||
		manual.Transport.URL != selection.Peer.Capability.Endpoint ||
		!strings.Contains(manual.Description, selection.Peer.Capability.ID.String()) {
		t.Fatalf("peer manual = %+v", manual)
	}
	if _, err := PeerCapabilityManual(selection, recipe.TaskTraining); err == nil {
		t.Fatal("unpublished task admitted")
	}
	misplaced := selection
	misplaced.Peer.PeerCapability = testutil.ArtifactID(t, artifact.KindEvidence, "different capability")
	if _, err := PeerCapabilityManual(misplaced, recipe.TaskGeneration); err == nil {
		t.Fatal("inconsistent placement admitted")
	}
}

// TestPeerHTTPJSONStreamInvocationReceipt pins the receipt chain around one
// streamed peer invocation: the admitted receipt lands before bytes leave,
// the completed receipt closes over the capability evidence, and a failed
// transfer leaves a terminal failed receipt carrying the exact error.
func TestPeerHTTPJSONStreamInvocationReceipt(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		_, _ = response.Write([]byte("{\"token\":\"a\"}\n{\"token\":\"b\"}\n"))
	}))
	defer server.Close()
	selection := peerInvocationSelection(t, server.URL)
	manual, err := PeerCapabilityManual(selection, recipe.TaskGeneration)
	if err != nil {
		t.Fatal(err)
	}
	operation := testutil.ArtifactID(t, artifact.KindEvidence, "peer invocation operation")
	var output strings.Builder
	receipt, err := InvokeRemotePeer(
		ctx, server.Client(), store, manual, selection, recipe.TaskGeneration,
		operation, strings.NewReader(`{"prompt":"peer"}`), &output,
	)
	if err != nil {
		t.Fatal(err)
	}
	if receipt.State != runrecord.StageCompleted || !receipt.Previous.Valid() ||
		!strings.Contains(output.String(), `{"token":"b"}`) {
		t.Fatalf("completed receipt = %+v output=%q", receipt, output.String())
	}
	tip, found, err := runrecord.ResolveStageReceipt(ctx, store, operation, PeerInvocationNode)
	if err != nil || !found || tip.ID != receipt.ID {
		t.Fatalf("receipt chain tip = %+v found=%t err=%v", tip, found, err)
	}
	admittedContent, found, err := artifact.ReadContent(ctx, store, receipt.Previous)
	if err != nil || !found {
		t.Fatalf("admitted receipt content = found=%t err=%v", found, err)
	}
	running, err := runrecord.ParseStageReceipt(admittedContent.Data)
	if err != nil || running.State != runrecord.StageRunning || !running.Previous.Valid() {
		t.Fatalf("running receipt = %+v, %v", running, err)
	}

	failing := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.WriteHeader(http.StatusBadGateway)
	}))
	defer failing.Close()
	failedSelection := peerInvocationSelection(t, failing.URL)
	failedManual, err := PeerCapabilityManual(failedSelection, recipe.TaskGeneration)
	if err != nil {
		t.Fatal(err)
	}
	failedOperation := testutil.ArtifactID(t, artifact.KindEvidence, "peer failed operation")
	failed, err := InvokeRemotePeer(
		ctx, failing.Client(), store, failedManual, failedSelection, recipe.TaskGeneration,
		failedOperation, strings.NewReader(`{}`), &strings.Builder{},
	)
	if err == nil || failed.State != runrecord.StageFailed || !strings.Contains(failed.Failure, "502") {
		t.Fatalf("failed receipt = %+v, %v", failed, err)
	}
}

// TestRemoteRecipeCapabilityRequiresExactUTCPManual pins the manual gate: a
// remote recipe capability invocation refuses any manual but the exact one
// its selection derives, and refuses before any receipt exists.
func TestRemoteRecipeCapabilityRequiresExactUTCPManual(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	selection := peerInvocationSelection(t, "https://peer.example/generate")
	forged, err := agenttool.NewManual(agenttool.Manual{
		Name: "peer.generation", Description: "Forged peer manual.",
		Effect: agenttool.EffectInspection,
		Transport: agenttool.Transport{
			Kind: agenttool.TransportHTTPJSONStream, URL: "https://attacker.example/generate",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := RequireExactPeerManual(forged, selection, recipe.TaskGeneration); err == nil ||
		!strings.Contains(err.Error(), "exact UTCP manual") {
		t.Fatalf("forged manual admitted: %v", err)
	}
	operation := testutil.ArtifactID(t, artifact.KindEvidence, "forged peer operation")
	if _, err := InvokeRemotePeer(
		ctx, nil, store, forged, selection, recipe.TaskGeneration,
		operation, strings.NewReader(`{}`), &strings.Builder{},
	); err == nil {
		t.Fatal("forged manual invoked")
	}
	if _, found, err := runrecord.ResolveStageReceipt(ctx, store, operation, PeerInvocationNode); err != nil || found {
		t.Fatalf("refused invocation left a receipt: found=%t err=%v", found, err)
	}
}

// TestCrossLaneCapabilityReuseDoesNotImportRuntimeCode pins the reuse
// boundary structurally: the peer invocation path reaches another lane's
// capability through HTTP and the exact manual only -- its files import no
// serving, inference, or composition runtime code.
func TestCrossLaneCapabilityReuseDoesNotImportRuntimeCode(t *testing.T) {
	forbidden := []string{
		"overgo/internal/inference", "overgo/internal/server",
		"overgo/internal/llamaserver", "overgo/internal/composition",
	}
	for _, file := range []string{"peer.go", "peer_invoke.go"} {
		parsed, err := parser.ParseFile(token.NewFileSet(), filepath.Join(".", file), nil, parser.ImportsOnly)
		if err != nil {
			t.Fatal(err)
		}
		for _, imported := range parsed.Imports {
			path := strings.Trim(imported.Path.Value, `"`)
			for _, runtimePath := range forbidden {
				if path == runtimePath {
					t.Errorf("%s imports runtime package %s", file, path)
				}
			}
		}
	}
}
