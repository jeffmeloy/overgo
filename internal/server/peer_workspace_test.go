package server

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

const (
	peerWorkspaceLimit       = 8
	peerWorkspaceApprovedNS  = int64(1_730_000_000_000_000_000)
	peerWorkspacePublishedNS = peerWorkspaceApprovedNS + 1
	peerWorkspaceObservedNS  = peerWorkspacePublishedNS + 1
	peerWorkspaceExpiresNS   = peerWorkspaceObservedNS + int64(time.Hour)
	peerWorkspaceNowNS       = peerWorkspaceObservedNS + int64(time.Minute)
	peerWorkspaceEndpoint    = "https://peer-control.example/v1/generation"
)

type peerWorkspaceGenerator struct {
	*fakeGenerator
	*PeerWorkspace
}

type peerWorkspaceBackend struct{}

func (peerWorkspaceBackend) Stage(context.Context, modelrecipe.PeerReplicaPlacement) error {
	return nil
}
func (peerWorkspaceBackend) Load(context.Context, modelrecipe.PeerReplicaPlacement) error { return nil }
func (peerWorkspaceBackend) ActiveLeases(context.Context, modelrecipe.PeerReplicaPlacement) (int, error) {
	return 0, nil
}
func (peerWorkspaceBackend) Unload(context.Context, modelrecipe.PeerReplicaPlacement) error {
	return nil
}

type peerWorkspaceFixture struct {
	handler     *Handler
	store       *overgodb.Store
	workspace   *PeerWorkspace
	environment artifact.ID
}

func TestPeerControlAPI(t *testing.T) {
	fixture := newPeerWorkspaceFixture(t, "peer-control-api", "")
	peer, publication := publishPeerControlFixture(t, fixture, "api-peer")
	inventory := serveTestRequest(fixture.handler, http.MethodGet, "/peers", "")
	if inventory.Code != http.StatusOK || !strings.Contains(inventory.Body.String(), peer.String()) ||
		!strings.Contains(inventory.Body.String(), publication.String()) || !strings.Contains(inventory.Body.String(), `"available":true`) {
		t.Fatalf("peer inventory status=%d body=%s", inventory.Code, inventory.Body.String())
	}
	state := serveTestRequest(fixture.handler, http.MethodPost, "/peers/state", peerWorkspaceJSON(t, peerStateRequest{
		Peer: peer, State: runrecord.PeerDraining, ChangedUnixNS: peerWorkspaceObservedNS + 1,
	}))
	if state.Code != http.StatusOK || !strings.Contains(state.Body.String(), string(runrecord.PeerDraining)) {
		t.Fatalf("peer state status=%d body=%s", state.Code, state.Body.String())
	}
	operationID := testutil.ArtifactID(t, artifact.KindEvidence, "peer-control-operation")
	planID := testutil.ArtifactID(t, artifact.KindProfile, "peer-control-plan")
	targetID := testutil.ArtifactID(t, artifact.KindProfile, "peer-control-target")
	modelID := testutil.ArtifactID(t, artifact.KindModel, "peer-control-model")
	if _, err := runrecord.PublishPeerReplicaReceipt(t.Context(), fixture.store, runrecord.PeerReplicaReceipt{
		Plan: planID, Operation: operationID, Target: targetID, Artifacts: []artifact.ID{modelID},
		Phase: runrecord.PeerReplicaStage, Attempt: 1, Outcome: runrecord.PeerReplicaSucceeded,
	}); err != nil {
		t.Fatal(err)
	}
	evidence := serveTestRequest(fixture.handler, http.MethodGet, "/peers/evidence?operation="+operationID.String(), "")
	if evidence.Code != http.StatusOK || !strings.Contains(evidence.Body.String(), modelID.String()) ||
		!strings.Contains(evidence.Body.String(), string(runrecord.PeerReplicaStage)) {
		t.Fatalf("peer evidence status=%d body=%s", evidence.Code, evidence.Body.String())
	}
}

func TestPeerControlAuthorization(t *testing.T) {
	fixture := newPeerWorkspaceFixture(t, "peer-control-auth", testAPIKey)
	unauthorized := serveTestRequest(fixture.handler, http.MethodGet, "/peers", "")
	if unauthorized.Code != http.StatusUnauthorized {
		t.Fatalf("unauthorized status=%d body=%s", unauthorized.Code, unauthorized.Body.String())
	}
	request := httptest.NewRequest(http.MethodGet, "/peers", nil)
	request.Header.Set("Authorization", testBearerToken)
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("authorized status=%d body=%s", response.Code, response.Body.String())
	}
}

func TestPeerControlProjectionBounds(t *testing.T) {
	fixture := newPeerWorkspaceFixture(t, "peer-control-bounds", "")
	publishPeerControlFixture(t, fixture, "bounded-one")
	publishPeerControlFixture(t, fixture, "bounded-two")
	response := serveTestRequest(fixture.handler, http.MethodGet, "/peers?limit=1", "")
	var projection PeerInventoryProjection
	if err := json.Unmarshal(response.Body.Bytes(), &projection); err != nil || response.Code != http.StatusOK ||
		len(projection.Peers) != 1 || !projection.Truncated {
		t.Fatalf("bounded projection=%+v status=%d err=%v", projection, response.Code, err)
	}
}

func TestPeerControlRefusalReasons(t *testing.T) {
	fixture := newPeerWorkspaceFixture(t, "peer-control-refusal", "")
	response := serveTestRequest(fixture.handler, http.MethodPost, "/peers/placement", peerWorkspaceJSON(t, modelrecipe.PeerPlacementRequest{
		Model: testutil.ArtifactID(t, artifact.KindModel, "missing-active-model"), Task: recipe.TaskGeneration,
		NowUnixNS: peerWorkspaceNowNS,
	}))
	if response.Code != http.StatusUnprocessableEntity || !strings.Contains(response.Body.String(), "peer_refused") ||
		!strings.Contains(response.Body.String(), "peer replica policy") {
		t.Fatalf("placement refusal status=%d body=%s", response.Code, response.Body.String())
	}
	unavailable := serveTestRequest(newTestHandler(t, &fakeGenerator{}), http.MethodGet, "/peers", "")
	if unavailable.Code != http.StatusNotImplemented {
		t.Fatalf("unavailable peer workspace status=%d body=%s", unavailable.Code, unavailable.Body.String())
	}
}

func newPeerWorkspaceFixture(t *testing.T, name, apiKey string) peerWorkspaceFixture {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	environment := testutil.ArtifactID(t, artifact.KindEvidence, name+"-environment")
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "peer/workspace/environment/" + name, Artifacts: []artifact.Descriptor{{ID: environment}},
	}); err != nil {
		t.Fatal(err)
	}
	workspace, err := (PeerWorkspaceConfig{Store: store, Backend: peerWorkspaceBackend{}, Limit: peerWorkspaceLimit}).Open()
	if err != nil {
		t.Fatal(err)
	}
	workspace.clock = func() time.Time { return time.Unix(0, peerWorkspaceNowNS).UTC() }
	handler, err := New(Config{
		ModelID: testModelID, MaxTokens: testMaxTokens, DefaultTemperature: testNeutralTemperature,
		DefaultTopP: testFullTopP, Analysis: testAnalysisPolicy, Repository: store, APIKey: apiKey,
		MaxStoredResponses: peerWorkspaceLimit,
	}, &peerWorkspaceGenerator{fakeGenerator: &fakeGenerator{}, PeerWorkspace: workspace})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handler.Close() })
	return peerWorkspaceFixture{handler: handler, store: store, workspace: workspace, environment: environment}
}

func publishPeerControlFixture(t *testing.T, fixture peerWorkspaceFixture, name string) (artifact.ID, artifact.ID) {
	t.Helper()
	public, _, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	enroll := serveTestRequest(fixture.handler, http.MethodPost, "/peers/enroll", peerWorkspaceJSON(t, runrecord.PeerEnrollment{
		Name: name, Environment: fixture.environment, PublicKey: public,
		ApprovedBy: "peer-operator", ApprovedUnixNS: peerWorkspaceApprovedNS,
	}))
	var enrolled struct {
		Peer       artifact.ID              `json:"peer"`
		Enrollment runrecord.PeerEnrollment `json:"enrollment"`
	}
	if err := json.Unmarshal(enroll.Body.Bytes(), &enrolled); err != nil || enroll.Code != http.StatusCreated {
		t.Fatalf("enrollment status=%d body=%s err=%v", enroll.Code, enroll.Body.String(), err)
	}
	capability := serveTestRequest(fixture.handler, http.MethodPost, "/peers/capability", peerWorkspaceJSON(t, peerCapabilityRequest{
		Peer: enrolled.Peer, PublishedUnixNS: peerWorkspacePublishedNS,
		Capability: modelrecipe.RemotePeerCapability{
			Environment: fixture.environment, Tasks: []recipe.Task{recipe.TaskGeneration}, Endpoint: peerWorkspaceEndpoint,
		},
	}))
	var published struct {
		PublicationID artifact.ID `json:"publication_id"`
	}
	if err := json.Unmarshal(capability.Body.Bytes(), &published); err != nil || capability.Code != http.StatusCreated {
		t.Fatalf("capability status=%d body=%s err=%v", capability.Code, capability.Body.String(), err)
	}
	heartbeat := serveTestRequest(fixture.handler, http.MethodPost, "/peers/heartbeat", peerWorkspaceJSON(t, runrecord.PeerHeartbeat{
		Peer: enrolled.Peer, Capability: published.PublicationID,
		ObservedUnixNS: peerWorkspaceObservedNS, ExpiresUnixNS: peerWorkspaceExpiresNS,
	}))
	if heartbeat.Code != http.StatusCreated {
		t.Fatalf("heartbeat status=%d body=%s", heartbeat.Code, heartbeat.Body.String())
	}
	return enrolled.Peer, published.PublicationID
}

func peerWorkspaceJSON(t *testing.T, value any) string {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(encoded)
}
