package capabilityruntime

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestRemotePeerSpilloverRequiresCompatibilityEvidence(t *testing.T) {
	id := func(kind artifact.Kind, label string) artifact.ID { return testutil.ArtifactID(t, kind, label) }
	modelID := id(artifact.KindModel, "peer-model")
	recipeID := id(artifact.KindRecipe, "peer-recipe")
	resourcesID := id(artifact.KindProfile, "peer-resources")
	compatibilityID := id(artifact.KindEvidence, "peer-compatibility")
	capabilityID := id(artifact.KindEvidence, "peer-capability")
	input, responseBody := `{"prompt":"peer"}`, `{"result":"remote"}`
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodPost || request.Header.Get(peerModelHeader) != modelID.String() ||
			request.Header.Get(peerRecipeHeader) != recipeID.String() ||
			request.Header.Get(peerResourcesHeader) != resourcesID.String() ||
			request.Header.Get(peerCompatibilityHeader) != compatibilityID.String() {
			t.Errorf("peer request headers=%v", request.Header)
		}
		body, _ := io.ReadAll(request.Body)
		if string(body) != input {
			t.Errorf("peer input=%q", body)
		}
		_, _ = response.Write([]byte(responseBody))
	}))
	defer server.Close()
	selection := modelrecipe.CapabilityEvidenceSelection{
		Session:    modelrecipe.SessionSpillover,
		Activation: modelrecipe.Activation{Definition: recipe.Definition{Model: modelID, ID: recipeID}},
		Resources:  modelrecipe.ComponentSessionPlan{Identity: resourcesID},
		Peer: modelrecipe.RemotePeerCompatibility{
			ID: compatibilityID, Model: modelID, Recipe: recipeID, Resources: resourcesID,
			PeerCapability: capabilityID,
			Capability:     modelrecipe.RemotePeerCapability{ID: capabilityID, Endpoint: server.URL},
		},
	}
	var output strings.Builder
	if err := ExecuteRemotePeer(t.Context(), server.Client(), selection, strings.NewReader(input), &output); err != nil {
		t.Fatal(err)
	}
	if output.String() != responseBody {
		t.Fatalf("peer output=%q", output.String())
	}
	selection.Peer = modelrecipe.RemotePeerCompatibility{}
	if err := ExecuteRemotePeer(t.Context(), server.Client(), selection, strings.NewReader(input), &output); err == nil {
		t.Fatal("evidence-free peer execution accepted")
	}
}
