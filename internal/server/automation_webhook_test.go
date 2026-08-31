package server

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/workflowruntime"
)

func TestWebhookAutomationDispatchAndRecovery(t *testing.T) {
	fixture := newAutomationServerFixture(t)
	defer fixture.store.Close()
	workspace := fixture.handler.generator.(*automationWorkspaceGenerator).AutomationWorkspace
	secret := []byte("server-webhook-secret-never-persisted")
	key := commitAutomationServerBlob(t, fixture.store, artifact.KindProfile, "server-webhook-key-profile")
	schema := commitAutomationServerBlob(t, fixture.store, artifact.KindProfile, "server-webhook-payload-schema")
	authority := commitAutomationServerBlob(t, fixture.store, artifact.KindEvidence, "server-webhook-authority")
	workspace.webhookKeyResolver = func(_ context.Context, requested artifact.ID) ([]byte, error) {
		if requested != key {
			return nil, errors.New("unexpected webhook key authority")
		}
		return bytes.Clone(secret), nil
	}

	var adapterCalls atomic.Int32
	for module, adapter := range workspace.adapters {
		original := adapter
		workspace.adapters[module] = workflowruntime.AdapterFunc(func(
			ctx context.Context,
			request workflowruntime.StepRequest,
		) (map[recipe.PortName]workflowruntime.Value, error) {
			adapterCalls.Add(1)
			return original.Execute(ctx, request)
		})
	}
	definition := publishWebhookAutomationFromAPI(t, fixture, key, schema, authority)
	activate := serveTestRequest(fixture.handler, http.MethodPost, "/automations/activate", marshalAutomationJSON(t, map[string]any{
		"definition": definition,
	}))
	if activate.Code != http.StatusOK {
		t.Fatalf("activate status=%d body=%s", activate.Code, activate.Body.String())
	}
	fixture.handler.config.APIKey = testAPIKey

	payload := []byte(`{"tokens":"webhook value"}`)
	acceptedResponse := serveWebhookRequest(fixture.handler, payload, "request-1", webhookSignature(secret, payload), "")
	accepted := requireWebhookResult(t, acceptedResponse, http.StatusAccepted)
	if accepted.Delivery.Disposition != runrecord.WebhookDeliveryAccepted || accepted.Execution == nil {
		t.Fatalf("accepted webhook = %+v", accepted)
	}
	completed, err := fixture.handler.operations.Wait(t.Context(), accepted.Execution.Operation)
	if err != nil || completed.State != operation.StateCompleted || adapterCalls.Load() != 1 {
		t.Fatalf("accepted operation = (%+v, %v), adapter calls=%d", completed, err, adapterCalls.Load())
	}

	duplicateResponse := serveWebhookRequest(fixture.handler, payload, "request-1", webhookSignature(secret, payload), "")
	duplicate := requireWebhookResult(t, duplicateResponse, http.StatusAccepted)
	if duplicate.Delivery.Disposition != runrecord.WebhookDeliveryDuplicate || duplicate.Execution == nil ||
		duplicate.Execution.Operation != accepted.Execution.Operation || duplicate.Delivery.Plan != accepted.Delivery.Plan ||
		adapterCalls.Load() != 1 {
		t.Fatalf("same-process duplicate = %+v, adapter calls=%d", duplicate, adapterCalls.Load())
	}

	restartedWorkspace, err := (AutomationWorkspaceConfig{
		Store: fixture.store, Catalog: workspace.catalog, Adapters: workspace.adapters,
		Tools: workspace.tools, WebhookKeyResolver: workspace.webhookKeyResolver, Limit: workspace.limit,
	}).Open()
	if err != nil {
		t.Fatal(err)
	}
	restartedGenerator := &automationWorkspaceGenerator{fakeGenerator: &fakeGenerator{}, AutomationWorkspace: restartedWorkspace}
	restarted := newTestHandlerForRepository(t, fixture.store, restartedGenerator)
	restarted.config.APIKey = testAPIKey
	recoveredResponse := serveWebhookRequest(restarted, payload, "request-1", webhookSignature(secret, payload), "")
	recovered := requireWebhookResult(t, recoveredResponse, http.StatusAccepted)
	if recovered.Delivery.Disposition != runrecord.WebhookDeliveryDuplicate || recovered.Execution == nil ||
		recovered.Execution.Operation != accepted.Execution.Operation || recovered.Delivery.Plan != accepted.Delivery.Plan {
		t.Fatalf("restarted duplicate = %+v, accepted = %+v", recovered, accepted)
	}
	recoveredStatus, err := restarted.operations.Wait(t.Context(), recovered.Execution.Operation)
	if err != nil || recoveredStatus.State != operation.StateCompleted || adapterCalls.Load() != 1 {
		t.Fatalf("recovered operation = (%+v, %v), adapter calls=%d", recoveredStatus, err, adapterCalls.Load())
	}

	missingRequest := httptest.NewRequest(
		http.MethodPost, "/automations/webhook?name=webhook-report", bytes.NewReader(payload),
	)
	missingRequest.Header.Set("Content-Type", "application/json")
	missingRequest.Header.Set("Idempotency-Key", "request-missing-signature")
	missingRequest.Header.Set("Authorization", testBearerToken)
	missingRecorder := httptest.NewRecorder()
	restarted.ServeHTTP(missingRecorder, missingRequest)
	missing := requireWebhookResult(t, missingRecorder, http.StatusUnauthorized)
	if missing.Delivery.Reason != runrecord.WebhookReasonMissingSignature || missing.Execution != nil ||
		len(restarted.operations.List()) != 1 {
		t.Fatalf("bearer substitution = %+v, operations=%d", missing, len(restarted.operations.List()))
	}

	upperSignature := "A" + webhookSignature(secret, payload)[1:]
	invalidResponse := serveWebhookRequest(restarted, payload, "request-invalid-signature", upperSignature, "")
	invalid := requireWebhookResult(t, invalidResponse, http.StatusUnauthorized)
	if invalid.Delivery.Reason != runrecord.WebhookReasonInvalidSignature || invalid.Execution != nil ||
		len(restarted.operations.List()) != 1 {
		t.Fatalf("non-lowercase signature = %+v, operations=%d", invalid, len(restarted.operations.List()))
	}

	oversizedPayload := bytes.Repeat([]byte("x"), 1026)
	oversizedResponse := serveWebhookRequest(
		restarted, oversizedPayload, "request-oversized", webhookSignature(secret, oversizedPayload), "",
	)
	oversized := requireWebhookResult(t, oversizedResponse, http.StatusRequestEntityTooLarge)
	if oversized.Delivery.Reason != runrecord.WebhookReasonPayloadTooLarge ||
		!oversized.Delivery.PayloadObservedPrefix || oversized.Delivery.PayloadBytes != 1025 || oversized.Execution != nil {
		t.Fatalf("oversized delivery = %+v", oversized)
	}

	assertWebhookSecretAbsent(t, fixture.store, secret, acceptedResponse.Body.Bytes(), recoveredResponse.Body.Bytes())
}

func publishWebhookAutomationFromAPI(
	t *testing.T,
	fixture automationServerFixture,
	key, schema, authority artifact.ID,
) artifact.ID {
	t.Helper()
	request := AutomationDefinitionInput{
		Name: "webhook-report", Recipe: fixture.definition.ID,
		Trigger: recipe.AutomationTriggerPolicy{Kind: recipe.AutomationTriggerWebhook, Webhook: &recipe.WebhookTriggerPolicy{
			Signature: recipe.WebhookSignaturePolicy{Algorithm: "hmac-sha256", Header: "X-Overgo-Signature", Key: key},
			Payload: recipe.WebhookPayloadPolicy{
				ContentType: "application/json", Schema: schema, MaxBytes: 1024,
				IdempotencyHeader: "Idempotency-Key",
			},
			Rate:      recipe.WebhookRatePolicy{WindowSeconds: 60, MaxDeliveries: 8},
			Authority: authority, Workflow: fixture.definition.ID,
		}},
		Delivery: recipe.AutomationDeliveryPolicy{Kind: recipe.AutomationDeliveryArtifact},
	}
	response := serveTestRequest(fixture.handler, http.MethodPost, "/automations/definitions", marshalAutomationJSON(t, request))
	if response.Code != http.StatusCreated {
		t.Fatalf("publish webhook status=%d body=%s", response.Code, response.Body.String())
	}
	var result struct {
		ID artifact.ID `json:"id"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil || !result.ID.Valid() {
		t.Fatalf("published webhook definition = (%s, %v)", result.ID, err)
	}
	return result.ID
}

func serveWebhookRequest(
	handler http.Handler,
	payload []byte,
	idempotency, signature, authorization string,
) *httptest.ResponseRecorder {
	request := httptest.NewRequest(
		http.MethodPost, "/automations/webhook?name=webhook-report", bytes.NewReader(payload),
	)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Idempotency-Key", idempotency)
	request.Header.Set("X-Overgo-Signature", signature)
	if authorization != "" {
		request.Header.Set("Authorization", authorization)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
}

func requireWebhookResult(
	t *testing.T,
	response *httptest.ResponseRecorder,
	status int,
) automationWebhookResult {
	t.Helper()
	if response.Code != status {
		t.Fatalf("webhook status=%d want=%d body=%s", response.Code, status, response.Body.String())
	}
	var result automationWebhookResult
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil || !result.DeliveryID.Valid() {
		t.Fatalf("webhook result = (%+v, %v), body=%s", result, err, response.Body.String())
	}
	return result
}

func webhookSignature(key, payload []byte) string {
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write(payload)
	return hex.EncodeToString(mac.Sum(nil))
}

func assertWebhookSecretAbsent(t *testing.T, store *overgodb.Store, secret []byte, responses ...[]byte) {
	t.Helper()
	for _, response := range responses {
		if bytes.Contains(response, secret) {
			t.Fatal("webhook response exposed signing key material")
		}
	}
	result, err := store.Query(t.Context(), overgodb.Query{
		MaxResults: 512, Projection: overgodb.ProjectArtifacts,
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, descriptor := range result.Artifacts {
		content, found, readErr := artifact.ReadContent(t.Context(), store, descriptor.ID)
		if readErr != nil {
			t.Fatal(readErr)
		}
		if found && bytes.Contains(content.Data, secret) {
			t.Fatalf("artifact %s exposed signing key material", descriptor.ID)
		}
	}
}
