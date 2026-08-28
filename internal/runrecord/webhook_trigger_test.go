// Package runrecord_test verifies evidence-facing trigger authority without creating an ownership cycle.
package runrecord_test

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestWebhookTriggerPolicyContract(t *testing.T) {
	policy, err := (recipe.AutomationTriggerPolicy{Kind: recipe.AutomationTriggerWebhook, Webhook: &recipe.WebhookTriggerPolicy{
		Signature: recipe.WebhookSignaturePolicy{Algorithm: "hmac-sha256", Header: "X-Signature", Key: testutil.ArtifactID(t, artifact.KindProfile, "key")},
		Payload:   recipe.WebhookPayloadPolicy{ContentType: "application/json", Schema: testutil.ArtifactID(t, artifact.KindProfile, "schema"), MaxBytes: 1, IdempotencyHeader: "Idempotency-Key"},
		Rate:      recipe.WebhookRatePolicy{WindowSeconds: 1, MaxDeliveries: 1}, Authority: testutil.ArtifactID(t, artifact.KindEvidence, "authority"), Workflow: testutil.ArtifactID(t, artifact.KindRecipe, "workflow"),
	}}).Identify()
	if err != nil || policy.ID.Kind() != artifact.KindProfile {
		t.Fatalf("webhook policy = (%+v, %v)", policy, err)
	}
}
