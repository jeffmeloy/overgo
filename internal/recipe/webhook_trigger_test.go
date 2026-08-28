package recipe

import (
	"testing"

	"overgo/internal/artifact"
)

func TestWebhookTriggerPolicyContract(t *testing.T) {
	id := func(kind artifact.Kind, text string) artifact.ID {
		value, err := artifact.IdentifyBytes(kind, []byte(text))
		if err != nil {
			t.Fatal(err)
		}
		return value
	}
	fixture := AutomationTriggerPolicy{Kind: AutomationTriggerWebhook, Webhook: &WebhookTriggerPolicy{
		Signature: WebhookSignaturePolicy{Algorithm: "hmac-sha256", Header: "X-Overgo-Signature", Key: id(artifact.KindProfile, "webhook-key")},
		Payload:   WebhookPayloadPolicy{ContentType: "application/json", Schema: id(artifact.KindProfile, "webhook-schema"), MaxBytes: 4096, IdempotencyHeader: "Idempotency-Key"},
		Rate:      WebhookRatePolicy{WindowSeconds: 60, MaxDeliveries: 10},
		Authority: id(artifact.KindEvidence, "webhook-authority"), Workflow: id(artifact.KindRecipe, "webhook-workflow"),
	}}
	first, err := fixture.Identify()
	if err != nil || len(first.Lineage()) != 4 {
		t.Fatalf("webhook policy = (%+v, %v)", first, err)
	}
	mutations := []func(*AutomationTriggerPolicy){
		func(value *AutomationTriggerPolicy) { value.Webhook.Signature.Algorithm = "sha1" },
		func(value *AutomationTriggerPolicy) { value.Webhook.Payload.MaxBytes = 0 },
		func(value *AutomationTriggerPolicy) { value.Webhook.Rate.MaxDeliveries = 0 },
		func(value *AutomationTriggerPolicy) { value.Webhook.Authority = artifact.ID{} },
		func(value *AutomationTriggerPolicy) { value.Webhook.Workflow = artifact.ID{} },
	}
	for index, mutate := range mutations {
		candidate := fixture
		webhook := *fixture.Webhook
		candidate.Webhook = &webhook
		mutate(&candidate)
		if _, err := candidate.Identify(); err == nil {
			t.Fatalf("mutation %d was admitted", index)
		}
	}
}
