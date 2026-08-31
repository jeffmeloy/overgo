package runrecord

import (
	"bytes"
	"errors"
	"io"
	"slices"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestWebhookDeliveryLedgerIdempotency(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	policy := publishWebhookPolicyFixture(t, store)
	plan := testutil.ArtifactID(t, artifact.KindProfile, "webhook-plan")
	testutil.PublishArtifact(t, store, plan)
	ledger := WebhookDeliveryLedger{Repository: store}
	payload := bytes.Repeat([]byte(`{"event":"created"}`), 64)
	originalPayload := slices.Clone(payload)
	const rawKey = "raw-idempotency-secret"
	input := WebhookDeliveryInput{
		Policy: policy.ID, Plan: plan, Source: "github", IdempotencyKey: rawKey, Payload: payload,
		Signature: WebhookSignatureVerified, Disposition: WebhookDeliveryAccepted,
	}

	start := make(chan struct{})
	results := make(chan WebhookDelivery, 2)
	errorsSeen := make(chan error, 2)
	var wait sync.WaitGroup
	for range 2 {
		wait.Go(func() {
			<-start
			record, publishErr := ledger.Publish(ctx, input)
			if publishErr != nil {
				errorsSeen <- publishErr
				return
			}
			results <- record
		})
	}
	close(start)
	wait.Wait()
	close(results)
	close(errorsSeen)
	for publishErr := range errorsSeen {
		t.Fatal(publishErr)
	}
	var accepted, duplicate WebhookDelivery
	for record := range results {
		switch record.Disposition {
		case WebhookDeliveryAccepted:
			accepted = record
		case WebhookDeliveryDuplicate:
			duplicate = record
		default:
			t.Fatalf("concurrent disposition = %q", record.Disposition)
		}
	}
	if !accepted.ID.Valid() || accepted.Plan != plan || duplicate.Plan != plan || duplicate.Original != accepted.ID || duplicate.Payload != accepted.Payload ||
		accepted.CausalRoot() != accepted.ID || duplicate.CausalRoot() != accepted.ID {
		t.Fatalf("concurrent records = accepted %+v duplicate %+v", accepted, duplicate)
	}
	payload[0] ^= 0xff
	assertWebhookPayload(t, store, accepted.Payload, originalPayload)

	current, found, err := ledger.Current(ctx, policy.ID, "github", rawKey)
	if err != nil || !found || current.ID != accepted.ID || current.Plan != plan {
		t.Fatalf("current original = (%+v, %v, %v)", current, found, err)
	}
	assertWebhookRelation(t, store, duplicate.ID, accepted.ID, artifact.RelationDuplicateOf)
	assertWebhookRelation(t, store, duplicate.ID, plan, artifact.RelationDependsOn)
	assertWebhookRelation(t, store, accepted.ID, plan, artifact.RelationDependsOn)
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	store, err = overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ledger.Repository = store
	current, found, err = ledger.Current(ctx, policy.ID, "github", rawKey)
	if err != nil || !found || current.ID != accepted.ID || current.Plan != plan {
		t.Fatalf("reopened original = (%+v, %v, %v)", current, found, err)
	}
	driftPlan := testutil.ArtifactID(t, artifact.KindProfile, "replacement-webhook-plan")
	testutil.PublishArtifact(t, store, driftPlan)
	if _, err := ledger.Publish(ctx, WebhookDeliveryInput{
		Policy: policy.ID, Source: "github", IdempotencyKey: "planless-key", Payload: originalPayload,
		Signature: WebhookSignatureVerified, Disposition: WebhookDeliveryAccepted,
	}); err == nil {
		t.Fatal("accepted delivery omitted its exact compiled plan")
	}
	restarted, err := ledger.Publish(ctx, WebhookDeliveryInput{
		Policy: policy.ID, Plan: driftPlan, Source: "github", IdempotencyKey: rawKey, Payload: originalPayload,
		Signature: WebhookSignatureVerified, Disposition: WebhookDeliveryAccepted,
	})
	if err != nil || restarted.ID != duplicate.ID || restarted.Original != accepted.ID || restarted.Plan != plan {
		t.Fatalf("restart duplicate = (%+v, %v)", restarted, err)
	}
	replay, err := ledger.Publish(ctx, WebhookDeliveryInput{
		Policy: policy.ID, Plan: plan, Source: "github", IdempotencyKey: rawKey, Payload: originalPayload,
		Signature: WebhookSignatureVerified, Disposition: WebhookDeliveryReplay, Original: accepted.ID,
	})
	if err != nil || replay.CausalRoot() != accepted.ID {
		t.Fatalf("replay = (%+v, %v)", replay, err)
	}
	assertWebhookRelation(t, store, replay.ID, accepted.ID, artifact.RelationDerivedFrom)
	assertWebhookRelation(t, store, replay.ID, plan, artifact.RelationDependsOn)
	if _, err := ledger.Publish(ctx, WebhookDeliveryInput{
		Policy: policy.ID, Plan: driftPlan, Source: "github", IdempotencyKey: rawKey, Payload: originalPayload,
		Signature: WebhookSignatureVerified, Disposition: WebhookDeliveryReplay, Original: accepted.ID,
	}); err == nil {
		t.Fatal("replay accepted a plan different from its original")
	}

	mismatch, err := ledger.Publish(ctx, WebhookDeliveryInput{
		Policy: policy.ID, Plan: driftPlan, Source: "github", IdempotencyKey: rawKey, Payload: []byte(`{"event":"changed"}`),
		Signature: WebhookSignatureVerified, Disposition: WebhookDeliveryAccepted,
	})
	if err != nil || mismatch.Disposition != WebhookDeliveryRefused || mismatch.Reason != WebhookReasonPayloadMismatch ||
		mismatch.Original != accepted.ID || mismatch.Plan != plan || mismatch.CausalRoot() != mismatch.ID {
		t.Fatalf("payload mismatch = (%+v, %v)", mismatch, err)
	}

	for _, fixture := range []WebhookDeliveryInput{
		{Policy: policy.ID, Source: "github", IdempotencyKey: "ignored-key", Payload: []byte("ignored"), Signature: WebhookSignatureUnchecked, Disposition: WebhookDeliveryIgnored, Reason: WebhookReasonInactivePolicy},
		{Policy: policy.ID, Source: "github", IdempotencyKey: "skipped-key", Payload: []byte("skipped"), Signature: WebhookSignatureVerified, Disposition: WebhookDeliverySkipped, Reason: WebhookReasonRateLimited},
		{Policy: policy.ID, Plan: plan, Source: "github", IdempotencyKey: "foreign-content-key", Payload: []byte("bounded"), Signature: WebhookSignatureVerified, Disposition: WebhookDeliveryRefused, Reason: WebhookReasonContentTypeMismatch},
	} {
		if _, err := ledger.Publish(ctx, fixture); err != nil {
			t.Fatal(err)
		}
	}
	observedPrefix := bytes.Repeat([]byte("x"), int(policy.Webhook.Payload.MaxBytes)+1)
	oversized, err := ledger.Publish(ctx, WebhookDeliveryInput{
		Policy: policy.ID, Plan: plan, Source: "github", IdempotencyKey: "oversized-key", Payload: observedPrefix,
		PayloadObservedPrefix: true, Signature: WebhookSignatureVerified, Disposition: WebhookDeliveryAccepted,
	})
	if err != nil || oversized.Disposition != WebhookDeliveryRefused || oversized.Reason != WebhookReasonPayloadTooLarge ||
		!oversized.PayloadObservedPrefix || oversized.PayloadStored || oversized.PayloadBytes != uint64(len(observedPrefix)) {
		t.Fatalf("observed-prefix refusal = (%+v, %v)", oversized, err)
	}
	if _, err := ledger.Publish(ctx, WebhookDeliveryInput{
		Policy: policy.ID, Plan: plan, Source: "github", IdempotencyKey: "false-prefix-key", Payload: []byte("bounded"),
		PayloadObservedPrefix: true, Signature: WebhookSignatureVerified, Disposition: WebhookDeliveryAccepted,
	}); err == nil {
		t.Fatal("bounded body accepted as an observed prefix")
	}
	refused, err := ledger.Publish(ctx, WebhookDeliveryInput{
		Policy: policy.ID, Plan: plan, Source: "github", IdempotencyKey: "unpoisoned-key", Payload: []byte("bounded"),
		Signature: WebhookSignatureInvalid, Disposition: WebhookDeliveryAccepted,
	})
	if err != nil || refused.Disposition != WebhookDeliveryRefused || refused.Reason != WebhookReasonInvalidSignature {
		t.Fatalf("invalid signature refusal = (%+v, %v)", refused, err)
	}
	afterRefusal, err := ledger.Publish(ctx, WebhookDeliveryInput{
		Policy: policy.ID, Plan: plan, Source: "github", IdempotencyKey: "unpoisoned-key", Payload: []byte("bounded"),
		Signature: WebhookSignatureVerified, Disposition: WebhookDeliveryAccepted,
	})
	if err != nil || afterRefusal.Disposition != WebhookDeliveryAccepted {
		t.Fatalf("post-refusal acceptance = (%+v, %v)", afterRefusal, err)
	}

	seen := map[WebhookDeliveryDisposition]bool{}
	_, err = overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{Kind: artifact.KindEvidence, MediaType: WebhookDeliveryMediaType, Schema: WebhookDeliverySchema}},
		Order:     overgodb.DocumentOldestFirst, MaxResults: 32,
	}, webhookDeliveryCodec.Parse, func(view overgodb.DocumentView, record WebhookDelivery) error {
		seen[record.Disposition] = true
		if bytes.Contains(view.Content.Data, []byte(rawKey)) || bytes.Contains(view.Content.Data, originalPayload) {
			return errors.New("webhook record retained transient secret or payload bytes")
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, disposition := range []WebhookDeliveryDisposition{
		WebhookDeliveryAccepted, WebhookDeliveryIgnored, WebhookDeliverySkipped,
		WebhookDeliveryDuplicate, WebhookDeliveryRefused, WebhookDeliveryReplay,
	} {
		if !seen[disposition] {
			t.Fatalf("missing durable disposition %q", disposition)
		}
	}

	descriptor, stream, found, err := store.OpenContent(ctx, accepted.Payload)
	if err != nil || !found {
		t.Fatalf("payload descriptor = (%+v, %v, %v)", descriptor, found, err)
	}
	if closer, ok := stream.(io.Closer); ok {
		if err := closer.Close(); err != nil {
			t.Fatal(err)
		}
	}
	corrupt := slices.Clone(originalPayload)
	corrupt[0] ^= 0xff
	if _, err := store.Commit(ctx, artifact.Batch{Key: "webhook/test/corrupt", Contents: []artifact.Content{{Descriptor: descriptor, Data: corrupt}}}); err == nil {
		t.Fatal("mutated bytes committed under the original payload identity")
	}
	assertWebhookPayload(t, store, accepted.Payload, originalPayload)
}

func publishWebhookPolicyFixture(t *testing.T, repository artifact.Repository) recipe.AutomationTriggerPolicy {
	t.Helper()
	workflow := testutil.ArtifactID(t, artifact.KindRecipe, "webhook-workflow")
	key := testutil.ArtifactID(t, artifact.KindProfile, "webhook-key")
	schema := testutil.ArtifactID(t, artifact.KindProfile, "webhook-schema")
	authority := testutil.ArtifactID(t, artifact.KindEvidence, "webhook-authority")
	for _, id := range []artifact.ID{workflow, key, schema, authority} {
		testutil.PublishArtifact(t, repository, id)
	}
	policy, err := (recipe.AutomationTriggerPolicy{Kind: recipe.AutomationTriggerWebhook, Webhook: &recipe.WebhookTriggerPolicy{
		Signature: recipe.WebhookSignaturePolicy{Algorithm: "hmac-sha256", Header: "X-Signature", Key: key},
		Payload:   recipe.WebhookPayloadPolicy{ContentType: "application/json", Schema: schema, MaxBytes: 1 << 20, IdempotencyHeader: "Idempotency-Key"},
		Rate:      recipe.WebhookRatePolicy{WindowSeconds: 60, MaxDeliveries: 8}, Authority: authority, Workflow: workflow,
	}}).Identify()
	if err != nil {
		t.Fatal(err)
	}
	content, err := policy.ArtifactContent()
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch("webhook/test/policy", []artifact.Content{content}, policy.Lineage(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), repository, batch); err != nil {
		t.Fatal(err)
	}
	return policy
}

func assertWebhookPayload(t *testing.T, reader artifact.Reader, id artifact.ID, want []byte) {
	t.Helper()
	descriptor, stream, found, err := reader.OpenContent(t.Context(), id)
	if err != nil || !found {
		t.Fatalf("payload = (%+v, %v, %v)", descriptor, found, err)
	}
	got, err := io.ReadAll(stream)
	if err != nil || !bytes.Equal(got, want) || descriptor.MediaType != WebhookPayloadMediaType || descriptor.Schema != WebhookPayloadSchema {
		t.Fatalf("payload round trip = (%d bytes, %+v, %v)", len(got), descriptor, err)
	}
}

func assertWebhookRelation(t *testing.T, reader artifact.Reader, child, parent artifact.ID, relation artifact.Relation) {
	t.Helper()
	edges, err := reader.Parents(t.Context(), child)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(edges, artifact.Lineage{Child: child, Parent: parent, Relation: relation}) {
		t.Fatalf("lineage for %s lacks %s %s: %+v", child, relation, parent, edges)
	}
}
