package runrecord

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/textcheck"
)

const (
	// WebhookDeliveryMediaType identifies immutable inbound-delivery evidence.
	WebhookDeliveryMediaType = "application/vnd.overgo.webhook-delivery+json"
	// WebhookDeliverySchema identifies the delivery ledger contract.
	WebhookDeliverySchema = "overgo/webhook-delivery/v1"
	// WebhookPayloadMediaType identifies verbatim bounded request bodies.
	WebhookPayloadMediaType = "application/octet-stream"
	// WebhookPayloadSchema identifies immutable webhook payload blobs.
	WebhookPayloadSchema = "overgo/webhook-payload/v1"
	// WebhookDeliveryAliasRoot scopes the accepted original by policy and key digest.
	WebhookDeliveryAliasRoot = "automation/webhook-delivery/"
)

// WebhookDeliveryDisposition is the closed inbound-delivery lifecycle.
type WebhookDeliveryDisposition string

const (
	// WebhookDeliveryAccepted owns a new idempotency identity.
	WebhookDeliveryAccepted WebhookDeliveryDisposition = "accepted"
	// WebhookDeliveryIgnored records an inactive ingress policy.
	WebhookDeliveryIgnored WebhookDeliveryDisposition = "ignored"
	// WebhookDeliverySkipped records a rate-limited delivery.
	WebhookDeliverySkipped WebhookDeliveryDisposition = "skipped"
	// WebhookDeliveryDuplicate links an exact retry to its accepted original.
	WebhookDeliveryDuplicate WebhookDeliveryDisposition = "duplicate"
	// WebhookDeliveryRefused records a closed, non-executable refusal.
	WebhookDeliveryRefused WebhookDeliveryDisposition = "refused"
	// WebhookDeliveryReplay records an explicit replay of an accepted original.
	WebhookDeliveryReplay WebhookDeliveryDisposition = "replay"
)

// WebhookSignatureResult records only verification outcome, never signature material.
type WebhookSignatureResult string

const (
	// WebhookSignatureVerified means verification succeeded.
	WebhookSignatureVerified WebhookSignatureResult = "verified"
	// WebhookSignatureInvalid means supplied signature material did not verify.
	WebhookSignatureInvalid WebhookSignatureResult = "invalid"
	// WebhookSignatureMissing means required signature material was absent.
	WebhookSignatureMissing WebhookSignatureResult = "missing"
	// WebhookSignatureUnchecked means verification was intentionally not attempted.
	WebhookSignatureUnchecked WebhookSignatureResult = "unchecked"
)

// WebhookDeliveryReason is a bounded refusal/skip vocabulary, not an error string.
type WebhookDeliveryReason string

const (
	// WebhookReasonInactivePolicy explains an ignored delivery.
	WebhookReasonInactivePolicy WebhookDeliveryReason = "inactive-policy"
	// WebhookReasonRateLimited explains a skipped delivery.
	WebhookReasonRateLimited WebhookDeliveryReason = "rate-limited"
	// WebhookReasonInvalidSignature refuses failed verification.
	WebhookReasonInvalidSignature WebhookDeliveryReason = "invalid-signature"
	// WebhookReasonMissingSignature refuses absent signature material.
	WebhookReasonMissingSignature WebhookDeliveryReason = "missing-signature"
	// WebhookReasonMissingIdempotency refuses an absent external key.
	WebhookReasonMissingIdempotency WebhookDeliveryReason = "missing-idempotency"
	// WebhookReasonPayloadTooLarge refuses a body outside its policy bound.
	WebhookReasonPayloadTooLarge WebhookDeliveryReason = "payload-too-large"
	// WebhookReasonPayloadMismatch refuses key reuse with different bytes.
	WebhookReasonPayloadMismatch WebhookDeliveryReason = "payload-mismatch"
	// WebhookReasonInvalidPayload refuses an empty or malformed body.
	WebhookReasonInvalidPayload WebhookDeliveryReason = "invalid-payload"
	// WebhookReasonAuthorityDenied refuses an unauthorized source.
	WebhookReasonAuthorityDenied WebhookDeliveryReason = "authority-denied"
)

// WebhookDelivery is one immutable inbound observation. Payload is its
// content digest; PayloadStored distinguishes a bounded blob from a refused
// body for which only the digest and byte count were retained.
type WebhookDelivery struct {
	Version       uint16                     `json:"version"`
	Policy        artifact.ID                `json:"policy"`
	Source        string                     `json:"source"`
	Idempotency   artifact.ID                `json:"idempotency,omitzero"`
	Payload       artifact.ID                `json:"payload"`
	PayloadBytes  uint64                     `json:"payload_bytes"`
	PayloadStored bool                       `json:"payload_stored,omitempty"`
	Signature     WebhookSignatureResult     `json:"signature"`
	Disposition   WebhookDeliveryDisposition `json:"disposition"`
	Reason        WebhookDeliveryReason      `json:"reason,omitempty"`
	Original      artifact.ID                `json:"original,omitzero"`
	ID            artifact.ID                `json:"-"`
}

// CausalRoot returns the ingress evidence downstream executions must retain.
func (value WebhookDelivery) CausalRoot() artifact.ID {
	if value.Disposition == WebhookDeliveryDuplicate || value.Disposition == WebhookDeliveryReplay {
		return value.Original
	}
	return value.ID
}

// WebhookDeliveryInput contains transient ingress bytes and secrets. Only
// their content identities and typed outcomes enter WebhookDelivery.
type WebhookDeliveryInput struct {
	Policy         artifact.ID
	Source         string
	IdempotencyKey string
	Payload        []byte
	Signature      WebhookSignatureResult
	Disposition    WebhookDeliveryDisposition
	Reason         WebhookDeliveryReason
	Original       artifact.ID
}

var webhookDeliveryCodec = artifact.JSONDocumentCodec(
	"webhook delivery", artifact.KindEvidence, WebhookDeliveryMediaType, WebhookDeliverySchema,
	canonicalizeWebhookDelivery,
	func(value WebhookDelivery) artifact.ID { return value.ID },
	func(value *WebhookDelivery, id artifact.ID) { value.ID = id },
	func(value WebhookDelivery) WebhookDelivery { return value },
)

var webhookPayloadContract = artifact.DocumentContract{
	Kind: artifact.KindFile, MediaType: WebhookPayloadMediaType, Schema: WebhookPayloadSchema,
}

// WebhookDeliveryLedger owns durable ingress observation and idempotency CAS.
type WebhookDeliveryLedger struct{ Repository artifact.Repository }

// Require loads one exact delivery record for recovery or replay.
func (ledger WebhookDeliveryLedger) Require(ctx context.Context, id artifact.ID) (WebhookDelivery, error) {
	return webhookDeliveryCodec.Require(ctx, ledger.Repository, id)
}

// Current resolves the accepted original for one policy-scoped external key.
func (ledger WebhookDeliveryLedger) Current(
	ctx context.Context,
	policy artifact.ID,
	source, idempotencyKey string,
) (WebhookDelivery, bool, error) {
	idempotency, err := webhookIdempotency(policy, source, idempotencyKey)
	if err != nil {
		return WebhookDelivery{}, false, err
	}
	return ledger.resolve(ctx, policy, idempotency)
}

// Publish records one delivery. Accepted publication atomically claims its
// scoped idempotency digest; contenders publish duplicate or mismatch evidence
// linked to the accepted original instead of silently returning it.
func (ledger WebhookDeliveryLedger) Publish(ctx context.Context, input WebhookDeliveryInput) (WebhookDelivery, error) {
	if ctx == nil || ledger.Repository == nil || !textcheck.LowerIdentifier(input.Source, len(input.Source)) ||
		input.Disposition == WebhookDeliveryDuplicate {
		return WebhookDelivery{}, errors.New("run record: invalid webhook delivery input")
	}
	policy, err := recipe.RequireAutomationTriggerPolicy(ctx, ledger.Repository, input.Policy)
	if err != nil || policy.Kind != recipe.AutomationTriggerWebhook || policy.Webhook == nil {
		return WebhookDelivery{}, errors.Join(errors.New("run record: webhook policy is absent"), err)
	}
	idempotency := artifact.ID{}
	if input.IdempotencyKey != "" {
		idempotency, err = webhookIdempotency(policy.ID, input.Source, input.IdempotencyKey)
		if err != nil {
			return WebhookDelivery{}, err
		}
	}
	payloadID, err := artifact.IdentifyBytes(artifact.KindFile, input.Payload)
	if err != nil {
		return WebhookDelivery{}, err
	}
	value := WebhookDelivery{
		Version: artifact.InitialDocumentVersion, Policy: policy.ID, Source: input.Source,
		Idempotency: idempotency, Payload: payloadID, PayloadBytes: uint64(len(input.Payload)),
		Signature: input.Signature, Disposition: input.Disposition, Reason: input.Reason, Original: input.Original,
	}
	var payload *artifact.Content
	switch {
	case len(input.Payload) == 0:
		value.Disposition, value.Reason, value.Original = WebhookDeliveryRefused, WebhookReasonInvalidPayload, artifact.ID{}
	case uint64(len(input.Payload)) > policy.Webhook.Payload.MaxBytes:
		value.Disposition, value.Reason, value.Original = WebhookDeliveryRefused, WebhookReasonPayloadTooLarge, artifact.ID{}
	default:
		content, contentErr := webhookPayloadContract.ContentBytes(input.Payload)
		if contentErr != nil {
			value.Disposition, value.Reason, value.Original = WebhookDeliveryRefused, WebhookReasonPayloadTooLarge, artifact.ID{}
			break
		}
		value.PayloadStored = true
		payload = &content
	}
	if value.Disposition == WebhookDeliveryAccepted {
		switch value.Signature {
		case WebhookSignatureInvalid:
			value.Disposition, value.Reason = WebhookDeliveryRefused, WebhookReasonInvalidSignature
		case WebhookSignatureMissing:
			value.Disposition, value.Reason = WebhookDeliveryRefused, WebhookReasonMissingSignature
		case WebhookSignatureVerified:
			if !value.Idempotency.Valid() {
				value.Disposition, value.Reason = WebhookDeliveryRefused, WebhookReasonMissingIdempotency
			}
		default:
			return WebhookDelivery{}, errors.New("run record: accepted webhook signature is unchecked")
		}
	}
	if value.Disposition == WebhookDeliveryAccepted {
		return ledger.publishAccepted(ctx, value, *payload)
	}
	if value.Disposition == WebhookDeliveryReplay {
		return ledger.publishReplay(ctx, value)
	}
	value, err = webhookDeliveryCodec.New(value)
	if err != nil {
		return WebhookDelivery{}, err
	}
	return value, ledger.publish(ctx, value, payload, false)
}

func (ledger WebhookDeliveryLedger) publishAccepted(
	ctx context.Context,
	value WebhookDelivery,
	payload artifact.Content,
) (WebhookDelivery, error) {
	if current, found, err := ledger.resolve(ctx, value.Policy, value.Idempotency); err != nil {
		return WebhookDelivery{}, err
	} else if found {
		return ledger.publishCollision(ctx, current, value, &payload)
	}
	identified, err := webhookDeliveryCodec.New(value)
	if err != nil {
		return WebhookDelivery{}, err
	}
	if err = ledger.publish(ctx, identified, &payload, true); err == nil {
		return identified, nil
	}
	winner, found, resolveErr := ledger.resolve(ctx, value.Policy, value.Idempotency)
	if resolveErr == nil && found {
		return ledger.publishCollision(ctx, winner, value, &payload)
	}
	return WebhookDelivery{}, errors.Join(err, resolveErr)
}

func (ledger WebhookDeliveryLedger) publishCollision(
	ctx context.Context,
	original, attempted WebhookDelivery,
	payload *artifact.Content,
) (WebhookDelivery, error) {
	if original.Disposition != WebhookDeliveryAccepted || original.Policy != attempted.Policy ||
		original.Source != attempted.Source || original.Idempotency != attempted.Idempotency {
		return WebhookDelivery{}, errors.New("run record: webhook idempotency alias differs")
	}
	attempted.Original = original.ID
	if original.Payload == attempted.Payload && original.PayloadBytes == attempted.PayloadBytes {
		attempted.Disposition, attempted.Reason = WebhookDeliveryDuplicate, ""
		payload = nil
	} else {
		attempted.Disposition, attempted.Reason = WebhookDeliveryRefused, WebhookReasonPayloadMismatch
	}
	identified, err := webhookDeliveryCodec.New(attempted)
	if err != nil {
		return WebhookDelivery{}, err
	}
	return identified, ledger.publish(ctx, identified, payload, false)
}

func (ledger WebhookDeliveryLedger) publishReplay(
	ctx context.Context,
	value WebhookDelivery,
) (WebhookDelivery, error) {
	if !value.Original.Valid() || value.Signature != WebhookSignatureVerified || !value.Idempotency.Valid() || !value.PayloadStored {
		return WebhookDelivery{}, errors.New("run record: invalid webhook replay")
	}
	original, err := ledger.Require(ctx, value.Original)
	if err != nil || original.Disposition != WebhookDeliveryAccepted || original.Policy != value.Policy ||
		original.Source != value.Source || original.Idempotency != value.Idempotency ||
		original.Payload != value.Payload || original.PayloadBytes != value.PayloadBytes {
		return WebhookDelivery{}, errors.Join(errors.New("run record: webhook replay differs from original"), err)
	}
	identified, err := webhookDeliveryCodec.New(value)
	if err != nil {
		return WebhookDelivery{}, err
	}
	return identified, ledger.publish(ctx, identified, nil, false)
}

func (ledger WebhookDeliveryLedger) publish(
	ctx context.Context,
	value WebhookDelivery,
	payload *artifact.Content,
	claim bool,
) error {
	content, err := webhookDeliveryCodec.Content(value)
	if err != nil {
		return err
	}
	contents := []artifact.Content{content}
	parents := []artifact.ID{value.Policy}
	if value.PayloadStored {
		parents = append(parents, value.Payload)
	}
	if payload != nil {
		contents = append(contents, *payload)
	}
	lineage := artifact.DependencyLineage(value.ID, parents...)
	if value.Original.Valid() {
		relation := artifact.RelationDependsOn
		switch value.Disposition {
		case WebhookDeliveryDuplicate:
			relation = artifact.RelationDuplicateOf
		case WebhookDeliveryReplay:
			relation = artifact.RelationDerivedFrom
		}
		lineage = append(lineage, artifact.Lineage{Child: value.ID, Parent: value.Original, Relation: relation})
	}
	var aliases []artifact.AliasBinding
	key := "automation/webhook-delivery/record/" + value.ID.String()
	if claim {
		aliases = []artifact.AliasBinding{{Name: webhookDeliveryAlias(value.Policy, value.Idempotency), Target: value.ID}}
		key, err = uniquePublicationKey("automation/webhook-delivery/claim/", value.ID)
		if err != nil {
			return err
		}
	}
	batch, err := artifact.NewDocumentBatch(key, contents, lineage, aliases)
	if err == nil {
		_, err = artifact.CommitBatch(ctx, ledger.Repository, batch)
	}
	if errors.Is(err, artifact.ErrNoChange) {
		return nil
	}
	return err
}

func (ledger WebhookDeliveryLedger) resolve(
	ctx context.Context,
	policy, idempotency artifact.ID,
) (WebhookDelivery, bool, error) {
	return webhookDeliveryCodec.Resolve(ctx, ledger.Repository, webhookDeliveryAlias(policy, idempotency))
}

func webhookDeliveryAlias(policy, idempotency artifact.ID) string {
	return WebhookDeliveryAliasRoot + policy.String() + "/" + idempotency.String()
}

func webhookIdempotency(policy artifact.ID, source, key string) (artifact.ID, error) {
	if policy.Kind() != artifact.KindProfile || !textcheck.LowerIdentifier(source, len(source)) || key == "" {
		return artifact.ID{}, errors.New("run record: invalid webhook idempotency input")
	}
	return artifact.JSONID(artifact.KindEvidence, struct {
		Policy artifact.ID `json:"policy"`
		Source string      `json:"source"`
		Key    string      `json:"key"`
	}{Policy: policy, Source: source, Key: key})
}

func canonicalizeWebhookDelivery(value *WebhookDelivery) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Policy.Kind() != artifact.KindProfile ||
		!textcheck.LowerIdentifier(value.Source, len(value.Source)) || value.Payload.Kind() != artifact.KindFile ||
		!validWebhookSignature(value.Signature) || value.Original.Valid() && value.Original.Kind() != artifact.KindEvidence {
		return errors.New("run record: invalid webhook delivery")
	}
	if value.PayloadStored {
		if _, err := webhookPayloadContract.Descriptor(value.Payload, value.PayloadBytes); err != nil {
			return errors.New("run record: invalid webhook payload storage fact")
		}
	}
	if !value.PayloadStored && (value.Disposition != WebhookDeliveryRefused ||
		value.Reason != WebhookReasonInvalidPayload && value.Reason != WebhookReasonPayloadTooLarge) {
		return errors.New("run record: invalid webhook payload storage fact")
	}
	switch value.Disposition {
	case WebhookDeliveryAccepted:
		if value.Signature != WebhookSignatureVerified || !value.Idempotency.Valid() || value.Reason != "" || value.Original.Valid() || !value.PayloadStored {
			return errors.New("run record: invalid accepted webhook delivery")
		}
	case WebhookDeliveryIgnored:
		if value.Reason != WebhookReasonInactivePolicy || value.Original.Valid() || !value.PayloadStored {
			return errors.New("run record: invalid ignored webhook delivery")
		}
	case WebhookDeliverySkipped:
		if value.Reason != WebhookReasonRateLimited || value.Original.Valid() || !value.PayloadStored {
			return errors.New("run record: invalid skipped webhook delivery")
		}
	case WebhookDeliveryDuplicate:
		if value.Signature != WebhookSignatureVerified || !value.Idempotency.Valid() || value.Reason != "" || !value.Original.Valid() || !value.PayloadStored {
			return errors.New("run record: invalid duplicate webhook delivery")
		}
	case WebhookDeliveryReplay:
		if value.Signature != WebhookSignatureVerified || !value.Idempotency.Valid() || value.Reason != "" || !value.Original.Valid() || !value.PayloadStored {
			return errors.New("run record: invalid replay webhook delivery")
		}
	case WebhookDeliveryRefused:
		if !validWebhookRefusal(*value) {
			return errors.New("run record: invalid refused webhook delivery")
		}
	default:
		return errors.New("run record: foreign webhook delivery disposition")
	}
	return nil
}

func validWebhookSignature(value WebhookSignatureResult) bool {
	switch value {
	case WebhookSignatureVerified, WebhookSignatureInvalid, WebhookSignatureMissing, WebhookSignatureUnchecked:
		return true
	default:
		return false
	}
}

func validWebhookRefusal(value WebhookDelivery) bool {
	switch value.Reason {
	case WebhookReasonInvalidSignature:
		return value.Signature == WebhookSignatureInvalid && !value.Original.Valid()
	case WebhookReasonMissingSignature:
		return value.Signature == WebhookSignatureMissing && !value.Original.Valid()
	case WebhookReasonMissingIdempotency, WebhookReasonInvalidPayload, WebhookReasonPayloadTooLarge, WebhookReasonAuthorityDenied:
		if value.Original.Valid() {
			return false
		}
		switch value.Reason {
		case WebhookReasonInvalidPayload:
			return !value.PayloadStored && value.PayloadBytes == 0
		case WebhookReasonPayloadTooLarge:
			return !value.PayloadStored && value.PayloadBytes > 0
		default:
			return value.PayloadStored
		}
	case WebhookReasonPayloadMismatch:
		return value.Signature == WebhookSignatureVerified && value.Idempotency.Valid() && value.Original.Valid() && value.PayloadStored
	default:
		return false
	}
}
