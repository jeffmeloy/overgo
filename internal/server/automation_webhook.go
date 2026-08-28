package server

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"math"
	"net/http"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/workflowruntime"
)

// AutomationWebhookKeyResolver returns caller-owned transient HMAC material
// for one exact immutable key profile. Implementations must not persist key
// bytes in OvergoDB; the workspace clears the returned slice after use.
type AutomationWebhookKeyResolver func(context.Context, artifact.ID) ([]byte, error)

// automationWebhookRequest retains transport values until the active webhook
// policy has selected the exact headers and payload bound.
type automationWebhookRequest struct {
	Name   string
	Header http.Header
	Body   io.Reader
}

// automationWebhookResult projects durable ingress evidence and, only for an
// executable delivery, the common workflow operation admitted from it.
type automationWebhookResult struct {
	DeliveryID artifact.ID                          `json:"delivery_id"`
	Delivery   runrecord.WebhookDelivery            `json:"delivery"`
	Execution  *workflowruntime.AutomationExecution `json:"execution,omitempty"`
}

type automationWebhookWorkspaceAPI interface {
	receiveAutomationWebhook(
		context.Context,
		*operation.Manager,
		automationWebhookRequest,
	) (automationWebhookResult, error)
}

// receiveAutomationWebhook verifies raw transport bytes against exact active
// authority, records the observation, then dispatches only executable ledger
// dispositions. The signing key remains an in-memory server dependency.
func (workspace *AutomationWorkspace) receiveAutomationWebhook(
	ctx context.Context,
	operations *operation.Manager,
	input automationWebhookRequest,
) (automationWebhookResult, error) {
	if workspace == nil || ctx == nil || operations == nil || input.Name == "" || input.Header == nil || input.Body == nil {
		return automationWebhookResult{}, errors.New("automation workspace: webhook ingress authority is absent")
	}
	runtime := workspace.runtime(operations)
	plan, err := runtime.PrepareWebhook(ctx, input.Name)
	if err != nil {
		return automationWebhookResult{}, err
	}
	if plan.Trigger.Kind != recipe.AutomationTriggerWebhook || plan.Trigger.Webhook == nil {
		return automationWebhookResult{}, errors.New("automation workspace: webhook trigger authority differs")
	}
	policy := *plan.Trigger.Webhook
	payload, observedPrefix, err := readBoundedWebhookPayload(input.Body, policy.Payload.MaxBytes)
	if err != nil {
		return automationWebhookResult{}, err
	}

	signature := runrecord.WebhookSignatureUnchecked
	disposition := runrecord.WebhookDeliveryAccepted
	reason := runrecord.WebhookDeliveryReason("")
	provided, present := soleWebhookHeader(input.Header, policy.Signature.Header)
	switch {
	case observedPrefix:
		// A bounded prefix is not the complete byte sequence and therefore is
		// never presented as signature-checked.
	case !present:
		signature = runrecord.WebhookSignatureMissing
	case workspace.webhookKeyResolver == nil:
		return automationWebhookResult{}, errors.New("automation workspace: webhook signing key resolver is absent")
	default:
		key, resolveErr := workspace.webhookKeyResolver(ctx, policy.Signature.Key)
		if resolveErr != nil || len(key) == 0 {
			clear(key)
			disposition = runrecord.WebhookDeliveryRefused
			reason = runrecord.WebhookReasonAuthorityDenied
			break
		}
		signature = verifyWebhookSignature(key, payload, provided)
		clear(key)
	}
	contentType, contentTypePresent := soleWebhookHeader(input.Header, "Content-Type")
	if !observedPrefix && signature == runrecord.WebhookSignatureVerified &&
		(!contentTypePresent || contentType != policy.Payload.ContentType) {
		disposition = runrecord.WebhookDeliveryRefused
		reason = runrecord.WebhookReasonContentTypeMismatch
	}
	idempotency, _ := soleWebhookHeader(input.Header, policy.Payload.IdempotencyHeader)

	delivery, err := (runrecord.WebhookDeliveryLedger{Repository: workspace.store}).Publish(ctx, runrecord.WebhookDeliveryInput{
		Policy: plan.Trigger.ID, Plan: plan.ID, Source: plan.Name,
		IdempotencyKey: idempotency,
		Payload:        payload, PayloadObservedPrefix: observedPrefix,
		Signature: signature, Disposition: disposition, Reason: reason,
	})
	if err != nil {
		return automationWebhookResult{}, err
	}
	result := automationWebhookResult{DeliveryID: delivery.ID, Delivery: delivery}
	if delivery.Disposition != runrecord.WebhookDeliveryAccepted &&
		delivery.Disposition != runrecord.WebhookDeliveryDuplicate &&
		delivery.Disposition != runrecord.WebhookDeliveryReplay {
		return result, nil
	}
	execution, err := runtime.DispatchWebhook(ctx, delivery.ID)
	if err != nil {
		return result, err
	}
	result.Execution = &execution
	return result, nil
}

func readBoundedWebhookPayload(reader io.Reader, maxBytes uint64) ([]byte, bool, error) {
	if reader == nil || maxBytes == 0 {
		return nil, false, errors.New("automation workspace: webhook payload bound is absent")
	}
	limit := int64(math.MaxInt64)
	if maxBytes < math.MaxInt64 {
		limit = int64(maxBytes) + 1
	}
	payload, err := io.ReadAll(io.LimitReader(reader, limit))
	if err != nil {
		return nil, false, err
	}
	return payload, uint64(len(payload)) > maxBytes, nil
}

func soleWebhookHeader(header http.Header, name string) (string, bool) {
	values := header.Values(name)
	returnValue := ""
	if len(values) == 1 {
		returnValue = values[0]
	}
	return returnValue, len(values) == 1 && returnValue != ""
}

func verifyWebhookSignature(key, payload []byte, encoded string) runrecord.WebhookSignatureResult {
	if len(encoded) != sha256.Size*2 || encoded != strings.ToLower(encoded) {
		return runrecord.WebhookSignatureInvalid
	}
	provided, err := hex.DecodeString(encoded)
	if err != nil {
		return runrecord.WebhookSignatureInvalid
	}
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write(payload)
	if !hmac.Equal(mac.Sum(nil), provided) {
		return runrecord.WebhookSignatureInvalid
	}
	return runrecord.WebhookSignatureVerified
}

func (h *Handler) automationWebhook(response http.ResponseWriter, request *http.Request) {
	workspace, ok := h.generator.(automationWebhookWorkspaceAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "automation webhook ingress is unavailable")
		return
	}
	names := request.URL.Query()["name"]
	if len(names) != 1 || names[0] == "" {
		writeError(response, http.StatusBadRequest, "invalid_request", "one automation name is required")
		return
	}
	result, err := workspace.receiveAutomationWebhook(
		context.WithoutCancel(request.Context()), h.operations,
		automationWebhookRequest{Name: names[0], Header: request.Header, Body: request.Body},
	)
	if err != nil {
		writeAutomationResult(response, http.StatusAccepted, result, err)
		return
	}
	writeJSON(response, automationWebhookStatus(result.Delivery), result)
}

func automationWebhookStatus(delivery runrecord.WebhookDelivery) int {
	switch delivery.Disposition {
	case runrecord.WebhookDeliveryAccepted, runrecord.WebhookDeliveryDuplicate, runrecord.WebhookDeliveryReplay:
		return http.StatusAccepted
	}
	switch delivery.Reason {
	case runrecord.WebhookReasonInvalidSignature, runrecord.WebhookReasonMissingSignature:
		return http.StatusUnauthorized
	case runrecord.WebhookReasonContentTypeMismatch:
		return http.StatusUnsupportedMediaType
	case runrecord.WebhookReasonPayloadTooLarge:
		return http.StatusRequestEntityTooLarge
	case runrecord.WebhookReasonPayloadMismatch:
		return http.StatusConflict
	case runrecord.WebhookReasonAuthorityDenied:
		return http.StatusForbidden
	default:
		return http.StatusBadRequest
	}
}
