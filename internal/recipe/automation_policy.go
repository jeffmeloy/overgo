package recipe

import (
	"context"
	"errors"
	"slices"
	"sort"
	"strings"
	"time"

	"overgo/internal/artifact"
)

const (
	// AutomationTriggerPolicyVersion is the trigger document revision.
	AutomationTriggerPolicyVersion = artifact.InitialDocumentVersion
	// AutomationTriggerPolicyMediaType identifies trigger policy documents.
	AutomationTriggerPolicyMediaType = "application/vnd.overgo.automation-trigger-policy+json"
	// AutomationTriggerPolicySchema identifies the trigger policy contract.
	AutomationTriggerPolicySchema = "overgo/automation-trigger-policy/v2"
	// AutomationDeliveryPolicyVersion is the delivery document revision.
	AutomationDeliveryPolicyVersion = artifact.InitialDocumentVersion
	// AutomationDeliveryPolicyMediaType identifies delivery policy documents.
	AutomationDeliveryPolicyMediaType = "application/vnd.overgo.automation-delivery-policy+json"
	// AutomationDeliveryPolicySchema identifies the delivery policy contract.
	AutomationDeliveryPolicySchema = "overgo/automation-delivery-policy/v1"
)

// AutomationTriggerKind selects one runtime-owned trigger path.
type AutomationTriggerKind string

const (
	// AutomationTriggerManual admits explicit operator or API invocation.
	AutomationTriggerManual AutomationTriggerKind = "manual"
	// AutomationTriggerSchedule admits scheduler-owned due-time invocation.
	AutomationTriggerSchedule AutomationTriggerKind = "schedule"
	// AutomationTriggerWebhook admits signed bounded HTTP delivery.
	AutomationTriggerWebhook AutomationTriggerKind = "webhook"
)

// WebhookSignaturePolicy binds verification to an algorithm, header, and key authority.
type WebhookSignaturePolicy struct {
	Algorithm string      `json:"algorithm"`
	Header    string      `json:"header"`
	Key       artifact.ID `json:"key"`
}

// WebhookPayloadPolicy bounds and types accepted request bodies.
type WebhookPayloadPolicy struct {
	ContentType       string      `json:"content_type"`
	Schema            artifact.ID `json:"schema"`
	MaxBytes          uint64      `json:"max_bytes"`
	IdempotencyHeader string      `json:"idempotency_header"`
}

// WebhookRatePolicy bounds accepted deliveries within one integral window.
type WebhookRatePolicy struct {
	WindowSeconds uint32 `json:"window_seconds"`
	MaxDeliveries uint32 `json:"max_deliveries"`
}

// WebhookTriggerPolicy is exact ingress authority without secret material.
type WebhookTriggerPolicy struct {
	Signature WebhookSignaturePolicy `json:"signature"`
	Payload   WebhookPayloadPolicy   `json:"payload"`
	Rate      WebhookRatePolicy      `json:"rate"`
	Authority artifact.ID            `json:"authority"`
	Workflow  artifact.ID            `json:"workflow"`
}

// AutomationTriggerPolicy is immutable trigger authority. Schedule retains
// the user-authored expression; the scheduler owns parsing and due-time logic.
type AutomationTriggerPolicy struct {
	Version        uint16                    `json:"version"`
	Kind           AutomationTriggerKind     `json:"kind"`
	Schedule       string                    `json:"schedule,omitempty"`
	AnchorUnixNano int64                     `json:"anchor_unix_nano,omitempty"`
	Missed         AutomationMissedRunPolicy `json:"missed,omitempty"`
	Webhook        *WebhookTriggerPolicy     `json:"webhook,omitempty"`
	ID             artifact.ID               `json:"-"`
}

// AutomationMissedRunPolicy selects which overdue schedule slot may run.
type AutomationMissedRunPolicy string

const (
	// AutomationMissedLatest skips older overdue slots and claims the latest.
	AutomationMissedLatest AutomationMissedRunPolicy = "latest"
	// AutomationMissedCatchUpOne claims the oldest outstanding slot per scan.
	AutomationMissedCatchUpOne AutomationMissedRunPolicy = "catch-up-one"
)

// AutomationDeliveryKind selects a repository artifact or approved tool path.
type AutomationDeliveryKind string

const (
	// AutomationDeliveryArtifact retains results inside repository authority.
	AutomationDeliveryArtifact AutomationDeliveryKind = "artifact"
	// AutomationDeliveryTool invokes an approved registered tool.
	AutomationDeliveryTool AutomationDeliveryKind = "tool"
)

// AutomationDeliveryPolicy is immutable effect authority. Tool delivery binds
// an exact manual plus approval evidence; artifact delivery has no external effect.
type AutomationDeliveryPolicy struct {
	Version             uint16                 `json:"version"`
	Kind                AutomationDeliveryKind `json:"kind"`
	Tool                artifact.ID            `json:"tool,omitzero"`
	Authorization       artifact.ID            `json:"authorization,omitzero"`
	Destinations        []string               `json:"destinations,omitempty"`
	DestinationArgument string                 `json:"destination_argument,omitempty"`
	PayloadArgument     string                 `json:"payload_argument,omitempty"`
	IdempotencyArgument string                 `json:"idempotency_argument,omitempty"`
	ID                  artifact.ID            `json:"-"`
}

var automationTriggerPolicyCodec = artifact.JSONDocumentCodec(
	"automation trigger policy", artifact.KindProfile,
	AutomationTriggerPolicyMediaType, AutomationTriggerPolicySchema,
	func(value *AutomationTriggerPolicy) error {
		if value == nil || value.Version != AutomationTriggerPolicyVersion {
			return errors.New("recipe: invalid automation trigger policy")
		}
		value.Schedule = strings.TrimSpace(value.Schedule)
		switch value.Kind {
		case AutomationTriggerManual:
			if value.Schedule != "" || value.AnchorUnixNano != 0 || value.Missed != "" || value.Webhook != nil {
				return errors.New("recipe: manual trigger carries a schedule")
			}
		case AutomationTriggerSchedule:
			if value.Schedule == "" || strings.ContainsAny(value.Schedule, "\x00\r\n") || value.Webhook != nil {
				return errors.New("recipe: scheduled trigger requires one bounded expression")
			}
			period, err := time.ParseDuration(value.Schedule)
			if err != nil || period <= 0 || value.AnchorUnixNano < 0 ||
				value.Missed != AutomationMissedLatest && value.Missed != AutomationMissedCatchUpOne {
				return errors.New("recipe: scheduled trigger has invalid cadence authority")
			}
		case AutomationTriggerWebhook:
			if value.Schedule != "" || value.AnchorUnixNano != 0 || value.Missed != "" || value.Webhook == nil ||
				!validWebhookTrigger(*value.Webhook) {
				return errors.New("recipe: webhook trigger authority is incomplete")
			}
		default:
			return errors.New("recipe: unsupported automation trigger")
		}
		return nil
	},
	func(value AutomationTriggerPolicy) artifact.ID { return value.ID },
	func(value *AutomationTriggerPolicy, id artifact.ID) { value.ID = id },
	func(value AutomationTriggerPolicy) AutomationTriggerPolicy {
		if value.Webhook != nil {
			webhook := *value.Webhook
			value.Webhook = &webhook
		}
		return value
	},
)

var automationDeliveryPolicyCodec = artifact.JSONDocumentCodec(
	"automation delivery policy", artifact.KindProfile,
	AutomationDeliveryPolicyMediaType, AutomationDeliveryPolicySchema,
	func(value *AutomationDeliveryPolicy) error {
		if value == nil || value.Version != AutomationDeliveryPolicyVersion {
			return errors.New("recipe: invalid automation delivery policy")
		}
		switch value.Kind {
		case AutomationDeliveryArtifact:
			if value.Tool.Valid() || value.Authorization.Valid() || len(value.Destinations) != 0 ||
				value.DestinationArgument != "" || value.PayloadArgument != "" || value.IdempotencyArgument != "" {
				return errors.New("recipe: artifact delivery carries external effect authority")
			}
		case AutomationDeliveryTool:
			if value.Tool.Kind() != artifact.KindRecipe || value.Authorization.Kind() != artifact.KindEvidence {
				return errors.New("recipe: tool delivery lacks manual or authorization authority")
			}
			if len(value.Destinations) == 0 || value.DestinationArgument == "" ||
				value.PayloadArgument == "" || value.IdempotencyArgument == "" ||
				value.DestinationArgument == value.PayloadArgument ||
				value.DestinationArgument == value.IdempotencyArgument ||
				value.PayloadArgument == value.IdempotencyArgument {
				return errors.New("recipe: tool delivery argument authority is incomplete")
			}
			for _, destination := range value.Destinations {
				if strings.TrimSpace(destination) != destination || destination == "" || strings.ContainsAny(destination, "\x00\r\n") {
					return errors.New("recipe: invalid automation delivery destination")
				}
			}
			sort.Strings(value.Destinations)
			if compact := slices.Compact(value.Destinations); len(compact) != len(value.Destinations) {
				return errors.New("recipe: duplicate automation delivery destination")
			}
		default:
			return errors.New("recipe: unsupported automation delivery")
		}
		return nil
	},
	func(value AutomationDeliveryPolicy) artifact.ID { return value.ID },
	func(value *AutomationDeliveryPolicy, id artifact.ID) { value.ID = id },
	func(value AutomationDeliveryPolicy) AutomationDeliveryPolicy {
		value.Destinations = slices.Clone(value.Destinations)
		return value
	},
)

// Identify canonicalizes and identifies a trigger policy.
func (value AutomationTriggerPolicy) Identify() (AutomationTriggerPolicy, error) {
	value.Version, value.ID = AutomationTriggerPolicyVersion, artifact.ID{}
	return automationTriggerPolicyCodec.New(value)
}

// ArtifactContent returns canonical trigger policy content.
func (value AutomationTriggerPolicy) ArtifactContent() (artifact.Content, error) {
	return automationTriggerPolicyCodec.Content(value)
}

// ValidateIdentity verifies trigger policy identity.
func (value AutomationTriggerPolicy) ValidateIdentity() error {
	return automationTriggerPolicyCodec.ValidateIdentity(value)
}

// Lineage binds webhook policy to workflow, signature key, payload schema, and approval authority.
func (value AutomationTriggerPolicy) Lineage() []artifact.Lineage {
	if value.Webhook == nil || !value.ID.Valid() {
		return nil
	}
	return artifact.DependencyLineage(value.ID, value.Webhook.Workflow, value.Webhook.Signature.Key, value.Webhook.Payload.Schema, value.Webhook.Authority)
}

func validWebhookTrigger(value WebhookTriggerPolicy) bool {
	validText := func(text string) bool {
		return text != "" && strings.TrimSpace(text) == text && !strings.ContainsAny(text, "\x00\r\n\t")
	}
	return value.Signature.Algorithm == "hmac-sha256" && validText(value.Signature.Header) &&
		value.Signature.Key.Kind() == artifact.KindProfile && validText(value.Payload.ContentType) &&
		value.Payload.Schema.Kind() == artifact.KindProfile && value.Payload.MaxBytes > 0 &&
		validText(value.Payload.IdempotencyHeader) && value.Payload.IdempotencyHeader != value.Signature.Header &&
		value.Rate.WindowSeconds > 0 && value.Rate.MaxDeliveries > 0 &&
		value.Authority.Kind() == artifact.KindEvidence && value.Workflow.Kind() == artifact.KindRecipe
}

// Due returns the deterministic schedule slot due at now. Previous is the
// last durable claim; nil means this schedule has never claimed a slot.
func (value AutomationTriggerPolicy) Due(now time.Time, previous *time.Time) (time.Time, bool, error) {
	if value.ValidateIdentity() != nil || value.Kind != AutomationTriggerSchedule {
		return time.Time{}, false, errors.New("recipe: schedule policy is invalid")
	}
	period, err := time.ParseDuration(value.Schedule)
	if err != nil {
		return time.Time{}, false, err
	}
	anchor := time.UnixMicro(value.AnchorUnixNano / int64(time.Microsecond)).
		Add(time.Duration(value.AnchorUnixNano % int64(time.Microsecond))).UTC()
	now = now.UTC()
	if now.Before(anchor) {
		return time.Time{}, false, nil
	}
	latest := anchor.Add(time.Duration(now.Sub(anchor)/period) * period)
	if previous == nil {
		return latest, true, nil
	}
	prior := previous.UTC()
	next := prior.Add(period)
	if next.After(now) {
		return time.Time{}, false, nil
	}
	if value.Missed == AutomationMissedCatchUpOne {
		return next, true, nil
	}
	return latest, latest.After(prior), nil
}

// Identify canonicalizes and identifies a delivery policy.
func (value AutomationDeliveryPolicy) Identify() (AutomationDeliveryPolicy, error) {
	value.Version, value.ID = AutomationDeliveryPolicyVersion, artifact.ID{}
	return automationDeliveryPolicyCodec.New(value)
}

// ArtifactContent returns canonical delivery policy content.
func (value AutomationDeliveryPolicy) ArtifactContent() (artifact.Content, error) {
	return automationDeliveryPolicyCodec.Content(value)
}

// ValidateIdentity verifies delivery policy identity.
func (value AutomationDeliveryPolicy) ValidateIdentity() error {
	return automationDeliveryPolicyCodec.ValidateIdentity(value)
}

// RequireAutomationTriggerPolicy loads exact trigger authority.
func RequireAutomationTriggerPolicy(ctx context.Context, reader artifact.Reader, id artifact.ID) (AutomationTriggerPolicy, error) {
	return automationTriggerPolicyCodec.Require(ctx, reader, id)
}

// RequireAutomationDeliveryPolicy loads exact delivery authority.
func RequireAutomationDeliveryPolicy(ctx context.Context, reader artifact.Reader, id artifact.ID) (AutomationDeliveryPolicy, error) {
	return automationDeliveryPolicyCodec.Require(ctx, reader, id)
}
