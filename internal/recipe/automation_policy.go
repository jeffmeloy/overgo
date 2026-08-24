package recipe

import (
	"context"
	"errors"
	"strings"

	"overgo/internal/artifact"
)

const (
	// AutomationTriggerPolicyVersion is the trigger document revision.
	AutomationTriggerPolicyVersion = artifact.InitialDocumentVersion
	// AutomationTriggerPolicyMediaType identifies trigger policy documents.
	AutomationTriggerPolicyMediaType = "application/vnd.overgo.automation-trigger-policy+json"
	// AutomationTriggerPolicySchema identifies the trigger policy contract.
	AutomationTriggerPolicySchema = "overgo/automation-trigger-policy/v1"
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
)

// AutomationTriggerPolicy is immutable trigger authority. Schedule retains
// the user-authored expression; the scheduler owns parsing and due-time logic.
type AutomationTriggerPolicy struct {
	Version  uint16                `json:"version"`
	Kind     AutomationTriggerKind `json:"kind"`
	Schedule string                `json:"schedule,omitempty"`
	ID       artifact.ID           `json:"-"`
}

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
	Version       uint16                 `json:"version"`
	Kind          AutomationDeliveryKind `json:"kind"`
	Tool          artifact.ID            `json:"tool,omitzero"`
	Authorization artifact.ID            `json:"authorization,omitzero"`
	ID            artifact.ID            `json:"-"`
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
			if value.Schedule != "" {
				return errors.New("recipe: manual trigger carries a schedule")
			}
		case AutomationTriggerSchedule:
			if value.Schedule == "" || strings.ContainsAny(value.Schedule, "\x00\r\n") {
				return errors.New("recipe: scheduled trigger requires one bounded expression")
			}
		default:
			return errors.New("recipe: unsupported automation trigger")
		}
		return nil
	},
	func(value AutomationTriggerPolicy) artifact.ID { return value.ID },
	func(value *AutomationTriggerPolicy, id artifact.ID) { value.ID = id },
	func(value AutomationTriggerPolicy) AutomationTriggerPolicy { return value },
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
			if value.Tool.Valid() || value.Authorization.Valid() {
				return errors.New("recipe: artifact delivery carries external effect authority")
			}
		case AutomationDeliveryTool:
			if value.Tool.Kind() != artifact.KindRecipe || value.Authorization.Kind() != artifact.KindEvidence {
				return errors.New("recipe: tool delivery lacks manual or authorization authority")
			}
		default:
			return errors.New("recipe: unsupported automation delivery")
		}
		return nil
	},
	func(value AutomationDeliveryPolicy) artifact.ID { return value.ID },
	func(value *AutomationDeliveryPolicy, id artifact.ID) { value.ID = id },
	func(value AutomationDeliveryPolicy) AutomationDeliveryPolicy { return value },
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
