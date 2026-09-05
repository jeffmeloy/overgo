package dataset

import (
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/recipecontract"
)

const (
	// AudioSignalProfileMediaType identifies decoded-signal measurement evidence.
	AudioSignalProfileMediaType = "application/vnd.overgo.audio-signal-profile+json"
	// AudioSignalProfileSchema identifies the first signal-profile schema.
	AudioSignalProfileSchema = "overgo/audio-signal-profile/v1"
	// AudioAdmissionPolicyMediaType identifies reusable signal-admission policy.
	AudioAdmissionPolicyMediaType = "application/vnd.overgo.audio-admission-policy+json"
	// AudioAdmissionPolicySchema identifies the first admission-policy schema.
	AudioAdmissionPolicySchema = "overgo/audio-admission-policy/v1"
	// AudioAdmissionDecisionMediaType identifies durable admission evidence.
	AudioAdmissionDecisionMediaType = "application/vnd.overgo.audio-admission-decision+json"
	// AudioAdmissionDecisionSchema identifies the first admission-decision schema.
	AudioAdmissionDecisionSchema = "overgo/audio-admission-decision/v1"
)

// AudioSignalProfileDocument publishes one source-bound signal measurement.
type AudioSignalProfileDocument struct {
	Version uint16                                   `json:"version"`
	Profile recipecontract.DecodedAudioSignalProfile `json:"profile"`
	ID      artifact.ID                              `json:"-"`
}

// AudioAdmissionPolicyDocument publishes reusable artifact-owned thresholds.
type AudioAdmissionPolicyDocument struct {
	Version uint16                              `json:"version"`
	Policy  recipecontract.AudioAdmissionPolicy `json:"policy"`
	ID      artifact.ID                         `json:"-"`
}

// AudioAdmissionDecisionDocument binds a deterministic decision to exact
// signal-profile and policy artifacts.
type AudioAdmissionDecisionDocument struct {
	Version  uint16                                `json:"version"`
	Signal   artifact.ID                           `json:"signal"`
	Policy   artifact.ID                           `json:"policy"`
	Decision recipecontract.AudioAdmissionDecision `json:"decision"`
	ID       artifact.ID                           `json:"-"`
}

var audioSignalProfileCodec = artifact.JSONDocumentCodec(
	"audio signal profile", artifact.KindEvidence, AudioSignalProfileMediaType, AudioSignalProfileSchema,
	canonicalizeAudioSignalProfile, func(value AudioSignalProfileDocument) artifact.ID { return value.ID },
	func(value *AudioSignalProfileDocument, id artifact.ID) { value.ID = id },
	func(value AudioSignalProfileDocument) AudioSignalProfileDocument { return value },
)

var audioAdmissionPolicyCodec = artifact.JSONDocumentCodec(
	"audio admission policy", artifact.KindProfile, AudioAdmissionPolicyMediaType, AudioAdmissionPolicySchema,
	canonicalizeAudioAdmissionPolicy, func(value AudioAdmissionPolicyDocument) artifact.ID { return value.ID },
	func(value *AudioAdmissionPolicyDocument, id artifact.ID) { value.ID = id },
	func(value AudioAdmissionPolicyDocument) AudioAdmissionPolicyDocument { return value },
)

var audioAdmissionDecisionCodec = artifact.JSONDocumentCodec(
	"audio admission decision", artifact.KindEvidence, AudioAdmissionDecisionMediaType, AudioAdmissionDecisionSchema,
	canonicalizeAudioAdmissionDecision, func(value AudioAdmissionDecisionDocument) artifact.ID { return value.ID },
	func(value *AudioAdmissionDecisionDocument, id artifact.ID) { value.ID = id }, cloneAudioAdmissionDecision,
)

func canonicalizeAudioSignalProfile(document *AudioSignalProfileDocument) error {
	if document == nil || document.Version != artifact.InitialDocumentVersion {
		return errors.New("dataset: invalid audio signal profile envelope")
	}
	return document.Profile.Validate()
}

func canonicalizeAudioAdmissionPolicy(document *AudioAdmissionPolicyDocument) error {
	if document == nil || document.Version != artifact.InitialDocumentVersion {
		return errors.New("dataset: invalid audio admission policy envelope")
	}
	return document.Policy.Validate()
}

func canonicalizeAudioAdmissionDecision(document *AudioAdmissionDecisionDocument) error {
	if document == nil || document.Version != artifact.InitialDocumentVersion ||
		document.Signal.Kind() != artifact.KindEvidence || document.Policy.Kind() != artifact.KindProfile {
		return errors.New("dataset: invalid audio admission decision envelope")
	}
	slices.Sort(document.Decision.Violations)
	document.Decision.Violations = slices.Compact(document.Decision.Violations)
	return document.Decision.Validate()
}

func cloneAudioAdmissionDecision(document AudioAdmissionDecisionDocument) AudioAdmissionDecisionDocument {
	document.Decision.Violations = slices.Clone(document.Decision.Violations)
	return document
}
