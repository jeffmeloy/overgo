package speechrecognition

import (
	"context"
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
)

const (
	// ExecutionProfileMediaType identifies an operation-declared speech recognition profile.
	ExecutionProfileMediaType = "application/vnd.overgo.speech-recognition-profile+json"
	// ExecutionProfileSchema identifies the first speech recognition execution profile schema.
	ExecutionProfileSchema = "overgo/speech-recognition-profile/v1"
)

// ExecutionProfile carries operation-named frontend, grouping, encoder, and
// decoding declarations. Concrete tensor names remain artifact data rather
// than model-family branches in executable code.
type ExecutionProfile struct {
	Version    uint16                        `json:"version"`
	Frontend   audiodsp.FrontendConfig       `json:"frontend"`
	Grouping   audiodsp.GroupedFeatureConfig `json:"grouping"`
	Encoder    Declaration                   `json:"encoder"`
	BlankToken int                           `json:"blank_token"`
	Language   string                        `json:"language,omitzero"`
	ID         artifact.ID                   `json:"-"`
}

var executionProfileCodec = artifact.JSONDocumentCodec(
	"speech recognition execution profile", artifact.KindProfile,
	ExecutionProfileMediaType, ExecutionProfileSchema,
	canonicalizeExecutionProfile,
	func(value ExecutionProfile) artifact.ID { return value.ID },
	func(value *ExecutionProfile, id artifact.ID) { value.ID = id },
	cloneExecutionProfile,
)

// NewExecutionProfile identifies one immutable speech recognition execution declaration.
func NewExecutionProfile(frontend audiodsp.FrontendConfig, grouping audiodsp.GroupedFeatureConfig, encoder Declaration, blankToken int, language string) (ExecutionProfile, error) {
	return executionProfileCodec.NewInitial(ExecutionProfile{
		Frontend: frontend, Grouping: grouping, Encoder: encoder,
		BlankToken: blankToken, Language: language,
	})
}

// RequireExecutionProfile loads one exact speech recognition execution declaration.
func RequireExecutionProfile(ctx context.Context, reader artifact.Reader, id artifact.ID) (ExecutionProfile, error) {
	return executionProfileCodec.Require(ctx, reader, id)
}

// ValidateIdentity verifies the profile's content-derived identity.
func (profile ExecutionProfile) ValidateIdentity() error {
	return executionProfileCodec.ValidateIdentity(profile)
}

// Content returns the canonical profile artifact.
func (profile ExecutionProfile) Content() (artifact.Content, error) {
	return executionProfileCodec.Content(profile)
}

// Batch returns one atomic profile publication.
func (profile ExecutionProfile) Batch(key string) (artifact.Batch, error) {
	return executionProfileCodec.Batch(key, profile, nil, nil)
}

func canonicalizeExecutionProfile(profile *ExecutionProfile) error {
	if profile == nil || profile.Version != artifact.InitialDocumentVersion ||
		profile.Grouping.StackFrames <= 0 || profile.Grouping.DeltaRadius < 0 ||
		profile.Grouping.FinalFrameSamples < 0 || profile.BlankToken < 0 ||
		profile.Language != strings.TrimSpace(profile.Language) || strings.ContainsAny(profile.Language, "\r\n") {
		return errors.New("speech recognition: invalid execution profile")
	}
	return nil
}

func cloneExecutionProfile(profile ExecutionProfile) ExecutionProfile {
	profile.Frontend.ResampleTaps = slices.Clone(profile.Frontend.ResampleTaps)
	if profile.Frontend.Normalize != nil {
		normalize := *profile.Frontend.Normalize
		profile.Frontend.Normalize = &normalize
	}
	if profile.Frontend.Log.DynamicRange != nil {
		dynamicRange := *profile.Frontend.Log.DynamicRange
		profile.Frontend.Log.DynamicRange = &dynamicRange
	}
	profile.Encoder.Blocks = slices.Clone(profile.Encoder.Blocks)
	return profile
}
