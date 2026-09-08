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
	Transducer *TransducerBinding            `json:"transducer,omitzero"`
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
func NewExecutionProfile(profile ExecutionProfile) (ExecutionProfile, error) {
	return executionProfileCodec.NewInitial(profile)
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
	if binding := profile.Transducer; binding != nil {
		if profile.Grouping != (audiodsp.GroupedFeatureConfig{StackFrames: 1}) ||
			binding.Blank != profile.BlankToken || binding.Bands != int(profile.Frontend.Geometry.FeatureBins) {
			return errors.New("speech recognition: recurrent decoder requires ungrouped matching features and blank")
		}
	}
	return nil
}

func cloneExecutionProfile(profile ExecutionProfile) ExecutionProfile {
	profile.Frontend = profile.Frontend.Clone()
	if profile.Transducer != nil {
		binding := *profile.Transducer
		binding.Subsampling = slices.Clone(binding.Subsampling)
		binding.Recurrent = slices.Clone(binding.Recurrent)
		profile.Transducer = &binding
	}
	profile.Encoder.Blocks = slices.Clone(profile.Encoder.Blocks)
	for i := range profile.Encoder.Blocks {
		if binding := profile.Encoder.Blocks[i].Attention.ProjectedRelative; binding != nil {
			value := *binding
			profile.Encoder.Blocks[i].Attention.ProjectedRelative = &value
		}
	}
	return profile
}
