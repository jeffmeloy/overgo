package speechrecognition

import (
	"context"
	"errors"

	"overgo/internal/artifact"
)

// AlignmentProfile declares how a CTC path becomes word/sample intervals.
// Pooled hop cells start at the requested source span and use the product of
// frontend hop, feature grouping and encoder pooling strides. Token support
// excludes surrounding blanks; confidence is the geometric mean probability
// over target-emitting frames, not a calibrated word correctness estimate.
type AlignmentProfile struct {
	Version       uint16      `json:"version"`
	FrameMapping  string      `json:"frame_mapping"`
	WordMapping   string      `json:"word_mapping"`
	BlankBoundary string      `json:"blank_boundary"`
	Confidence    string      `json:"confidence"`
	ID            artifact.ID `json:"-"`
}

var alignmentProfileCodec = artifact.JSONDocumentCodec("CTC alignment profile", artifact.KindProfile,
	"application/vnd.overgo.ctc-alignment-profile+json", "overgo/ctc-alignment-profile/v1",
	func(profile *AlignmentProfile) error {
		if profile == nil || profile.Version != artifact.InitialDocumentVersion ||
			profile.FrameMapping != "pooled-hop-cells" || profile.WordMapping != "whitespace-token-spans" ||
			profile.BlankBoundary != "excluded" || profile.Confidence != "geometric-mean-target-probability" {
			return errors.New("CTC alignment: unsupported mapping declaration")
		}
		return nil
	}, func(profile AlignmentProfile) artifact.ID { return profile.ID },
	func(profile *AlignmentProfile, id artifact.ID) { profile.ID = id },
	func(profile AlignmentProfile) AlignmentProfile { return profile })

// NewAlignmentProfile identifies an explicit supported mapping declaration.
func NewAlignmentProfile(profile AlignmentProfile) (AlignmentProfile, error) {
	return alignmentProfileCodec.NewInitial(profile)
}

// RequireAlignmentProfile loads one exact supported mapping declaration.
func RequireAlignmentProfile(ctx context.Context, reader artifact.Reader, id artifact.ID) (AlignmentProfile, error) {
	return alignmentProfileCodec.Require(ctx, reader, id)
}

// Batch publishes the declaration through the shared document owner.
func (profile AlignmentProfile) Batch(key string) (artifact.Batch, error) {
	return alignmentProfileCodec.Batch(key, profile, nil, nil)
}
