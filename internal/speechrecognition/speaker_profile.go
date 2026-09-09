package speechrecognition

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/checked"
)

// SpeakerProfile binds the acoustic operations and probability-to-sample
// policy to exact model, inventory and license artifacts. Labels are local to
// one recording; they do not identify people or persist across recordings.
type SpeakerProfile struct {
	Version       uint16                  `json:"version"`
	Model         artifact.ID             `json:"model"`
	Inventory     artifact.ID             `json:"inventory"`
	License       artifact.ID             `json:"license"`
	Frontend      audiodsp.FrontendConfig `json:"frontend"`
	MaximumOffset float32                 `json:"maximum_offset"`
	Encoder       Declaration             `json:"encoder"`
	Activity      ActivityBinding         `json:"activity"`
	Boundary      SpeakerBoundary         `json:"boundary"`
	ID            artifact.ID             `json:"-"`
}

// SpeakerBoundary uses a declared sample grid and strict threshold crossings.
// Equal scores retain state. An open final span ends at the last grid tick,
// not at an invented next frame. Padding is followed by minimum-length
// filtering and source-extent clipping. Adjacent padded spans are merged.
type SpeakerBoundary struct {
	Threshold      float32 `json:"threshold"`
	FrameSamples   uint64  `json:"frame_samples"`
	TickSamples    uint64  `json:"tick_samples"`
	PadSamples     uint64  `json:"pad_samples"`
	MinimumSamples uint64  `json:"minimum_samples"`
	FinalBoundary  string  `json:"final_boundary"`
}

func (b SpeakerBoundary) validate() error {
	if !checked.UnitInterval64(float64(b.Threshold)) || b.FrameSamples == 0 || b.TickSamples == 0 || b.FrameSamples%b.TickSamples != 0 ||
		b.PadSamples%b.TickSamples != 0 || b.MinimumSamples%b.TickSamples != 0 || b.FinalBoundary != "last-tick-clipped-to-source" {
		return errors.New("speaker boundary: invalid sample grid or threshold policy")
	}
	return nil
}

var speakerProfileCodec = artifact.JSONDocumentCodec("speaker activity profile", artifact.KindProfile,
	"application/vnd.overgo.speaker-activity-profile+json", "overgo/speaker-activity-profile/v1",
	func(p *SpeakerProfile) error {
		if p == nil || p.Version != artifact.InitialDocumentVersion || p.Model.Kind() != artifact.KindModel || p.Inventory.Kind() != artifact.KindTensorInventory || p.License.Kind() != artifact.KindFile ||
			!checked.PositiveFinite64(float64(p.MaximumOffset)) || p.Activity.Bands != int(p.Frontend.Geometry.FeatureBins) || p.Activity.MaxFrames <= 0 || p.Encoder.FeedbackAfter != 0 || len(p.Encoder.Blocks) == 0 || len(p.Activity.Blocks) == 0 {
			return errors.New("speaker activity: incomplete profile")
		}
		step := p.Frontend.Geometry.HopSamples
		for _, stage := range p.Activity.Subsampling {
			var ok bool
			step, ok = checked.Mul64(step, uint64(stage.Stride[1]))
			if !ok || step == 0 {
				return errors.New("speaker activity: sample step overflows")
			}
		}
		for _, block := range p.Encoder.Blocks {
			var ok bool
			step, ok = checked.Mul64(step, uint64(block.Convolution.Stride))
			if !ok || block.Convolution.Stride <= 0 {
				return errors.New("speaker activity: encoder step overflows")
			}
		}
		if step != p.Boundary.FrameSamples {
			return errors.New("speaker activity: output step differs from declared operations")
		}
		return p.Boundary.validate()
	}, func(p SpeakerProfile) artifact.ID { return p.ID }, func(p *SpeakerProfile, id artifact.ID) { p.ID = id }, func(p SpeakerProfile) SpeakerProfile {
		p.Frontend = p.Frontend.Clone()
		// Reuse the recognition declaration's deep-copy owner.
		p.Encoder = cloneExecutionProfile(ExecutionProfile{Encoder: p.Encoder}).Encoder
		p.Activity.Subsampling = slices.Clone(p.Activity.Subsampling)
		p.Activity.Blocks = slices.Clone(p.Activity.Blocks)
		return p
	})

// NewSpeakerProfile identifies one immutable speaker execution declaration.
func NewSpeakerProfile(profile SpeakerProfile) (SpeakerProfile, error) {
	return speakerProfileCodec.NewInitial(profile)
}

// RequireSpeakerProfile loads one exact declaration and its dependency lineage.
func RequireSpeakerProfile(ctx context.Context, reader artifact.Reader, id artifact.ID) (SpeakerProfile, error) {
	return speakerProfileCodec.RequireExactLineage(ctx, reader, id, SpeakerProfile.Lineage)
}

// Lineage binds this profile to the exact model, inventory and license evidence.
func (p SpeakerProfile) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(p.ID, p.Model, p.Inventory, p.License)
}

// Batch publishes this profile and its immutable dependencies atomically.
func (p SpeakerProfile) Batch(key string) (artifact.Batch, error) {
	return speakerProfileCodec.Batch(key, p, p.Lineage(), nil)
}
