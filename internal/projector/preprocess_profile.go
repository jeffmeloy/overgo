package projector

import (
	"context"
	_ "embed"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/strictjson"
)

const (
	mediaPreprocessProfileMediaType = "application/vnd.overgo.media-preprocess-profile+json"
	mediaPreprocessProfileSchema    = "overgo/media-preprocess-profile/v1"
)

//go:embed preprocess_profiles.json
var mediaPreprocessCatalogJSON []byte

// MediaPreprocessProfile defines recipe-bound raster and audio policy.
type MediaPreprocessProfile struct {
	ID                         artifact.ID      `json:"-"`
	Version                    uint16           `json:"version"`
	Image                      MediaPixelBudget `json:"image"`
	Video                      MediaPixelBudget `json:"video"`
	AudioAttentionRopeFreqBase float32          `json:"audio_attention_rope_freq_base,omitzero"`
	ModelConfig                artifact.ID      `json:"model_config,omitzero"`
	ImageAttention             string           `json:"image_attention,omitzero"`
}

type mediaPreprocessCatalogEntry struct {
	Projectors []string               `json:"projectors"`
	Profile    MediaPreprocessProfile `json:"profile"`
}

var mediaPreprocessProfileCodec = artifact.JSONDocumentCodec(
	"media preprocess profile", artifact.KindProfile,
	mediaPreprocessProfileMediaType, mediaPreprocessProfileSchema,
	func(profile *MediaPreprocessProfile) error {
		if profile.Version != artifact.InitialDocumentVersion {
			return errors.New("projector: invalid media preprocess profile version")
		}
		if profile.ImageAttention != "" || profile.ModelConfig != (artifact.ID{}) {
			if profile.ModelConfig.Kind() != artifact.KindProfile ||
				(profile.ImageAttention != "causal" && profile.ImageAttention != "vision") {
				return errors.New("projector: image attention requires a bound model declaration")
			}
			if profile.Image == (MediaPixelBudget{}) && profile.Video == (MediaPixelBudget{}) && profile.AudioAttentionRopeFreqBase == 0 {
				return nil // Encoder geometry is declared in its artifact.
			}
		}
		if profile.AudioAttentionRopeFreqBase != 0 {
			if !checked.PositiveFinite32(profile.AudioAttentionRopeFreqBase) {
				return errors.New("projector: invalid audio attention frequency base")
			}
			if profile.Image == (MediaPixelBudget{}) && profile.Video == (MediaPixelBudget{}) {
				return nil // Raster geometry is already declared by the tower artifact.
			}
		}
		if err := profile.Image.validate(); err != nil {
			return err
		}
		return profile.Video.validate()
	},
	func(profile MediaPreprocessProfile) artifact.ID { return profile.ID },
	func(profile *MediaPreprocessProfile, id artifact.ID) { profile.ID = id }, nil,
)

// Content returns validated publication bytes.
func (profile MediaPreprocessProfile) Content() (artifact.Content, error) {
	return mediaPreprocessProfileCodec.Content(profile)
}

// BindModelConfig seals an explicit image-attention declaration into the
// processor profile. The projection recipe then binds this exact config ID.
func (profile MediaPreprocessProfile) BindModelConfig(config modelartifact.ModelConfigDocument) (MediaPreprocessProfile, error) {
	if _, err := config.Content(); err != nil {
		return MediaPreprocessProfile{}, err
	}
	if config.Generation == nil || config.Generation.ImageAttention == "" {
		return MediaPreprocessProfile{}, errors.New("projector: model config omits image attention")
	}
	profile.ID = artifact.ID{}
	profile.ModelConfig = config.ID
	profile.ImageAttention = config.Generation.ImageAttention
	return mediaPreprocessProfileCodec.New(profile)
}

// ResolvePreprocessProfile validates the exact recipe-bound processor against
// encoder geometry and its pinned text-model config, never a latest config.
func ResolvePreprocessProfile(ctx context.Context, store artifact.Reader, definition recipe.Definition, declared *MediaPreprocessProfile) (*MediaPreprocessProfile, error) {
	_, bound := definition.PrimaryDependency(recipe.DependencyProcessorProfile)
	if !bound {
		if declared != nil {
			return nil, errors.New("projector: required processor profile is absent")
		}
		return nil, nil
	}
	profile, err := modelrecipe.ResolveProfileDependency(ctx, store, definition, recipe.DependencyProcessorProfile, mediaPreprocessProfileCodec)
	if err != nil {
		return nil, err
	}
	expected := MediaPreprocessProfile{Version: artifact.InitialDocumentVersion}
	if declared != nil {
		expected = *declared
	}
	if profile.ModelConfig.Valid() {
		content, found, err := artifact.ReadContent(ctx, store, profile.ModelConfig)
		if err != nil {
			return nil, err
		}
		if !found {
			return nil, errors.New("projector: bound model config is missing")
		}
		config, err := modelartifact.ParseModelConfigDocument(content.Data)
		if err != nil {
			return nil, err
		}
		modelID, ok := definition.PrimaryDependency(recipe.DependencyModel)
		if !ok || config.Model != modelID || config.ID != profile.ModelConfig {
			return nil, errors.New("projector: image attention config differs from recipe model or identity")
		}
		expected, err = expected.BindModelConfig(config)
		if err != nil {
			return nil, err
		}
	}
	if profile.ID != expected.ID {
		return nil, errors.New("projector: processor profile differs from artifact and config declarations")
	}
	return &profile, nil
}

func catalogMediaPreprocessProfile(projectorType string) (MediaPreprocessProfile, bool, error) {
	var catalog []mediaPreprocessCatalogEntry
	if err := strictjson.DecodeBytes(mediaPreprocessCatalogJSON, &catalog); err != nil {
		return MediaPreprocessProfile{}, false, err
	}
	for _, entry := range catalog {
		if slices.Contains(entry.Projectors, projectorType) {
			profile, err := mediaPreprocessProfileCodec.New(entry.Profile)
			return profile, err == nil, err
		}
	}
	return MediaPreprocessProfile{}, false, nil
}
