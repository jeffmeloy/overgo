package projector

import (
	_ "embed"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

const (
	mediaPreprocessProfileMediaType = "application/vnd.overgo.media-preprocess-profile+json"
	mediaPreprocessProfileSchema    = "overgo/media-preprocess-profile/v1"
)

//go:embed preprocess_profiles.json
var mediaPreprocessCatalogJSON []byte

// MediaPreprocessProfile defines recipe-bound raster policy.
type MediaPreprocessProfile struct {
	ID      artifact.ID      `json:"-"`
	Version uint16           `json:"version"`
	Image   MediaPixelBudget `json:"image"`
	Video   MediaPixelBudget `json:"video"`
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
