package routedlm

import (
	"context"
	_ "embed"
	"errors"
	"path/filepath"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/strictjson"
)

const (
	flowProfileMediaType = "application/vnd.overgo.flow-profile+json"
	flowProfileSchema    = "overgo/flow-profile/v1"
)

//go:embed flow_profiles.json
var flowProfileCatalogJSON []byte

type flowProfileCatalogEntry struct {
	ModelTypes    []string    `json:"model_types"`
	Architectures []string    `json:"architectures"`
	Profile       FlowProfile `json:"profile"`
}

var flowProfileCodec = artifact.JSONDocumentCodec(
	"flow profile", artifact.KindProfile, flowProfileMediaType, flowProfileSchema,
	func(profile *FlowProfile) error { return profile.validate() },
	func(profile FlowProfile) artifact.ID { return profile.ID },
	func(profile *FlowProfile, id artifact.ID) { profile.ID = id }, nil,
)

// Content returns validated publication bytes.
func (profile FlowProfile) Content() (artifact.Content, error) {
	return flowProfileCodec.Content(profile)
}

// InspectFlowProfile resolves publication input from model metadata.
func InspectFlowProfile(modelDir string) (FlowProfile, error) {
	var metadata struct {
		Architectures []string `json:"architectures"`
		ModelType     string   `json:"model_type"`
	}
	if err := jsonfile.Decode(filepath.Join(modelDir, "config.json"), &metadata); err != nil {
		return FlowProfile{}, err
	}
	var catalog []flowProfileCatalogEntry
	if err := strictjson.DecodeBytes(flowProfileCatalogJSON, &catalog); err != nil {
		return FlowProfile{}, err
	}
	for _, entry := range catalog {
		if !slices.Contains(entry.ModelTypes, metadata.ModelType) || !slices.ContainsFunc(
			metadata.Architectures,
			func(architecture string) bool { return slices.Contains(entry.Architectures, architecture) },
		) {
			continue
		}
		return flowProfileCodec.New(entry.Profile)
	}
	return FlowProfile{}, errors.New("routed lm: no declared flow profile")
}

// ResolveFlowProfile loads the active recipe dependency.
func ResolveFlowProfile(
	ctx context.Context,
	reader artifact.Reader,
	definition recipe.Definition,
) (FlowProfile, error) {
	return modelrecipe.ResolveProfileDependency(
		ctx, reader, definition, recipe.DependencyFlowProfile, flowProfileCodec,
	)
}
