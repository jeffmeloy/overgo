package modelrecipe

import (
	"context"
	"errors"
	"math"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

// ComponentSession: one ordered recipe-stage lifetime.
type ComponentSession struct {
	Identity      artifact.ID
	Node          recipe.NodeID
	Module        recipe.ModuleID
	Model         artifact.ID
	Session       recipe.SessionPolicy
	Placement     recipe.Placement
	Residency     recipe.ResidencyPolicy
	ArtifactBytes uint64
}

// ComponentSessionPlan: compiled stage lifetimes plus unique artifact extent.
type ComponentSessionPlan struct {
	Identity      artifact.ID
	Recipe        artifact.ID
	ArtifactBytes uint64
	Components    []ComponentSession
}

type componentSessionIdentity struct {
	Node          recipe.NodeID          `json:"node"`
	Module        recipe.ModuleID        `json:"module"`
	Model         artifact.ID            `json:"model"`
	Session       recipe.SessionPolicy   `json:"session"`
	Placement     recipe.Placement       `json:"placement"`
	Residency     recipe.ResidencyPolicy `json:"residency,omitempty"`
	ArtifactBytes uint64                 `json:"artifact_bytes"`
}

type componentSessionPlanIdentity struct {
	Recipe     artifact.ID   `json:"recipe"`
	Components []artifact.ID `json:"components"`
}

// CompileComponentSessionPlan: ordered catalog-derived component lifetimes.
func CompileComponentSessionPlan(ctx context.Context, reader artifact.Reader, program recipe.Program) (ComponentSessionPlan, error) {
	if ctx == nil || reader == nil {
		return ComponentSessionPlan{}, errors.New("model recipe: component session authority is absent")
	}
	definition := program.Definition()
	if definition.ID.Kind() != artifact.KindRecipe || definition.Model.Kind() != artifact.KindModel {
		return ComponentSessionPlan{}, errors.New("model recipe: invalid component session identity")
	}
	stages := program.Stages()
	extents := make(map[artifact.ID]uint64)
	components := make([]ComponentSession, 0, len(stages))
	var total uint64
	for _, stage := range stages {
		node := stage.Node
		if node.Session == "" {
			continue
		}
		modelID, ok := definition.Dependency(recipe.DependencyModel, node.ModelSlot)
		if !ok {
			return ComponentSessionPlan{}, errors.New("model recipe: component model binding is absent")
		}
		bytes, known := extents[modelID]
		if !known {
			var err error
			bytes, err = manifestArtifactBytes(ctx, reader, modelID, map[artifact.ID]struct{}{})
			if err != nil || bytes == 0 || total > math.MaxUint64-bytes {
				return ComponentSessionPlan{}, errors.Join(errors.New("model recipe: component artifact extent is unavailable"), err)
			}
			extents[modelID], total = bytes, total+bytes
		}
		body := componentSessionIdentity{
			Node: node.ID, Module: node.Module, Model: modelID, Session: node.Session,
			Placement: node.Placement, Residency: node.Residency, ArtifactBytes: bytes,
		}
		id, err := artifact.JSONID(artifact.KindProfile, body)
		if err != nil {
			return ComponentSessionPlan{}, err
		}
		components = append(components, ComponentSession{
			Identity: id, Node: body.Node, Module: body.Module, Model: body.Model,
			Session: body.Session, Placement: body.Placement, Residency: body.Residency,
			ArtifactBytes: body.ArtifactBytes,
		})
	}
	if len(components) == 0 {
		return ComponentSessionPlan{}, errors.New("model recipe: component session policy is absent")
	}
	componentIDs := make([]artifact.ID, len(components))
	for index := range components {
		componentIDs[index] = components[index].Identity
	}
	id, err := artifact.JSONID(artifact.KindProfile, componentSessionPlanIdentity{
		Recipe: definition.ID, Components: componentIDs,
	})
	if err != nil {
		return ComponentSessionPlan{}, err
	}
	return ComponentSessionPlan{
		Identity: id, Recipe: definition.ID, ArtifactBytes: total, Components: components,
	}, nil
}

func (plan ComponentSessionPlan) RequestScoped() bool {
	for _, component := range plan.Components {
		if component.Session == recipe.SessionRequest {
			return true
		}
	}
	return false
}

func manifestArtifactBytes(ctx context.Context, reader artifact.Reader, id artifact.ID, seen map[artifact.ID]struct{}) (uint64, error) {
	if _, duplicate := seen[id]; duplicate {
		return 0, nil
	}
	seen[id] = struct{}{}
	manifest, found, err := reader.Manifest(ctx, id)
	if err != nil {
		return 0, err
	}
	if !found {
		descriptor, found, err := reader.Artifact(ctx, id)
		if err != nil || !found || descriptor.Size == 0 {
			return 0, errors.Join(errors.New("model recipe: session component is absent"), err)
		}
		return descriptor.Size, nil
	}
	var total uint64
	for _, component := range manifest.Components {
		bytes, err := manifestArtifactBytes(ctx, reader, component.Artifact, seen)
		if err != nil || total > math.MaxUint64-bytes {
			return 0, errors.Join(errors.New("model recipe: session artifact extent overflow"), err)
		}
		total += bytes
	}
	return total, nil
}
