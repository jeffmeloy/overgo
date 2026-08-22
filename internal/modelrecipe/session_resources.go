package modelrecipe

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/recipe"
)

// ComponentSession defines one ordered recipe-stage lifetime.
type ComponentSession struct {
	Identity      artifact.ID            `json:"identity,omitzero"`
	Node          recipe.NodeID          `json:"node"`
	Module        recipe.ModuleID        `json:"module"`
	Model         artifact.ID            `json:"model"`
	Session       recipe.SessionPolicy   `json:"session"`
	Placement     recipe.Placement       `json:"placement"`
	Residency     recipe.ResidencyPolicy `json:"residency,omitempty"`
	ArtifactBytes uint64                 `json:"artifact_bytes"`
}

// ComponentSessionPlan defines compiled stage lifetimes plus unique artifact extent.
type ComponentSessionPlan struct {
	Identity      artifact.ID
	Recipe        artifact.ID
	ArtifactBytes uint64
	Components    []ComponentSession
}

type componentSessionPlanDocument struct {
	Version       uint16             `json:"version"`
	Recipe        artifact.ID        `json:"recipe"`
	ArtifactBytes uint64             `json:"artifact_bytes"`
	Components    []ComponentSession `json:"components"`
}

const (
	componentSessionPlanVersion = 1
)

var componentSessionPlanContract = artifact.DocumentContract{
	Kind: artifact.KindProfile, MediaType: "application/vnd.overgo.component-session-plan+json",
	Schema: "overgo/component-session-plan/v1",
}

// CompileComponentSessionPlan compiles ordered catalog-derived component lifetimes.
func CompileComponentSessionPlan(ctx context.Context, reader artifact.Reader, program recipe.Program) (ComponentSessionPlan, error) {
	if ctx == nil || reader == nil {
		return ComponentSessionPlan{}, errors.New("model recipe: component session authority is absent")
	}
	return compileComponentSessionPlan(ctx, program, func(ctx context.Context, id artifact.ID) (uint64, error) {
		return manifestArtifactBytes(ctx, reader, id, map[artifact.ID]struct{}{})
	})
}

// CompileComponentSessionPlanWithExtents compiles ordered lifetimes from validated extents.
func CompileComponentSessionPlanWithExtents(
	ctx context.Context,
	program recipe.Program,
	extent func(context.Context, artifact.ID) (uint64, error),
) (ComponentSessionPlan, error) {
	if ctx == nil || extent == nil {
		return ComponentSessionPlan{}, errors.New("model recipe: component extent authority is absent")
	}
	return compileComponentSessionPlan(ctx, program, extent)
}

func compileComponentSessionPlan(
	ctx context.Context,
	program recipe.Program,
	extent func(context.Context, artifact.ID) (uint64, error),
) (ComponentSessionPlan, error) {
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
		if !node.Session.Valid() || !node.Residency.Valid() {
			return ComponentSessionPlan{}, errors.New("model recipe: component lifetime is invalid")
		}
		modelID, ok := definition.Dependency(recipe.DependencyModel, node.ModelSlot)
		if !ok {
			return ComponentSessionPlan{}, errors.New("model recipe: component model binding is absent")
		}
		bytes, known := extents[modelID]
		if !known {
			resolved, err := extent(ctx, modelID)
			bytes = resolved
			next, valid := checked.Add64(total, bytes)
			if err != nil || !checked.Nonzero(bytes) || !valid {
				return ComponentSessionPlan{}, errors.Join(errors.New("model recipe: component artifact extent is unavailable"), err)
			}
			extents[modelID], total = bytes, next
		}
		component := ComponentSession{
			Node: node.ID, Module: node.Module, Model: modelID, Session: node.Session,
			Placement: node.Placement, Residency: node.Residency, ArtifactBytes: bytes,
		}
		id, err := artifact.JSONID(artifact.KindProfile, component)
		if err != nil {
			return ComponentSessionPlan{}, err
		}
		component.Identity = id
		components = append(components, component)
	}
	if len(components) == 0 {
		return ComponentSessionPlan{}, errors.New("model recipe: component session policy is absent")
	}
	body := componentSessionPlanDocument{
		Version: componentSessionPlanVersion, Recipe: definition.ID,
		ArtifactBytes: total, Components: components,
	}
	id, err := artifact.JSONID(artifact.KindProfile, body)
	if err != nil {
		return ComponentSessionPlan{}, err
	}
	return ComponentSessionPlan{
		Identity: id, Recipe: definition.ID, ArtifactBytes: total, Components: components,
	}, nil
}

func (plan ComponentSessionPlan) Content() (artifact.Content, error) {
	body := componentSessionPlanDocument{
		Version: componentSessionPlanVersion, Recipe: plan.Recipe,
		ArtifactBytes: plan.ArtifactBytes, Components: plan.Components,
	}
	content, err := artifact.JSONContent(componentSessionPlanContract, body)
	if err != nil {
		return artifact.Content{}, err
	}
	if content.Descriptor.ID != plan.Identity {
		return artifact.Content{}, errors.New("model recipe: component session plan identity differs")
	}
	return content, nil
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
		if err != nil || !found || !checked.Nonzero(descriptor.Size) {
			return 0, errors.Join(errors.New("model recipe: session component is absent"), err)
		}
		return descriptor.Size, nil
	}
	var total uint64
	for _, component := range manifest.Components {
		bytes, err := manifestArtifactBytes(ctx, reader, component.Artifact, seen)
		next, valid := checked.Add64(total, bytes)
		if err != nil || !valid {
			return 0, errors.Join(errors.New("model recipe: session artifact extent overflow"), err)
		}
		total = next
	}
	return total, nil
}
