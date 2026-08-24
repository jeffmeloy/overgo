package modelrecipe

import (
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

type RuntimeDescription struct {
	Identity      ProgramIdentity         `json:"identity"`
	Task          recipe.Task             `json:"task"`
	Stages        []recipe.Stage          `json:"stages"`
	RequiredFacts []recipe.Dependency     `json:"required_facts"`
	Inputs        []recipe.Input          `json:"inputs,omitempty"`
	Outputs       []recipe.Output         `json:"outputs"`
	Evidence      []artifact.ID           `json:"evidence,omitempty"`
	CacheIdentity artifact.ID             `json:"cache_identity"`
	Interaction   recipe.InteractionScope `json:"-"`
}

func Describe(plan Plan) (RuntimeDescription, error) {
	if err := plan.ValidateServing(); err != nil {
		return RuntimeDescription{}, err
	}
	program, err := recipe.CompileProgram(plan.Recipe, catalog)
	if err != nil {
		return RuntimeDescription{}, err
	}
	definition := program.Definition()
	interaction, err := program.InteractionScope(definition.Outputs[0].Source.Node)
	if err != nil {
		return RuntimeDescription{}, err
	}
	return RuntimeDescription{
		Identity:      plan.Identity,
		Task:          definition.Task,
		Stages:        program.Stages(),
		RequiredFacts: slices.Clone(definition.Dependencies),
		Inputs:        slices.Clone(definition.Inputs),
		Outputs:       slices.Clone(definition.Outputs),
		Evidence:      slices.Clone(plan.Evidence),
		CacheIdentity: plan.Identity.Recipe,
		Interaction:   interaction,
	}, nil
}
