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

// Describe: runtime description of a validated serving plan.
func Describe(plan Plan) (RuntimeDescription, error) {
	if err := plan.ValidateServing(); err != nil {
		return RuntimeDescription{}, err
	}
	return DescribeDefinition(plan.Identity, plan.Recipe, plan.Evidence)
}

// DescribeDefinition compiles definition into its runtime description
// under identity; interaction scope = the first output's node. Shared by
// serving plans (Describe) and the remote relay (no model plan to validate).
func DescribeDefinition(identity ProgramIdentity, definition recipe.Definition, evidence []artifact.ID) (RuntimeDescription, error) {
	program, err := recipe.CompileProgram(definition, catalog)
	if err != nil {
		return RuntimeDescription{}, err
	}
	compiled := program.Definition()
	interaction, err := program.InteractionScope(compiled.Outputs[0].Source.Node)
	if err != nil {
		return RuntimeDescription{}, err
	}
	return RuntimeDescription{
		Identity:      identity,
		Task:          compiled.Task,
		Stages:        program.Stages(),
		RequiredFacts: slices.Clone(compiled.Dependencies),
		Inputs:        slices.Clone(compiled.Inputs),
		Outputs:       slices.Clone(compiled.Outputs),
		Evidence:      slices.Clone(evidence),
		CacheIdentity: identity.Recipe,
		Interaction:   interaction,
	}, nil
}
