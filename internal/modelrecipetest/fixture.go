// Package modelrecipetest owns executable capability fixtures.
package modelrecipetest

import (
	"context"
	"fmt"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

const verificationCodeCommit = "0123456789abcdef0123456789abcdef01234567"

type Capability struct {
	Model   artifact.ID
	Program recipe.Program
	Runtime *workflowruntime.Runtime
}

// CandidateExecution compiles the real pre-activation capability selection for
// runtime and media-executor fixtures; failures remain fatal to the calling test.
func CandidateExecution(t testing.TB, store artifact.Reader, program recipe.Program) modelrecipe.CapabilityEvidenceSelection {
	t.Helper()
	execution, err := modelrecipe.CompileCandidateExecution(t.Context(), store, program)
	if err != nil {
		t.Fatal(err)
	}
	return execution
}

// PublishModelDefinition publishes a small, fully typed model-definition
// authority suitable for tests that must reject descriptor-only model claims.
func PublishModelDefinition(
	ctx context.Context,
	store artifact.Repository,
	key string,
	modelID artifact.ID,
) (modelrecipe.ResolvedModelDefinition, error) {
	if ctx == nil || store == nil || modelID.Kind() != artifact.KindModel {
		return modelrecipe.ResolvedModelDefinition{}, fmt.Errorf("model recipe fixture: invalid model-definition publication")
	}
	profile, found := model.LookupArchitecture("llama")
	if !found {
		return modelrecipe.ResolvedModelDefinition{}, fmt.Errorf("model recipe fixture: llama profile is absent")
	}
	profileDocument, err := modelrecipe.NewProfileDocument(profile)
	if err != nil {
		return modelrecipe.ResolvedModelDefinition{}, err
	}
	tensors, err := modelartifact.NewTensorInventoryDocument(
		modelID,
		modelartifact.TensorFormatGGUF,
		[]modelartifact.TensorFact{{Name: "weight", Shape: []uint64{}, Storage: "f32", Bytes: 4}},
	)
	if err != nil {
		return modelrecipe.ResolvedModelDefinition{}, err
	}
	spec := model.Spec{
		CommonSpec: model.CommonSpec{
			Architecture: "llama", BlockCount: 1, ContextLength: 128,
			EmbeddingLength: 8, FeedForwardLength: 16, RMSNormEpsilon: 1e-5,
		},
		AttentionSpec: model.AttentionSpec{
			HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, RopeFrequencyBase: 10_000,
		},
	}
	document, err := modelrecipe.NewModelDefinitionDocument(profileDocument, tensors, spec)
	if err != nil {
		return modelrecipe.ResolvedModelDefinition{}, err
	}
	resolved, err := document.Resolve(profileDocument, tensors)
	if err != nil {
		return modelrecipe.ResolvedModelDefinition{}, err
	}
	profileData, err := profileDocument.Content()
	if err != nil {
		return modelrecipe.ResolvedModelDefinition{}, err
	}
	profileContent, err := (artifact.DocumentContract{
		Kind: artifact.KindProfile, MediaType: modelrecipe.ProfileMediaType, Schema: modelrecipe.ProfileSchema,
	}).Content(profileDocument.ID, profileData)
	if err != nil {
		return modelrecipe.ResolvedModelDefinition{}, err
	}
	derivation, err := modelrecipe.NewCatalogProfileDerivation(profile)
	if err != nil {
		return modelrecipe.ResolvedModelDefinition{}, err
	}
	derivationContent, err := derivation.Content()
	if err != nil {
		return modelrecipe.ResolvedModelDefinition{}, err
	}
	tensorContent, err := tensors.Content()
	if err != nil {
		return modelrecipe.ResolvedModelDefinition{}, err
	}
	definitionContent, err := document.Content()
	if err != nil {
		return modelrecipe.ResolvedModelDefinition{}, err
	}
	batch, err := artifact.NewDocumentBatch(
		key,
		[]artifact.Content{derivationContent, profileContent, tensorContent, definitionContent},
		[]artifact.Lineage{
			{Child: profileDocument.ID, Parent: derivation.ID, Relation: artifact.RelationDerivedFrom},
			{Child: tensors.ID, Parent: modelID, Relation: artifact.RelationDerivedFrom},
			{Child: document.ID, Parent: modelID, Relation: artifact.RelationDerivedFrom},
			{Child: document.ID, Parent: profileDocument.ID, Relation: artifact.RelationDependsOn},
			{Child: document.ID, Parent: tensors.ID, Relation: artifact.RelationDependsOn},
		},
		nil,
	)
	if err != nil {
		return modelrecipe.ResolvedModelDefinition{}, err
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		return modelrecipe.ResolvedModelDefinition{}, err
	}
	return resolved, nil
}

func PublishVerification(
	ctx context.Context,
	store artifact.Repository,
	key string,
	recipeID artifact.ID,
) (modelrecipe.Verification, error) {
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: key, OS: "test", Arch: "test", Device: "host", Backend: "go", Driver: "test",
	})
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	environmentBatch, err := environment.Batch(key + "/environment")
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	if _, err := artifact.CommitBatch(ctx, store, environmentBatch); err != nil {
		return modelrecipe.Verification{}, err
	}
	durationNS := uint64(len(key))
	record, err := runrecord.NewGateRecord(
		recipeID, environment.ID, verificationCodeCommit,
		runrecord.OutcomeSucceeded, "", durationNS,
		[]runrecord.GateStep{{
			Name: "verify", Phase: runrecord.PhaseValidate,
			Outcome: runrecord.StepSucceeded, DurationNS: durationNS,
		}},
	)
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	batch, err := record.Batch(key + "/gate")
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		return modelrecipe.Verification{}, err
	}
	return modelrecipe.Verification{Gate: record.Result.ID, Run: record.Run.ID}, nil
}

// PublishActivation provisions a fixture through the real candidate,
// validation, evidence and activation lifecycle, without model loading.
func PublishActivation(ctx context.Context, store artifact.Repository, key string, definition recipe.Definition) error {
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, key+"/candidate", definition); err != nil {
		return err
	}
	if _, _, err := modelrecipe.Transition(ctx, store, key+"/validated", definition, recipe.StatusValidated, nil, nil); err != nil {
		return err
	}
	verification, err := PublishVerification(ctx, store, key+"/verification", definition.ID)
	if err != nil {
		return err
	}
	_, _, err = modelrecipe.ActivateVerified(ctx, store, key+"/active", definition, verification,
		recipe.EvidenceVerified, "serving fixture activation", nil, nil)
	return err
}

func NewCapability(t testing.TB, name string, task recipe.Task) Capability {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	check(t, err)
	t.Cleanup(func() { check(t, store.Close()) })
	modelID := testutil.ArtifactID(t, artifact.KindModel, name)
	testutil.PublishArtifact(t, store, modelID)
	var compiled recipe.Definition
	if task == recipe.TaskImageGen {
		compiled, err = modelrecipe.GenerationDefinition(modelrecipe.ModuleOscillatorImagePrepare, modelID, artifact.ID{})
	} else if task == recipe.TaskVideoGen {
		compiled, err = modelrecipe.GenerationDefinition(modelrecipe.ModuleOscillatorVideoPrepare, modelID, artifact.ID{})
	} else {
		compiled, err = modelrecipe.CapabilityDefinition(task, modelID)
	}
	return newDefinedCapability(t, store, modelID, compiled, err)
}

func newDefinedCapability(t testing.TB, store artifact.Repository, modelID artifact.ID, compiled recipe.Definition, err error) Capability {
	t.Helper()
	check(t, err)
	program, err := modelrecipe.CompileCapability(compiled)
	check(t, err)
	runtime, err := workflowruntime.NewForProgram(store, program)
	check(t, err)
	return Capability{Model: modelID, Program: program, Runtime: runtime}
}

func check(t testing.TB, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func (f Capability) ExecuteScalar(key string, value any) (workflowruntime.Result, error) {
	definition := f.Program.Definition()
	if len(definition.Inputs) != 1 {
		return workflowruntime.Result{}, fmt.Errorf("model recipe fixture: want one input, got %d", len(definition.Inputs))
	}
	input := definition.Inputs[0]
	operation, err := workflowruntime.ExecutionID(definition.ID, key)
	if err != nil {
		return workflowruntime.Result{}, err
	}
	return f.Runtime.ExecuteProgram(context.Background(), key, operation, nil, f.Program, map[recipe.PortName]workflowruntime.Value{
		input.Name: {Kind: input.Data, Items: []workflowruntime.Datum{{Value: value}}},
	})
}

// MustExecuteScalar runs a scalar capability and returns its typed output.
func MustExecuteScalar[Value any](t testing.TB, fixture Capability, key string, input any) Value {
	t.Helper()
	result, err := fixture.ExecuteScalar(key, input)
	check(t, err)
	outputs := fixture.Program.Definition().Outputs
	if len(outputs) != 1 {
		t.Fatalf("model recipe fixture: want one output, got %d", len(outputs))
	}
	return Output[Value](t, result, outputs[0].Name)
}

func Output[Value any](t testing.TB, result workflowruntime.Result, name recipe.PortName) Value {
	t.Helper()
	datum, one := result.Outputs[name].Single()
	value, typed := datum.Value.(Value)
	if !one || !typed || !result.Commit.Valid() {
		t.Fatalf("model recipe output %q = (%T, %v, %v), commit=%v", name, datum.Value, one, typed, result.Commit)
	}
	return value
}
