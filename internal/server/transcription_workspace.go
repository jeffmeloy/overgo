package server

import (
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
	"overgo/internal/strictjson"
)

const transcriptionUploadLimit = "transcription-upload-limit"

// TranscriptionPolicy supplies resource bounds and an optional recipe pin.
// Native server startup binds an omitted pin to the selected model's active
// transcription recipe. A workspace always requires that binding.
type TranscriptionPolicy struct {
	Recipe      artifact.ID                   `json:"recipe,omitzero"`
	MemoryBytes uint64                        `json:"memory_bytes"`
	Inspection  dataset.AudioInspectionPolicy `json:"inspection"`
}

// LoadTranscriptionPolicy reads one bounded, strict server policy document.
func LoadTranscriptionPolicy(path string) (TranscriptionPolicy, error) {
	var policy TranscriptionPolicy
	if err := loadPolicyDocument(path, &policy); err != nil {
		return policy, err
	}
	if policy.MemoryBytes == 0 || policy.Recipe.Valid() && policy.Recipe.Kind() != artifact.KindRecipe {
		return policy, errors.New("transcription policy: memory bound and optional recipe identity required")
	}
	return policy, policy.Inspection.Validate()
}

func (policy TranscriptionPolicy) validate() error {
	if policy.Recipe.Kind() != artifact.KindRecipe || policy.MemoryBytes == 0 {
		return errors.New("transcription workspace: recipe and memory bound required")
	}
	return policy.Inspection.Validate()
}

// TranscriptionWorkspace adapts the native transcriber to the workflow owner.
type TranscriptionWorkspace struct {
	store          *overgodb.Store
	policy         TranscriptionPolicy
	program        recipe.Program
	sessions       *speechrecognition.Session
	commit         string
	environment    artifact.ID
	name, location string
}

// NewTranscriptionWorkspace loads one exact recipe on the CPU. The assembly
// caller supplies the verified source commit; all inference runs retain it.
func NewTranscriptionWorkspace(ctx context.Context, store *overgodb.Store, policy TranscriptionPolicy, commit string) (*TranscriptionWorkspace, error) {
	revision, revisionErr := runrecord.NewCodeRevision(strings.ToLower(commit))
	if ctx == nil || store == nil || revisionErr != nil {
		return nil, errors.New("transcription workspace: store, context and source commit required")
	}
	if err := policy.validate(); err != nil {
		return nil, err
	}
	if err := store.Refresh(ctx); err != nil {
		return nil, err
	}
	definition, err := recipe.RequireDefinition(ctx, store, policy.Recipe)
	if err != nil {
		return nil, err
	}
	activation, program, err := modelrecipe.ResolveActiveCapability(ctx, store, definition.Model, recipe.TaskTranscription)
	if err != nil {
		return nil, err
	}
	if activation.Definition.ID != policy.Recipe {
		return nil, errors.New("transcription workspace: configured recipe is not active")
	}
	environment, err := runrecord.CurrentEnvironment("cpu", "go")
	if err != nil {
		return nil, err
	}
	batch, err := environment.Batch("transcription/environment/" + environment.ID.String())
	if err != nil {
		return nil, err
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return nil, err
	}
	sessions, err := speechrecognition.LoadSession(ctx, store, policy.Recipe, policy.MemoryBytes)
	if err != nil {
		return nil, err
	}
	// A source label is optional presentation metadata; the loaded recipe and
	// model identity remain authoritative when no directory location is recorded.
	name, location := "Transcription", ""
	if directory, err := artifact.AvailablePath(ctx, store, definition.Model, artifact.LocationDirectory); err == nil {
		name, location = filepath.Base(directory), directory
	}
	return &TranscriptionWorkspace{store: store, policy: policy, program: program,
		sessions: sessions, commit: revision.Commit, environment: environment.ID, name: name, location: location}, nil
}

// Close drains and releases the recipe's resident model and mutable buffers.
func (workspace *TranscriptionWorkspace) Close(ctx context.Context) error {
	if workspace == nil || workspace.sessions == nil {
		return nil
	}
	return workspace.sessions.Close(ctx)
}

// WorkflowCapabilities exposes only the configured offline transcription recipe.
func (workspace *TranscriptionWorkspace) WorkflowCapabilities(_ context.Context, kind WorkflowKind) ([]WorkflowCapability, error) {
	if workspace == nil || workspace.sessions == nil || kind != WorkflowGeneration {
		return nil, nil
	}
	definition := workspace.program.Definition()
	return []WorkflowCapability{{Task: definition.Task, Recipe: definition.ID, Stages: workspace.program.Stages(), model: definition.Model, Model: definition.Model, Name: workspace.name, Location: workspace.location,
		transcriptionStream: workspace.openTranscriptionStream,
		Inputs:              definition.Inputs, Outputs: definition.Outputs,
		// The audio control names a stored file artifact: an attachment the
		// page stored through the intake route, or a card's stored id.
		Controls: []WorkflowControl{{Name: "audio", Type: WorkflowControlArtifact, Required: true, Label: "audio clip", Media: "audio"}},
	}}, nil
}

type transcriptionWorkflowInput struct {
	Audio artifact.ID `json:"audio"`
}

// ExecuteWorkflow performs fresh inference using the existing component lease.
func (workspace *TranscriptionWorkspace) ExecuteWorkflow(ctx context.Context, kind WorkflowKind, task recipe.Task, recipeID artifact.ID, raw json.RawMessage, reporter operation.Reporter) (completion operation.Completion, err error) {
	if workspace == nil || workspace.sessions == nil || ctx == nil || reporter == nil || kind != WorkflowGeneration || task != recipe.TaskTranscription || recipeID != workspace.policy.Recipe {
		return completion, errors.New("transcription workspace: workflow is not admitted")
	}
	defer func() {
		if err != nil && !completion.Run.Valid() {
			completion, err = failWorkflow(ctx, workspace.store, recipeID, []artifact.ID{workspace.program.Definition().Model}, "transcription_failed", err)
		}
	}()
	var input transcriptionWorkflowInput
	if err = strictjson.DecodeBytes(raw, &input); err != nil {
		return completion, err
	}
	if input.Audio.Kind() != artifact.KindFile {
		return completion, errors.New("transcription workspace: encoded audio file required")
	}
	descriptor, found, err := workspace.store.Artifact(ctx, input.Audio)
	if err != nil {
		return completion, err
	}
	if !found || descriptor.Size == 0 {
		return completion, errors.New("transcription workspace: audio is absent or empty")
	}
	if descriptor.Size > workspace.policy.Inspection.MaximumEncodedBytes {
		return failWorkflow(ctx, workspace.store, recipeID, []artifact.ID{workspace.program.Definition().Model, input.Audio},
			transcriptionUploadLimit, errors.New("transcription workspace: audio exceeds the encoded-byte bound"))
	}
	lease, err := workspace.sessions.Lease(ctx)
	if err != nil {
		return completion, err
	}
	defer func() { err = errors.Join(err, lease.Release()) }()
	// Recheck after admission: a recipe may be retired while this request waits.
	if err := workspace.checkActivation(ctx); err != nil {
		return completion, err
	}
	content, found, err := artifact.ReadContent(ctx, workspace.store, input.Audio)
	if err != nil {
		return completion, err
	}
	if !found {
		return completion, errors.New("transcription workspace: audio content is absent")
	}
	transcription, run, err := lease.Transcribe(ctx, content.Data, dataset.AudioPayloadOrigin{Container: input.Audio},
		workspace.policy.Inspection, speechrecognition.RunBinding{
			Key: "transcription/run/" + reporter.OperationID().String(), CodeCommit: workspace.commit, Environment: workspace.environment,
		})
	completion = operation.Completion{Run: run.ID, Outputs: run.Outputs}
	if err != nil || transcription.Text == "" {
		return completion, err
	}
	// The transcript's text beside the run's transcription document: the
	// page renders the text output as the assistant's turn.
	text, err := modelrecipe.TextAnswerContract.OwnedContentBytes([]byte(transcription.Text))
	if err != nil {
		return completion, err
	}
	if _, err := artifact.CommitBatch(ctx, workspace.store, artifact.Batch{
		Key: "transcription/text/" + text.Descriptor.ID.String(), Contents: []artifact.Content{text},
	}); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return completion, err
	}
	completion.Outputs = append(slices.Clone(run.Outputs), text.Descriptor.ID)
	return completion, nil
}

func (workspace *TranscriptionWorkspace) checkActivation(ctx context.Context) error {
	if err := workspace.store.Refresh(ctx); err != nil {
		return err
	}
	activation, _, err := modelrecipe.ResolveActiveCapability(ctx, workspace.store, workspace.program.Definition().Model, recipe.TaskTranscription)
	if err != nil {
		return err
	}
	if activation.Definition.ID != workspace.policy.Recipe {
		return errors.New("transcription workspace: configured recipe is no longer active")
	}
	return nil
}
