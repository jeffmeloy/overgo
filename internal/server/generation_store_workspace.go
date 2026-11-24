package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"sync"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/discovery"
	"overgo/internal/hfrepo"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/processmeasure"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/servingattempt"
)

// generationCatalog is what the store generation workspace needs from the
// media capability catalog, supplied by the assembly root so this package
// names no media executor package: one executor per media task, the
// controls a stage's request declares for a model directory, and the
// media artifact an executor's output becomes.
type generationCatalog struct {
	Execute map[recipe.Task]capabilityruntime.Executor
	// Preview shows, per task that previews, what a run reads of its input before any run (a MOSS depth section's map).
	Preview       map[recipe.Task]capabilityruntime.Executor
	Controls      func(entry recipe.ModuleID, directory string) ([]workflowControl, string)
	OutputContent func(output any) (artifact.Content, error)
}

// BindGenerationCatalog builds the catalog from a capability catalog's
// executors, its control declarations and its output publisher, so the
// launcher and the tests bind the same shape once while the catalog's own
// types stay outside this package: a capability names its executor and a
// control reports its declaration as one tuple, an object or a list control
// its parts as controls of its own type.
func BindGenerationCatalog[
	Capability interface {
		Executor() capabilityruntime.Executor
		// Previewer reports what shows a run's input before any run, nil where the capability previews nothing.
		Previewer() capabilityruntime.Executor
	},
	Control interface {
		Declared() (name, kind string, required bool, choices []string)
		// Defaulted reports the value a request that leaves the field out is given, empty where none is declared.
		Defaulted() string
		// Accompanies reports the field the control applies with, empty where it applies alone.
		Accompanies() string
		// Slot reports an artifact control's label and media kind (empty for a typed field).
		Slot() (label, media string)
		// Bounded reports a numeric control's declared default, step and rate; declared is false without them.
		Bounded() (defaultValue, step, rate int, declared bool)
		// Secondary reports whether the page should collapse an optional tuning control.
		Secondary() bool
		// Guidance describes the control's accepted input.
		Guidance() string
		// Labelled reports the common control a field is on every media page, the name a page shows it by; empty
		// where the page names it from the field.
		Labelled() string
		// Nested reports an object control's fields and a list control's item as the catalog's own control types, a
		// []Control and a *Control, none for a scalar: a constraint whose method named its own type parameter would hold
		// no type, so the walk that reaches declarations (internal/repoanalysis) would find none implementing it.
		Nested() (fields, item any)
		// ChoiceSettings reports what each of the control's choices fills, none where its choices fill nothing.
		ChoiceSettings() map[string]json.RawMessage
		// ChoiceShows reports the controls each of a choice control's choices shows, none where they hide nothing.
		ChoiceShows() map[string][]string
		// ChoiceNames reports the plain name a page offers each of a choice control's choices by, none where it offers
		// them as written.
		ChoiceNames() map[string]string
	},
](
	catalog map[recipe.Task]Capability,
	controls func(recipe.ModuleID, string) ([]Control, string),
	output func(any) (artifact.Content, error),
) generationCatalog {
	execute, preview := make(map[recipe.Task]capabilityruntime.Executor, len(catalog)), map[recipe.Task]capabilityruntime.Executor{}
	for task, capability := range catalog {
		execute[task] = capability.Executor()
		if previewer := capability.Previewer(); previewer != nil {
			preview[task] = previewer
		}
	}
	// bind is the controls as a page takes them, an object's fields and a list's item each bound as a control is.
	var bind func([]Control) []workflowControl
	bind = func(declared []Control) []workflowControl {
		bound := make([]workflowControl, 0, len(declared))
		for _, control := range declared {
			name, kind, required, choices := control.Declared()
			label, media := control.Slot()
			wire := workflowControl{Name: name, Type: workflowControlType(kind), Required: required, Advanced: control.Secondary(), Choices: choices, Default: control.Defaulted(), With: control.Accompanies(), Label: label, Media: media, Help: control.Guidance(), Role: control.Labelled(), Settings: control.ChoiceSettings(), Shows: control.ChoiceShows(), Names: control.ChoiceNames()}
			if defaultValue, step, rate, declared := control.Bounded(); declared {
				wire.Bounds = &workflowControlBounds{Default: defaultValue, Step: step, Rate: rate}
			}
			fields, item := control.Nested()
			if parts, _ := fields.([]Control); len(parts) > 0 {
				wire.Fields = bind(parts)
			}
			if part, _ := item.(*Control); part != nil {
				wire.Item = &bind([]Control{*part})[0]
			}
			bound = append(bound, wire)
		}
		return bound
	}
	return generationCatalog{
		Execute: execute,
		Preview: preview,
		Controls: func(entry recipe.ModuleID, directory string) ([]workflowControl, string) {
			declared, refusal := controls(entry, directory)
			return bind(declared), refusal
		},
		OutputContent: output,
	}
}

// storeGenerationWorkspace is the generation workspace over the store
// (professional GUI campaign, gui-generation-workspace): every active or
// candidate media recipe the store declares lists as a capability with the controls
// its request declares and the media inputs its recipe binds, executes
// through the same executors the recipe command uses, and publishes its
// output as a media artifact. Any launcher gets it with the store; no
// flag and no model name is involved.
type storeGenerationWorkspace struct {
	store   *overgodb.Store
	catalog generationCatalog
	limit   int
	mu      sync.Mutex
	memo    *discovery.Memo
}

// NewStoreGenerationWorkspace binds the workspace to the store and to the
// capability catalog the recipe command shares.
func NewStoreGenerationWorkspace(store *overgodb.Store, catalog generationCatalog, limit int) *storeGenerationWorkspace {
	return &storeGenerationWorkspace{store: store, catalog: catalog, limit: limit}
}

// generationTasks are the tasks a generation capability serves: the media
// outputs and the text answer to a question about an image.
var generationTasks = []recipe.Task{recipe.TaskImageGen, recipe.TaskVideoGen, recipe.TaskSpeech, recipe.TaskVQA}

// generationRunKeyPrefix roots every generation run record's batch key.
const generationRunKeyPrefix = "generation/run/"

// recordedInput is the request document the runtime recorded for an
// executed output (the decoded request under the capability's input
// schema): the run cites it as its input, so a gallery re-opens the
// record beside the output and a page regenerates from it. A bare
// output outside the measured envelope records none.
func recordedInput(output any) (artifact.Content, bool) {
	measured, ok := output.(capabilityruntime.Measured)
	if !ok || measured.Input.Descriptor.ID == (artifact.ID{}) {
		return artifact.Content{}, false
	}
	return measured.Input, true
}

// WorkflowCapabilities lists one capability per active or candidate media
// recipe with present bytes, controls declared by the executor's request
// type, its verification beside it, and a refusal on the entry when its
// request carries what a page cannot type.
func (workspace *storeGenerationWorkspace) WorkflowCapabilities(ctx context.Context, kind WorkflowKind) ([]WorkflowCapability, error) {
	if workspace == nil || kind != workflowGeneration {
		return nil, nil
	}
	// Retain verified file identities; refresh activations on every read.
	workspace.mu.Lock()
	defer workspace.mu.Unlock()
	if err := workspace.store.Refresh(ctx); err != nil {
		return nil, err
	}
	if workspace.memo == nil {
		workspace.memo = discovery.LoadMemo(ctx, workspace.store)
	}
	entries, _, err := discovery.CapabilityCatalogForTasks(ctx, workspace.store, workspace.limit, workspace.memo, generationTasks...)
	if err != nil {
		return nil, err
	}
	var capabilities []WorkflowCapability
	listed := map[artifact.ID]bool{}
	for _, entry := range entries {
		if !entry.Present {
			continue
		}
		listed[entry.Model] = true
		for _, capability := range entry.Capabilities {
			if capability.Stale != "" || !slices.Contains(generationTasks, capability.Task) {
				continue
			}
			if _, known := workspace.catalog.Execute[capability.Task]; !known {
				continue
			}
			selection, executionErr := modelrecipe.ResolveActiveExecution(ctx, workspace.store, entry.Model, capability.Task, modelrecipe.SessionWarm)
			program, refusal, state := selection.Program, "", modelrecipe.StateVerified
			if executionErr != nil {
				// An activation whose execution the store cannot resolve (a
				// recipe declaring no component session, a model without a
				// recorded extent) lists with the reason rather than failing
				// the workspace for every other activation.
				var err error
				if _, program, err = modelrecipe.ResolveActiveCapability(ctx, workspace.store, entry.Model, capability.Task); err != nil {
					return nil, fmt.Errorf("generation workspace: %s for %s: %w", capability.Task, entry.Model, err)
				}
				refusal, state = executionErr.Error(), modelrecipe.StateBroken
			}
			directory, _ := workspace.executionPath(ctx, entry.Model, entry.Location)
			if listing, staged := workspace.listing(entry, directory, program, selection, refusal, state); staged {
				capabilities = append(capabilities, listing)
			}
		}
	}
	registered, _, err := discovery.RegisteredCatalog(ctx, workspace.store, workspace.limit, workspace.memo)
	if err != nil {
		return nil, err
	}
	for _, entry := range registered {
		if !entry.Present || listed[entry.Model] || len(entry.Capabilities) != 0 {
			continue
		}
		directory, err := workspace.executionPath(ctx, entry.Model, entry.Location)
		if err != nil {
			continue
		}
		candidates := workspace.candidates(ctx, entry, directory)
		if len(candidates) == 0 {
			candidates = unrecipedMedia(entry, directory)
		}
		capabilities = append(capabilities, candidates...)
	}
	return capabilities, nil
}

// listing is the capability a page shows for program over entry's files, with the controls its first stage declares;
// staged is false for a program with no stage to run.
func (workspace *storeGenerationWorkspace) listing(entry discovery.CatalogEntry, directory string, program recipe.Program, execution modelrecipe.CapabilityEvidenceSelection, refusal string, state modelrecipe.VerificationState) (WorkflowCapability, bool) {
	definition := program.Definition()
	stages := program.Stages()
	if len(stages) == 0 {
		return WorkflowCapability{}, false
	}
	var declared []workflowControl
	if workspace.catalog.Controls != nil && refusal == "" {
		declared, refusal = workspace.catalog.Controls(stages[0].Module.ID, directory)
	}
	return WorkflowCapability{
		Task: definition.Task, Recipe: definition.ID, Stages: stages,
		Inputs: definition.Inputs, Outputs: definition.Outputs, Controls: declared,
		Model: entry.Model, Name: filepath.Base(directory), Location: entry.Location, Refusal: refusal, Verification: state,
		execution: execution, directory: directory,
	}, true
}

// candidates lists each media recipe the catalog holds over entry's model, which nothing has activated: offered to run as
// validation would run it, with its state beside it, since nothing has verified it.
func (workspace *storeGenerationWorkspace) candidates(ctx context.Context, entry discovery.CatalogEntry, directory string) []WorkflowCapability {
	var candidates []WorkflowCapability
	for _, definition := range entry.Candidates {
		if _, known := workspace.catalog.Execute[definition.Task]; !known || !slices.Contains(generationTasks, definition.Task) {
			continue
		}
		// A candidate whose execution does not resolve is shown with why, as a broken activation is; one this build
		// cannot compile has no stage to show.
		program, err := modelrecipe.CompileCapability(definition)
		var execution modelrecipe.CapabilityEvidenceSelection
		if err == nil {
			execution, err = modelrecipe.CompileCandidateExecution(ctx, workspace.store, program)
		}
		refusal := ""
		if err != nil {
			refusal = err.Error()
		}
		if listing, staged := workspace.listing(entry, directory, program, execution, refusal, modelrecipe.StateNotVerified); staged {
			candidates = append(candidates, listing)
		}
	}
	return candidates
}

// unrecipedMedia lists a media model the store holds with no recipe this build can run, as a capability the page shows
// and cannot run, saying what it needs: a source profile names its task, and no recipe it holds compiles. A model whose files no
// profile names lists nothing here, since nothing can say what it would run.
func unrecipedMedia(entry discovery.CatalogEntry, directory string) []WorkflowCapability {
	identity, err := hfrepo.InspectIdentity(directory)
	if err != nil {
		return nil
	}
	source, err := modelrecipe.ResolveGenerationSource("", identity)
	if err != nil || !slices.Contains(generationTasks, source.Task) {
		return nil
	}
	return []WorkflowCapability{{
		Task: source.Task, Model: entry.Model, Name: filepath.Base(directory), Location: entry.Location,
		Refusal: fmt.Sprintf("no %s recipe this build can run is held for this model: go run ./cmd/validate -onboard %s registers and validates one",
			source.Task, filepath.ToSlash(directory)),
		Verification: modelrecipe.StateNotVerified,
	}}
}

// ExecuteWorkflow runs a listed capability against its model bytes with the
// execution its listing resolved and the request the page typed, publishes
// the output as a media artifact and a run record, and returns both as the
// operation's receipt.
func (workspace *storeGenerationWorkspace) ExecuteWorkflow(ctx context.Context, kind WorkflowKind, capability WorkflowCapability, raw json.RawMessage, reporter operation.Reporter) (operation.Completion, error) {
	if workspace == nil || ctx == nil || reporter == nil || kind != workflowGeneration {
		return operation.Completion{}, errors.New("generation workspace: workflow is not admitted")
	}
	if capability.Refusal != "" {
		return operation.Completion{}, errors.New("generation workspace: " + capability.Refusal)
	}
	task, recipeID, listed := capability.Task, capability.Recipe, capability.execution.Activation.Definition
	execute, known := workspace.catalog.Execute[task]
	if !known || listed.ID != recipeID || listed.Task != task {
		return operation.Completion{}, fmt.Errorf("generation workspace: recipe %s is not a listed %s capability", recipeID, task)
	}
	if workspace.catalog.OutputContent == nil {
		return operation.Completion{}, errors.New("generation workspace: the catalog publishes no output")
	}
	// The submission's sources (a prompt enhancement the page accepted) are
	// the run's inputs beside its request, so the original stays its source.
	sources := workflowSources(ctx)
	started := time.Now()
	begin, _ := processmeasure.Counter()
	var repository artifact.Repository = workspace.store
	phases := generationPhases(capability.Stages)
	if len(phases) != 0 {
		reporter.Text("Loading model")
		repository = &generationProgressRepository{Store: workspace.store, recipe: recipeID, phases: phases, reporter: reporter}
	}
	// Each sampling step the model finishes is the operation's progress, which the page shows as the run goes
	// (moss-page/result).
	stepped := processmeasure.WithSteps(ctx, func(done, total int) {
		steps := uint64(total)
		reporter.Progress(uint64(done), &steps)
	})
	output, err := execute(stepped, repository, capability.directory, capability.execution, string(raw))
	end, _ := processmeasure.Counter()
	measured, _ := output.(capabilityruntime.Measured)
	var content artifact.Content
	if err == nil {
		content, err = workspace.catalog.OutputContent(capabilityruntime.Unwrap(output))
	}
	if err != nil {
		completion, failed := failWorkflow(ctx, workspace.store, recipeID, sources, generationFailure, err)
		workspace.observe(ctx, capability.Model, recipeID, completion.Run, task, started, end-begin, measured, failed)
		return completion, failed
	}
	// The recorded request is the run's input: the record a gallery opens.
	inputs := slices.Clone(sources)
	request, recorded := recordedInput(output)
	if recorded {
		inputs = append(inputs, request.Descriptor.ID)
	}
	// What the output was drawn from, where it says so, is an output of the run beside it, which the page shows; so is
	// the picture it was before its finishing sharpened it, the run's first output (moss-sharpened-pair).
	published := []artifact.Content{content}
	if before, sharpened := beforeFinishing(capabilityruntime.Unwrap(output)); sharpened && err == nil {
		var first artifact.Content
		if first, err = workspace.catalog.OutputContent(before); err == nil {
			published = append([]artifact.Content{first}, published...)
		}
	}
	if used, drawn, usedErr := promptsUsedContent(capabilityruntime.Unwrap(output)); usedErr != nil {
		err = usedErr
	} else if drawn {
		published = append(published, used)
	}
	outputs := make([]artifact.ID, len(published))
	for index, output := range published {
		outputs[index] = output.Descriptor.ID
	}
	var run runrecord.Run
	if err == nil {
		run, err = runrecord.NewRun(recipeID, runrecord.OutcomeSucceeded, inputs, outputs, "")
	}
	var batch artifact.Batch
	if err == nil {
		batch, err = run.Batch(generationRunKeyPrefix + run.ID.String())
	}
	if err == nil {
		if recorded {
			batch.Contents = append(batch.Contents, request)
		}
		batch.Contents = append(batch.Contents, published...)
		_, err = artifact.Publish(ctx, workspace.store, batch)
	}
	if err != nil {
		// A Stop in this window, or any refused publication, ends as a recorded run: the operation's
		// completion needs one, and no success was stored for it to name.
		completion, failed := failWorkflow(ctx, workspace.store, recipeID, sources, generationFailure, err)
		workspace.observe(ctx, capability.Model, recipeID, completion.Run, task, started, end-begin, measured, failed)
		return completion, failed
	}
	workspace.observe(ctx, capability.Model, recipeID, run.ID, task, started, end-begin, measured, nil)
	reporter.Publishing()
	return operation.Completion{Run: run.ID, Outputs: outputs}, nil
}

// generationPreview is what a preview shows of a run's input: a stored picture (a MOSS depth section's map) and its
// media type, which a page shows beside the controls it previews.
type generationPreview struct {
	Output    artifact.ID `json:"output"`
	MediaType string      `json:"media_type"`
}

// PreviewWorkflow shows what a listed capability's run would read of the page's request before any run, as the
// capability's preview makes it (a MOSS depth section's depth map as the workflow's PreviewImage shows it), and stores
// it so the page shows it; a capability that previews nothing answers errWorkflowPreviewUnavailable. A preview is one
// picture, so its page is the whole of it.
func (workspace *storeGenerationWorkspace) PreviewWorkflow(ctx context.Context, kind WorkflowKind, task recipe.Task, recipeID artifact.ID, raw json.RawMessage, _, _ int) (any, error) {
	capabilities, err := workspace.WorkflowCapabilities(ctx, kind)
	if err != nil {
		return nil, err
	}
	capability, err := selectWorkflowCapability(capabilities, task, recipeID)
	if err != nil {
		return nil, err
	}
	preview := workspace.catalog.Preview[task]
	if preview == nil || workspace.catalog.OutputContent == nil {
		return nil, errWorkflowPreviewUnavailable
	}
	if capability.Refusal != "" {
		return nil, errors.New("generation workspace: " + capability.Refusal)
	}
	if err := validateWorkflowInput(capability.Controls, raw); err != nil {
		return nil, err
	}
	output, err := preview(ctx, workspace.store, capability.directory, capability.execution, string(raw))
	if err != nil {
		return nil, err
	}
	content, err := workspace.catalog.OutputContent(capabilityruntime.Unwrap(output))
	if err != nil {
		return nil, err
	}
	if _, err := artifact.Publish(ctx, workspace.store, artifact.Batch{Key: "generation/preview/" + content.Descriptor.ID.String(),
		Artifacts: []artifact.Descriptor{content.Descriptor}, Contents: []artifact.Content{content}}); err != nil {
		return nil, err
	}
	return generationPreview{Output: content.Descriptor.ID, MediaType: content.Descriptor.MediaType}, nil
}

// generationPhases names the existing image pipeline's user-visible stages.
func generationPhases(stages []recipe.Stage) map[recipe.NodeID]string {
	labels := map[recipe.NodeID]string{}
	for _, stage := range stages {
		var label string
		switch stage.Module.ID {
		case modelrecipe.ModuleLatentImagePrepare:
			label = "Processing prompt"
		case modelrecipe.ModuleLatentImageIntegrate:
			label = "Generating image"
		case modelrecipe.ModuleLatentImageDecode:
			label = "Decoding image"
		}
		if label != "" {
			labels[stage.Node.ID] = label
		}
	}
	return labels
}

// generationProgressRepository projects durable stages onto the operation stream.
// Failed commits and receipts for other recipes publish no progress.
type generationProgressRepository struct {
	*overgodb.Store
	recipe   artifact.ID
	phases   map[recipe.NodeID]string
	reporter operation.Reporter
}

// Commit publishes progress only after the underlying stage batch is durable.
func (repository *generationProgressRepository) Commit(ctx context.Context, batch artifact.Batch) (artifact.CommitID, error) {
	id, err := repository.Store.Commit(ctx, batch)
	if err != nil {
		return id, err
	}
	for _, content := range batch.Contents {
		if content.Descriptor.Schema != runrecord.StageReceiptSchema {
			continue
		}
		receipt, parseErr := runrecord.ParseStageReceipt(content.Data)
		if parseErr != nil || receipt.Recipe != repository.recipe || receipt.State != runrecord.StageRunning {
			continue
		}
		if label := repository.phases[receipt.Node]; label != "" {
			repository.reporter.Text(label)
		}
	}
	return id, nil
}

// generationFailure is the failure code a failed media execution records.
const generationFailure = "generation_failed"

// observe records one media execution as the serving attempt the scorecard
// counts. The output is already made or refused, so a record the store
// refuses fails nothing.
func (workspace *storeGenerationWorkspace) observe(ctx context.Context, model, recipeID, run artifact.ID, task recipe.Task, started time.Time, wall time.Duration, measured capabilityruntime.Measured, cause error) {
	// The run the executor recorded is the one that made the attempt; the workspace's stands in where it reported none.
	if measured.Run.Valid() {
		run = measured.Run
	}
	_, _ = servingattempt.RecordWholeOutput(context.WithoutCancel(ctx), workspace.store, servingattempt.WholeOutput{
		Model: model, Recipe: recipeID, Run: run, Task: task, Started: started, Wall: wall,
		Load: measured.Load, Decode: measured.Decode, OutputTokens: measured.OutputTokens,
		// The server's process holds other models, so the executor's own peak is the attempt's, and a zero of it is a
		// measurement only where the session's meter was read: a model with no meter, a failed read and a stored answer
		// leave the peak absent, which the scorecard must not read as a model that runs on the host.
		PeakDeviceBytes: measured.PeakDeviceBytes, PeakMeasured: cause == nil && measured.PeakMeasured, Preset: media.PresetOf(measured.Output), Encoding: media.EncodingOf(measured.Output), Cause: cause,
	})
}

// executionPath names the model directory an executor reads: every media
// capability resolves a repository directory (config beside weights), so
// the most specific directory recorded for the model serves when the store
// holds one (a composite's own, never the parent its inventory shares),
// else the directory holding the present file the catalog resolved.
func (workspace *storeGenerationWorkspace) executionPath(ctx context.Context, model artifact.ID, location string) (string, error) {
	if path, err := discovery.ExecutionDirectory(ctx, workspace.store, model); err == nil {
		return path, nil
	}
	if location == "" {
		return "", fmt.Errorf("generation workspace: model %s has no bytes on disk", model)
	}
	return filepath.Dir(location), nil
}

// promptsUsedContract is the document of what a picture was drawn from, its prompt and negative prompt, published
// beside it as an output of its run (moss-prompts-used).
var promptsUsedContract = artifact.DocumentContract{Kind: artifact.KindOutput, MediaType: "application/json", Schema: "overgo.prompts-used.v1"}

// promptsUsed is what a picture was drawn from, as the MOSS Ultra workflow shows it beside each section's picture
// ("Prompt used", "Negative prompt used").
type promptsUsed struct {
	Prompt         string `json:"prompt"`
	NegativePrompt string `json:"negative_prompt"`
}

// promptsUsedContent is the document of the prompt and negative prompt output was drawn from, where it says so (a MOSS
// picture, latentimage EncodedImage.DrawnFrom), and false where it says nothing.
func promptsUsedContent(output any) (artifact.Content, bool, error) {
	drawn, says := output.(interface {
		DrawnFrom() (prompt, negative string)
	})
	if !says {
		return artifact.Content{}, false, nil
	}
	prompt, negative := drawn.DrawnFrom()
	if prompt == "" && negative == "" {
		return artifact.Content{}, false, nil
	}
	content, err := artifact.JSONContent(promptsUsedContract, promptsUsed{Prompt: prompt, NegativePrompt: negative})
	return content, err == nil, err
}

// beforeFinishing is the picture output was before its finishing sharpened it, as its capability publishes a picture,
// where it says so (a MOSS picture, latentimage EncodedImage.BeforeFinishing), and false where it was not sharpened.
func beforeFinishing(output any) (any, bool) {
	finished, says := output.(interface{ BeforeFinishing() (any, bool) })
	if !says {
		return nil, false
	}
	return finished.BeforeFinishing()
}
