package libraryintake

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/png"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/mediacapability"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// generationValidationRuns is the minimum number of fresh executions that can prove repeat identity.
const generationValidationRuns = 2

func prepareGenerationCandidate(ctx context.Context, store artifact.Repository, path string, task recipe.Task, capability mediacapability.Capability) (artifact.ID, recipe.Definition, error) {
	model, definition, err := mediacapability.Prepare(ctx, store, path, task, capability)
	if err != nil {
		return artifact.ID{}, recipe.Definition{}, err
	}
	if _, published, err := modelrecipe.Status(ctx, store, definition.ID); err != nil {
		return artifact.ID{}, recipe.Definition{}, err
	} else if !published {
		if _, _, err := modelrecipe.PublishCandidate(ctx, store, "recipe/candidate/"+definition.ID.String(), definition); err != nil {
			return artifact.ID{}, recipe.Definition{}, err
		}
	}
	return model, definition, nil
}

func prepareGenerationValidation(ctx context.Context, store *overgodb.Store, path string, task recipe.Task, capability mediacapability.Capability, revision, prompt string) (operation.Request, operation.Executor, error) {
	if capability.ValidationInput == nil || capability.Execute == nil || strings.TrimSpace(prompt) == "" {
		return operation.Request{}, nil, refusal{errors.New("library: source has no functional generation validation")}
	}
	input, err := capability.ValidationInput(path, prompt)
	if err != nil {
		return operation.Request{}, nil, refusal{err}
	}
	_, definition, err := prepareGenerationCandidate(ctx, store, path, task, capability)
	if err != nil {
		return operation.Request{}, nil, refusal{err}
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		return operation.Request{}, nil, err
	}
	execution, err := modelrecipe.CompileCandidateExecution(ctx, store, program)
	if err != nil {
		return operation.Request{}, nil, err
	}
	return operation.Request{Task: definition.Task, Recipe: definition.ID}, func(ctx context.Context, reporter operation.Reporter) (operation.Completion, error) {
		return validateGeneration(ctx, store, reporter, path, capability, execution, revision, input)
	}, nil
}

func validateGeneration(ctx context.Context, store *overgodb.Store, reporter operation.Reporter, path string, capability mediacapability.Capability, execution modelrecipe.CapabilityEvidenceSelection, revision, input string) (operation.Completion, error) {
	definition := execution.Program.Definition()
	var extent struct{ Width, Height int }
	if err := json.Unmarshal([]byte(input), &extent); err != nil || extent.Width <= 0 || extent.Height <= 0 {
		return operation.Completion{}, errors.New("library: validation request has no positive image extent")
	}
	stages := []string{"generate", "repeat", "publish", "activate"}
	total := uint64(len(stages))
	started := time.Now()
	var first artifact.Content
	var firstInput artifact.ID
	var measured capabilityruntime.Measured
	device, backend := runrecord.RecipeDevice(definition)
	fail := func(cause error) (operation.Completion, error) {
		verification, err := modelintake.PublishFailure(context.WithoutCancel(ctx), store, definition, revision, time.Since(started), device, backend, "functional generation validation failed: "+cause.Error(), "generation-validation-failed")
		return operation.Completion{Run: verification.Run}, errors.Join(cause, err)
	}
	for pass, stage := range stages[:generationValidationRuns] {
		reporter.Text("Checking image generation: " + stage)
		reporter.Progress(uint64(pass), &total)
		output, err := capability.Execute(capabilityruntime.FreshRun(ctx), store, path, execution, input)
		if err != nil {
			return fail(err)
		}
		var ok bool
		measured, ok = output.(capabilityruntime.Measured)
		if !ok || !measured.Input.Descriptor.ID.Valid() || !measured.Run.Valid() {
			return fail(errors.New("library: generation validation lacks measured input or run"))
		}
		content, err := mediacapability.OutputContent(measured.Output)
		if err != nil {
			return fail(err)
		}
		run, err := runrecord.RequireExactRun(ctx, store, measured.Run)
		if err != nil || run.Recipe != definition.ID || run.Outcome != runrecord.OutcomeSucceeded || !slices.Contains(run.Inputs, measured.Input.Descriptor.ID) || !slices.Contains(run.Outputs, content.Descriptor.ID) {
			return fail(errors.Join(errors.New("library: image validation run does not bind the candidate, request and output"), err))
		}
		decoded, format, err := image.Decode(bytes.NewReader(content.Data))
		if err != nil || format != "png" {
			return fail(errors.Join(fmt.Errorf("library: generation output format %q is not a decoded PNG", format), err))
		}
		if decoded.Bounds().Dx() != extent.Width || decoded.Bounds().Dy() != extent.Height {
			return fail(errors.New("library: decoded image extent differs from the validation request"))
		}
		stored, present, err := artifact.ReadContent(ctx, store, content.Descriptor.ID)
		if err != nil || !present || !bytes.Equal(stored.Data, content.Data) {
			return fail(errors.New("library: generated image was not durably published"))
		}
		if pass == 0 {
			first, firstInput = content, measured.Input.Descriptor.ID
		} else if content.Descriptor.ID != first.Descriptor.ID || measured.Input.Descriptor.ID != firstInput {
			return fail(errors.New("library: fresh generation repeat changed input or image identity"))
		}
	}
	if err := ctx.Err(); err != nil {
		return fail(err)
	}
	reporter.Progress(generationValidationRuns, &total)
	verification, err := modelintake.PublishMeasuredVerification(ctx, store, definition, revision, time.Since(started), device, backend, "functional image validation: two fresh requests decode and retain identical PNG output; model quality is evaluated separately", measured)
	if err != nil {
		return operation.Completion{}, err
	}
	reporter.Progress(total-1, &total)
	if err = modelrecipe.ActivateCapability(ctx, store, definition, verification, recipe.EvidenceVerified, "library functional image generation and repeat validation"); err != nil {
		return operation.Completion{}, err
	}
	reporter.Publishing()
	reporter.Progress(total, &total)
	return operation.Completion{Run: verification.Run, Outputs: []artifact.ID{verification.Gate, first.Descriptor.ID}}, nil
}
