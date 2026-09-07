package mediacapability

import (
	"context"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/strictjson"
	"overgo/internal/vqaserve"
)

// vqaCapability answers a question about a stored image: the model
// directory resolves as a Hugging Face inventory into the vqa recipe, and
// the executor runs the recipe's prepare and generate stages over the
// processor and the device pipeline the parity harness verifies, so the
// page serves the activation the harness proved.
func vqaCapability() Capability {
	return Capability{
		Resolve: func(path string) (Source, error) {
			inventory, err := modelartifact.FromHFPath(path)
			return definitionSource(inventory, err, func(modelID artifact.ID) (recipe.Definition, error) {
				return modelrecipe.CapabilityDefinition(recipe.TaskVQA, modelID)
			})
		},
		Execute: capabilityruntime.ExecutorCatalog{modelrecipe.ModuleVQAPrepare: executeVQA}.Execute,
	}
}

// executeVQA decodes the page request, reads the image from the store and
// runs the two-stage program: prepare over the processor, generate over
// the device pipeline under the request's decode budget. The answer text
// is the output; the per-node walls travel in the measured envelope.
func executeVQA(ctx context.Context, store artifact.Repository, path string, selection modelrecipe.CapabilityEvidenceSelection, raw string) (any, error) {
	var form vqaserve.Request
	if err := strictjson.DecodeBytes([]byte(raw), &form); err != nil {
		return nil, fmt.Errorf("decode vqa input: %w", err)
	}
	if err := vqaserve.ValidateRequest(form); err != nil {
		return nil, err
	}
	request, err := artifact.JSONContent(vqaserve.RequestContract, form)
	if err != nil {
		return nil, err
	}
	image, err := vqaserve.ReadImage(ctx, store, form.Image)
	if err != nil {
		return nil, err
	}
	program := selection.Program
	definition := program.Definition()
	answer, walls, err := vqaserve.RunProgram(ctx, store, definition.Model, program, capabilityruntime.RunKey(definition, request), image, form.Question,
		func(image []byte, question string) (vqaserve.Prepared, error) {
			return vqaserve.Prepare(path, image, question)
		},
		func(ctx context.Context, prepared vqaserve.Prepared) (string, error) {
			_, text, err := vqaserve.Answer(ctx, path, prepared, form.Budget(), nil)
			return text, err
		})
	if err != nil {
		return nil, err
	}
	return capabilityruntime.Measured{Output: answer, Phases: walls, Input: request}, nil
}
