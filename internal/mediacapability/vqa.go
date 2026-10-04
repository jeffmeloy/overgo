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
// is the output; the producer run and node walls travel in the measured
// envelope, with the wall of the pipeline's decode loop, which the pipeline
// times apart from its vision tower and prefill, and the tokens that loop
// gave: the answer's but its first, which is the prefill's. A generate stage
// the store recovered ran no pipeline and reports neither.
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
	var decoded vqaserve.Result
	answer, result, err := vqaserve.RunProgram(ctx, store, definition.Model, program, capabilityruntime.RunKey(definition, request), image, form.Question,
		func(image []byte, question string) (vqaserve.Prepared, error) {
			return vqaserve.Prepare(path, image, question)
		},
		func(ctx context.Context, prepared vqaserve.Prepared) (text string, err error) {
			decoded, text, err = vqaserve.Answer(ctx, path, prepared, form.Budget(), nil)
			return text, err
		})
	measured := capabilityruntime.Measured{Output: answer, Run: result.Run.ID, Phases: result.NodeWalls, Input: request}
	if err == nil {
		measured.OutputTokens, measured.Decode = uint64(decoded.Steps), decoded.DecodeWall
	}
	return measured, err
}
