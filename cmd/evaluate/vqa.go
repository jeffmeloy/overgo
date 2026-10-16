package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/sequencescore"
	"overgo/internal/tokenizer"
	"overgo/internal/vqaserve"
)

const (
	// frameSampleFPS and frameSampleCount cut a clip into the few frames the
	// one a suite shows is taken from: its middle.
	frameSampleFPS   = 1.0
	frameSampleCount = 9
	// frameFileMode and frameDirectoryMode are the permissions of a frame file
	// and of the directory written for a suite.
	frameFileMode      = 0o600
	frameDirectoryMode = 0o700
	// clipFileExtension is a clip's extension in a clip directory.
	clipFileExtension = ".mp4"
	// frameHalf divides a clip's frames to the middle one.
	frameHalf = 2
)

// vqaRuntime is a visual-question-answering model as an evaluation runtime: a
// case whose prompt names an image of the media root is asked of the model
// through the recipe's two stages, and answers generatively. The pipeline
// renders the question into its own prompt, so the prompt is not shaped.
type vqaRuntime struct {
	store    *overgodb.Store
	modelID  artifact.ID
	modelDir string
	program  recipe.Program
	*mediaFiles
	producer artifact.ID
	// attempt names this evaluation run: the workflow stores each answer under
	// its key, so a key that did not change between runs would replay the last
	// run's answers instead of asking the model again.
	attempt string
}

// ShapeChatPrompt returns the prompt itself: the pipeline shapes it.
func (*vqaRuntime) ShapeChatPrompt(prompt string) (string, error) { return prompt, nil }

// ScoreContinuations refuses: a VQA evaluation scores generated answers.
func (*vqaRuntime) ScoreContinuations(context.Context, string, []string) ([]sequencescore.Score, error) {
	return nil, errors.New("evaluate: a vqa evaluation scores generated answers, not continuation likelihoods")
}

// Generate asks the question of the image the prompt names, for as many steps
// as the options allow, and reports the answer as the one piece it wrote.
func (r *vqaRuntime) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	kind, name, question, named := evaluation.ParseMediaPrompt(prompt)
	if !named || kind != evaluation.MediaImage {
		return nil, "", errors.New("evaluate: a vqa case names the image its question asks about")
	}
	raw, err := r.mediaFiles.read(name)
	if err != nil {
		return nil, "", err
	}
	image, err := vqaserve.ImageContract.ContentBytes(raw)
	if err != nil {
		return nil, "", err
	}
	key, err := vqaRunKey(r.attempt, image.Descriptor.ID.String(), question, options.MaxNewTokens)
	if err != nil {
		return nil, "", err
	}
	text, result, err := vqaserve.RunProgram(ctx, r.store, r.modelID, r.program, key, image, question,
		func(image []byte, question string) (vqaserve.Prepared, error) {
			prepared, err := vqaserve.Prepare(r.modelDir, image, question)
			if err == nil && options.OnPromptEvaluated != nil {
				options.OnPromptEvaluated(inference.PromptEvaluation{Tokens: len(prepared.InputIDs)})
			}
			return prepared, err
		},
		func(ctx context.Context, prepared vqaserve.Prepared) (string, error) {
			_, answer, err := vqaserve.Answer(ctx, r.modelDir, prepared, options.MaxNewTokens, func(string) {})
			return answer, err
		})
	if err != nil {
		return nil, "", err
	}
	r.producer = result.Run.ID
	if options.OnToken != nil {
		if err := options.OnToken(inference.TokenEvent{Piece: text}); err != nil {
			return nil, "", err
		}
	}
	return nil, text, nil
}

// ProducingRun names the persisted workflow that supplied the last answer.
func (r *vqaRuntime) ProducingRun() artifact.ID { return r.producer }

// vqaRunKey names the workflow run of one question about one image in one
// evaluation attempt, under one decode budget: the run's stored answer is
// replayed for the same key, so each of these changes it.
func vqaRunKey(attempt, content, question string, steps int) (string, error) {
	key, err := artifact.JSONID(artifact.KindProfile, struct {
		Attempt, Content, Question string
		Steps                      int
	}{attempt, content, question, steps})
	if err != nil {
		return "", err
	}
	return "evaluation/vqa/" + key.DigestHex(), nil
}

// openVQASession opens the visual-question-answering model at the request's
// path under its active vqa recipe; its cases ask about images of the media root.
func openVQASession(ctx context.Context, store *overgodb.Store, value manifest, request modelRequest) (evaluationSession, error) {
	source, err := mediacapability.Catalog[recipe.TaskVQA].Resolve(request.Path)
	if err != nil {
		return nil, fmt.Errorf("evaluate: %s is no vqa model: %w", request.Path, err)
	}
	modelID := source.Inventory.Manifest.ID
	return openCapabilitySession(ctx, store, value, source, recipe.TaskVQA, nil, func(recipe.Definition) (evaluation.Runtime, func(context.Context) error, error) {
		_, program, err := modelrecipe.ResolveActiveCapability(ctx, store, modelID, recipe.TaskVQA)
		return &vqaRuntime{store: store, modelID: modelID, modelDir: request.Path, program: program, mediaFiles: &mediaFiles{root: request.MediaRoot}, attempt: time.Now().UTC().Format(time.RFC3339Nano)},
			func(context.Context) error { return nil }, err
	})
}

// writeFrames cuts, for each case of a frame description-matching suite, the
// middle frame of the clip its prompt names and writes it to directory under
// the name the prompt gives it.
func writeFrames(ctx context.Context, suite evaluation.DescriptionMatchingSuite, clips, directory, ffmpeg string) error {
	if err := os.MkdirAll(directory, frameDirectoryMode); err != nil {
		return err
	}
	for _, testCase := range suite.Cases {
		_, name, _, named := evaluation.ParseMediaPrompt(testCase.Prompt)
		if !named {
			return fmt.Errorf("evaluate: case %s names no frame to cut", testCase.Name)
		}
		clip := filepath.Join(clips, strings.TrimSuffix(name, filepath.Ext(name))+clipFileExtension)
		frame, err := cutFrame(ctx, clip, ffmpeg)
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(directory, name), frame, frameFileMode); err != nil {
			return err
		}
	}
	return nil
}
