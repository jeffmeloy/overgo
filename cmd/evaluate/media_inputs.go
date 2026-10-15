package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
)

type boundMedia struct {
	path  string
	input evaluation.MediaInput
}

// mediaFiles resolves locators before evidence lookup and rechecks the consumed snapshot.
type mediaFiles struct {
	root  string
	files map[string]boundMedia
}

func (files *mediaFiles) bind(path string, input evaluation.MediaInput) (evaluation.MediaInput, error) {
	if !input.Content.Valid() {
		id, err := artifact.IdentifyFile(artifact.KindFile, path)
		if err != nil {
			return evaluation.MediaInput{}, err
		}
		input.Content = id
	}
	reference, err := input.Reference()
	if err != nil {
		return evaluation.MediaInput{}, err
	}
	if files.files == nil {
		files.files = map[string]boundMedia{}
	}
	files.files[reference] = boundMedia{path, input}
	return input, nil
}

func (files *mediaFiles) read(reference string) ([]byte, error) {
	if files == nil {
		return nil, errors.New("evaluate: media input was not bound")
	}
	bound, found := files.files[reference]
	if !found {
		return nil, errors.New("evaluate: media input was not bound")
	}
	raw, err := os.ReadFile(bound.path)
	if err != nil {
		return nil, err
	}
	id, err := artifact.IdentifyBytes(bound.input.Content.Kind(), raw)
	if err != nil || id != bound.input.Content {
		return nil, errors.Join(err, errors.New("evaluate: media bytes changed after binding"))
	}
	return raw, nil
}

// ResolveMedia binds fixed media to its bytes, decoder and admitted answering runtime.
func (r projectionRuntime) ResolveMedia(ctx context.Context, kind evaluation.MediaKind, name string) (evaluation.MediaInput, error) {
	if r.mediaFiles == nil {
		return evaluation.MediaInput{}, errors.New("evaluate: media root is absent")
	}
	path, err := mediaPath(r.root, name)
	if err != nil {
		return evaluation.MediaInput{}, err
	}
	return r.bindMedia(ctx, kind, path, evaluation.MediaInput{})
}

func (r projectionRuntime) bindMedia(ctx context.Context, kind evaluation.MediaKind, path string, input evaluation.MediaInput) (evaluation.MediaInput, error) {
	if err := ctx.Err(); err != nil {
		return evaluation.MediaInput{}, err
	}
	input.Runtime = r.identity
	input.Decode = evaluation.MediaDecode{Format: "image"}
	if kind == evaluation.MediaVideo {
		video := r.runner.RuntimePolicy().Interactive.Video
		input.Decode = evaluation.MediaDecode{Format: "gif", FPS: video.FPS, Frames: video.MaxFrames}
		file, err := os.Open(path)
		if err != nil {
			return evaluation.MediaInput{}, err
		}
		var header [len("GIF87a")]byte
		_, readErr := io.ReadFull(file, header[:])
		if err := errors.Join(readErr, file.Close()); err != nil {
			return evaluation.MediaInput{}, err
		}
		if string(header[:]) != "GIF87a" && string(header[:]) != "GIF89a" {
			tool, err := media.ResolveFFmpeg(r.ffmpeg)
			if err != nil {
				return evaluation.MediaInput{}, err
			}
			input.Decode.Format = "ffmpeg"
			input.Decode.Tool, err = artifact.IdentifyFile(artifact.KindFile, tool)
			if err != nil {
				return evaluation.MediaInput{}, err
			}
		}
	}
	return r.mediaFiles.bind(path, input)
}

// ResolveMedia retains a generated payload's exact request and original producing run.
func (r judgedRuntime) ResolveMedia(ctx context.Context, kind evaluation.MediaKind, name string) (evaluation.MediaInput, error) {
	held, err := verifyRendered(ctx, r.store, r.root, r.model, r.recipe, name)
	if err != nil {
		return evaluation.MediaInput{}, err
	}
	return r.bindMedia(ctx, kind, filepath.Join(r.root, held.File), evaluation.MediaInput{Content: held.Output, Request: held.Input, Producer: held.Run})
}

// ResolveMedia binds image content to the admitted VQA workflow.
func (r *vqaRuntime) ResolveMedia(ctx context.Context, kind evaluation.MediaKind, name string) (evaluation.MediaInput, error) {
	if r.mediaFiles == nil {
		return evaluation.MediaInput{}, errors.New("evaluate: media root is absent")
	}
	if err := ctx.Err(); err != nil {
		return evaluation.MediaInput{}, err
	}
	if kind != evaluation.MediaImage {
		return evaluation.MediaInput{}, fmt.Errorf("evaluate: VQA requires image, got %s", kind)
	}
	path, err := mediaPath(r.root, name)
	if err != nil {
		return evaluation.MediaInput{}, err
	}
	return r.mediaFiles.bind(path, evaluation.MediaInput{Runtime: evaluation.MediaRuntime{Model: r.modelID, Recipe: r.program.Definition().ID}, Decode: evaluation.MediaDecode{Format: "image"}})
}

// projectionIdentity reuses the recipes already admitted by the runner and projector.
func projectionIdentity(ctx context.Context, store artifact.Reader, r projectionRuntime) (evaluation.MediaRuntime, error) {
	description, err := r.runner.RecipeRuntimeDescription(recipe.TaskInference)
	if err != nil {
		return evaluation.MediaRuntime{}, err
	}
	projection, found, err := modelrecipe.ActiveRecord(ctx, store, r.runner.ModelID(), recipe.TaskProjection)
	if err != nil || !found {
		return evaluation.MediaRuntime{}, errors.Join(err, errors.New("evaluate: admitted projection recipe is absent"))
	}
	return evaluation.MediaRuntime{Model: r.runner.ModelID(), Recipe: description.Identity.Recipe, Projection: projection.Definition.ID}, nil
}
