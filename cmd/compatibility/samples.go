package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image/gif"
	"image/png"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/jsonabbrev"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// mediaSamplesDir holds the exported generation outputs the media
// report links: every file is store content named by its SHA-256, so a
// sample is checkable against the evidence that produced it.
const mediaSamplesDir = "docs/media_samples"

// speechAudioSchema recognizes the speech pipeline's typed audio
// output so it can be exported as a playable WAV.
const speechAudioSchema = "overgo.speech-audio.v1"

// generatedVideoSchema recognizes the JSON-enveloped generated-video
// output whose data field owns the encoded clip bytes.
const generatedVideoSchema = "overgo.generated-video.v1"

// exportMediaSamples writes each healthy media activation's succeeded
// generation outputs under docs/media_samples as decodable files --
// PNG and GIF bytes exactly as the store holds them, speech audio
// re-encoded as 16-bit PCM WAV -- named by content SHA-256. Nothing is
// exported that a run did not publish.
func exportMediaSamples(root, repository string, scope mediaReportScope) error {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	entries, truncated, err := discovery.CapabilityCatalogForTasks(ctx, store, mediaCatalogLimit, discovery.LoadMemo(ctx, store), scope.Tasks...)
	if err != nil {
		return err
	}
	if truncated {
		return errors.New("media sample catalog is truncated")
	}
	selection, err := loadMediaSampleSelection(ctx, root, store, scope, entries)
	if err != nil {
		return err
	}
	directory := filepath.Join(root, mediaSamplesDir)
	if err := os.MkdirAll(directory, 0o755); err != nil {
		return err
	}
	written := 0
	for _, entry := range entries {
		for _, capability := range entry.Capabilities {
			if !mediaTask(capability.Task) || capability.Stale != "" {
				continue
			}
			activation, active, err := modelrecipe.ActiveRecord(ctx, store, entry.Model, capability.Task)
			if err != nil || !active {
				continue
			}
			count, err := exportRecipeSamples(ctx, store, activation.Definition.ID, directory, selection)
			if err != nil {
				return err
			}
			written += count
		}
	}
	fmt.Printf("exported %d media samples to %s\n", written, mediaSamplesDir)
	return nil
}

// exportRecipeSamples walks the runs recorded against one recipe
// definition and writes every decodable succeeded-run output.
func exportRecipeSamples(
	ctx context.Context,
	store *overgodb.Store,
	definition artifact.ID,
	directory string,
	selection mediaSampleSelection,
) (int, error) {
	written := 0
	err := visitRecipeSamples(ctx, store, definition, selection, func(sample sampleRef, data []byte) error {
		if err := validateMediaSampleContent(sample.Name, data); err != nil {
			return err
		}
		path := filepath.Join(directory, sample.Name)
		if _, statErr := os.Stat(path); statErr == nil {
			return validateMediaSampleFile(directory, sample.Name)
		}
		if err := os.WriteFile(path, data, 0o644); err != nil {
			return err
		}
		written++
		return nil
	})
	return written, err
}

// sampleRef pairs one exported sample with the abbreviated request
// that produced it, so the report shows the prompt and settings beside
// the artifact.
type sampleRef struct {
	Name                     string
	Request                  string
	Run, Output, Environment artifact.ID
	Commit                   string
	Inputs                   []artifact.ID
}

// recipeSampleNames lists the exportable samples one recipe
// definition's succeeded runs produced, in lineage order.
func recipeSampleNames(ctx context.Context, store *overgodb.Store, definition artifact.ID, selection mediaSampleSelection) ([]sampleRef, error) {
	var names []sampleRef
	seen := map[string]bool{}
	err := visitRecipeSamples(ctx, store, definition, selection, func(sample sampleRef, _ []byte) error {
		if !seen[sample.Name] {
			seen[sample.Name] = true
			names = append(names, sample)
		}
		return nil
	})
	return names, err
}

// visitRecipeSamples walks the runs recorded against one recipe
// definition and visits every decodable succeeded-run output.
func visitRecipeSamples(
	ctx context.Context,
	store *overgodb.Store,
	definition artifact.ID,
	selection mediaSampleSelection,
	visit func(sample sampleRef, data []byte) error,
) error {
	return visitRecipeRuns(ctx, store, definition, func(run runrecord.Run) error {
		if run.Outcome != runrecord.OutcomeSucceeded || !selection.includes(run) {
			return nil
		}
		request := runRequest(ctx, store, run.Inputs)
		for _, output := range run.Outputs {
			content, found, err := artifact.ReadContent(ctx, store, output)
			if err != nil {
				return err
			}
			if !found {
				return fmt.Errorf("media run %s output %s is unavailable", run.ID, output)
			}
			name, data, ok := sampleFile(content)
			if !ok {
				if content.Descriptor.Schema == generatedVideoSchema || content.Descriptor.Schema == speechAudioSchema {
					return fmt.Errorf("media run %s output %s has an invalid media envelope", run.ID, output)
				}
				continue
			}
			if err := visit(sampleRef{Name: name, Request: request, Run: run.ID, Output: output, Environment: run.Environment, Commit: run.CodeCommit, Inputs: run.Inputs}, data); err != nil {
				return err
			}
		}
		return nil
	})
}

// visitRecipeRuns supplies the same typed lineage to sample export and outcome
// reporting. Failed and cancelled records remain available to the latter.
func visitRecipeRuns(ctx context.Context, store *overgodb.Store, definition artifact.ID, visit func(runrecord.Run) error) error {
	edges, err := store.Children(ctx, definition)
	if err != nil {
		return err
	}
	for _, edge := range edges {
		if edge.Relation != artifact.RelationDependsOn || edge.Child.Kind() != artifact.KindRun {
			continue
		}
		run, err := runrecord.RequireRun(ctx, store, edge.Child)
		if err != nil {
			return err
		}
		if run.Recipe != definition {
			return fmt.Errorf("media run %s recipe lineage differs", run.ID)
		}
		if err := visit(run); err != nil {
			return err
		}
	}
	return nil
}

func validateMediaSampleFile(directory, name string) error {
	if filepath.Base(name) != name {
		return errors.New("invalid sample filename")
	}
	data, err := os.ReadFile(filepath.Join(directory, name))
	if err != nil {
		if pathErr, ok := errors.AsType[*os.PathError](err); ok {
			return fmt.Errorf("%s: %w", name, pathErr.Err)
		}
		return err
	}
	return validateMediaSampleContent(name, data)
}

func validateMediaSampleContent(name string, data []byte) error {
	var err error
	digest := sha256.Sum256(data)
	if hex.EncodeToString(digest[:]) != strings.TrimSuffix(name, filepath.Ext(name)) {
		return errors.New("sample bytes differ from their SHA-256 filename")
	}
	switch filepath.Ext(name) {
	case ".png":
		_, err = png.Decode(bytes.NewReader(data))
	case ".gif":
		_, err = gif.DecodeAll(bytes.NewReader(data))
	case ".wav":
		// WAV samples retain the existing exporter and content-hash check.
	default:
		return errors.New("unsupported sample format")
	}
	return err
}

// sampleFile decodes one run-output content into an exportable file:
// the bytes and a SHA-256 name with the extension its media type owns.
func sampleFile(content artifact.Content) (string, []byte, bool) {
	switch {
	case content.Descriptor.MediaType == media.PNGMediaType:
		return hashName(content.Data, "png"), content.Data, true
	case content.Descriptor.MediaType == media.GIFMediaType:
		return hashName(content.Data, "gif"), content.Data, true
	case content.Descriptor.Schema == generatedVideoSchema:
		var video struct {
			Data      []byte `json:"data"`
			MediaType string `json:"media_type"`
		}
		if err := json.Unmarshal(content.Data, &video); err != nil ||
			video.MediaType != media.GIFMediaType || len(video.Data) == 0 {
			return "", nil, false
		}
		return hashName(video.Data, "gif"), video.Data, true
	case content.Descriptor.Schema == speechAudioSchema:
		var audio struct {
			PCM        []float32 `json:"pcm"`
			SampleRate int       `json:"sample_rate"`
			Channels   int       `json:"channels"`
		}
		if err := json.Unmarshal(content.Data, &audio); err != nil || len(audio.PCM) == 0 {
			return "", nil, false
		}
		encoded, err := media.EncodeWAVPCM16(audio.PCM, audio.SampleRate)
		if err != nil {
			return "", nil, false
		}
		return hashName(encoded, "wav"), encoded, true
	}
	return "", nil, false
}

func hashName(data []byte, extension string) string {
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:]) + "." + extension
}

// runRequest reads the run's typed input document and renders it with
// bulk fields elided: the prompt and the settings stay readable, the
// embedded tensors and payloads state their extent instead of their
// values.
func runRequest(ctx context.Context, store *overgodb.Store, inputs []artifact.ID) string {
	for _, input := range inputs {
		content, found, err := artifact.ReadContent(ctx, store, input)
		if err != nil || !found || content.Descriptor.MediaType != "application/json" {
			continue
		}
		var document any
		if err := json.Unmarshal(content.Data, &document); err != nil {
			continue
		}
		abbreviated, err := json.Marshal(jsonabbrev.Value(document))
		if err != nil {
			continue
		}
		return string(abbreviated)
	}
	return ""
}
