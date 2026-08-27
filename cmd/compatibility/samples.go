package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
)

// mediaSamplesDir holds the exported verifier-run outputs the media
// report links: every file is store content named by its SHA-256, so a
// sample is checkable against the evidence that produced it.
const mediaSamplesDir = "docs/media_samples"

// speechAudioSchema recognizes the speech pipeline's typed audio
// output so it can be exported as a playable WAV.
const speechAudioSchema = "overgo.speech-audio.v1"

// exportMediaSamples writes each healthy media activation's succeeded
// verifier-run outputs under docs/media_samples as decodable files --
// PNG and GIF bytes exactly as the store holds them, speech audio
// re-encoded as 16-bit PCM WAV -- named by content SHA-256. Nothing is
// exported that a run did not publish.
func exportMediaSamples(root, repository string) error {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	entries, _, err := discovery.CapabilityCatalog(ctx, store, mediaCatalogLimit, nil)
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
			count, err := exportRecipeSamples(ctx, store, activation.Definition.ID, directory)
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
) (int, error) {
	edges, err := store.Children(ctx, definition)
	if err != nil {
		return 0, err
	}
	written := 0
	for _, edge := range edges {
		if edge.Relation != artifact.RelationDependsOn || edge.Child.Kind() != artifact.KindRun {
			continue
		}
		run, err := runrecord.RequireRun(ctx, store, edge.Child)
		if err != nil || run.Outcome != runrecord.OutcomeSucceeded {
			continue
		}
		for _, output := range run.Outputs {
			content, found, err := artifact.ReadContent(ctx, store, output)
			if err != nil || !found {
				continue
			}
			name, data, ok := sampleFile(content)
			if !ok {
				continue
			}
			path := filepath.Join(directory, name)
			if _, statErr := os.Stat(path); statErr == nil {
				continue
			}
			if err := os.WriteFile(path, data, 0o644); err != nil {
				return written, err
			}
			written++
		}
	}
	return written, nil
}

// sampleFile decodes one run-output content into an exportable file:
// the bytes and a SHA-256 name with the extension its media type owns.
func sampleFile(content artifact.Content) (string, []byte, bool) {
	switch {
	case content.Descriptor.MediaType == media.PNGMediaType:
		return hashName(content.Data, "png"), content.Data, true
	case content.Descriptor.MediaType == media.GIFMediaType:
		return hashName(content.Data, "gif"), content.Data, true
	case content.Descriptor.Schema == speechAudioSchema:
		var audio struct {
			PCM        []float32 `json:"pcm"`
			SampleRate int       `json:"sample_rate"`
			Channels   int       `json:"channels"`
		}
		if err := strictjson.DecodeBytes(content.Data, &audio); err != nil || len(audio.PCM) == 0 {
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
