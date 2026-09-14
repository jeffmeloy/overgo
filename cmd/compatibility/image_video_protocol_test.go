package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"image/gif"
	"image/png"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// This identity freezes the criteria and retained requests before implementation
// changes. Whitespace does not alter identity; changing a judge or case does.
const imageVideoProtocolSHA256 = "9a270a1847acc8b478fb2b7595fb254f9d89d315bf0dfc23e6a0d97a9ceb3eb2"

func checkImageVideoProtocolIdentity(data []byte) error {
	return checkMediaProtocolIdentity(data, imageVideoProtocolSHA256)
}

func checkMediaProtocolIdentity(data []byte, digest string) error {
	var value any
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	canonical, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if fmt.Sprintf("%x", sha256.Sum256(canonical)) != digest {
		return errors.New("media protocol: frozen inputs, references, or criteria changed")
	}
	return nil
}

func checkImageVideoObservation(value imageVideoObservation, content artifact.Content) error {
	if value.Artifact != content.Descriptor.ID {
		return errors.New("media protocol: output identity differs")
	}
	name, data, ok := sampleFile(content)
	if !ok {
		return errors.New("media protocol: output cannot be exported")
	}
	width, height, frames, delay := 0, 0, 0, 0
	switch filepath.Ext(name) {
	case ".png":
		picture, err := png.Decode(bytes.NewReader(data))
		if err != nil {
			return err
		}
		width, height, frames = picture.Bounds().Dx(), picture.Bounds().Dy(), 1
		if value.FPS != 0 || value.Delay != 0 {
			return errors.New("media protocol: still image has video timing")
		}
	case ".gif":
		video, err := gif.DecodeAll(bytes.NewReader(data))
		if err != nil {
			return err
		}
		width, height, frames = video.Config.Width, video.Config.Height, len(video.Image)
		for _, ticks := range video.Delay {
			delay += ticks
		}
		if value.FPS <= 0 {
			return errors.New("media protocol: declared frame rate absent")
		}
	default:
		return errors.New("media protocol: non-image/video output")
	}
	if width != value.Width || height != value.Height || frames != value.Frames || delay != value.Delay {
		return errors.New("media protocol: decoded geometry or timing differs")
	}
	return nil
}

func TestImageVideoProtocolAcceptance(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration + ": media protocol reads retained private-store outputs")
	}
	root := testutil.RepoRoot(t)
	document, err := plan.Load(filepath.Join(root, plan.Path))
	if err != nil {
		t.Fatal(err)
	}
	if document.Lane != "image_video_gen" || os.Getenv(dataroot.Env) == "" {
		t.Skip("integration: image_video_gen protocol requires its explicit data root")
	}
	path := filepath.Join(root, "docs/image_video_protocol.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkImageVideoProtocolIdentity(data); err != nil {
		t.Fatal(err)
	}
	var protocol imageVideoProtocol
	if err := jsonfile.DecodeStrict(path, &protocol); err != nil {
		t.Fatal(err)
	}
	var inventory imageVideoInventory
	if err := jsonfile.DecodeStrict(filepath.Join(root, "docs/image_video_inventory.json"), &inventory); err != nil {
		t.Fatal(err)
	}
	if protocol.Version != artifact.InitialDocumentVersion || protocol.Census != inventory.Census {
		t.Fatal("protocol inventory identity differs")
	}
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	coverage := map[artifact.ID]int{}
	for _, cell := range inventory.Cells {
		coverage[cell.Recipe] = 0
	}
	seen := map[string]bool{}
	for _, value := range protocol.Cases {
		if seen[value.ID] {
			t.Fatalf("duplicate case %s", value.ID)
		}
		seen[value.ID] = true
		if _, ok := coverage[value.Recipe]; !ok {
			t.Fatalf("foreign recipe %s", value.Recipe)
		}
		cell := inventory.Cells[slices.IndexFunc(inventory.Cells, func(cell imageVideoInventoryCell) bool { return cell.Recipe == value.Recipe })]
		if value.Model != cell.Model || value.Task != cell.Task {
			t.Fatal("case model/task differs")
		}
		run, err := runrecord.RequireRun(t.Context(), store, value.Run)
		if err != nil {
			t.Fatal(err)
		}
		if err := checkImageVideoCaseRun(value, run); err != nil {
			t.Fatal(err)
		}
		for _, input := range value.Inputs {
			if _, found, err := artifact.ReadContent(t.Context(), store, input); err != nil || !found {
				t.Fatalf("input %s absent: %v", input, err)
			}
		}
		if len(value.Outputs) != len(value.Observations) {
			t.Fatal("incomplete output observations")
		}
		for i, observation := range value.Observations {
			if observation.Artifact != value.Outputs[i] {
				t.Fatal("observation output order differs")
			}
			content, found, err := artifact.ReadContent(t.Context(), store, observation.Artifact)
			if err != nil || !found {
				t.Fatalf("output absent: %v", err)
			}
			if err := checkImageVideoObservation(observation, content); err != nil {
				t.Fatal(err)
			}
			changed := observation
			changed.Width++
			if checkImageVideoObservation(changed, content) == nil {
				t.Fatal("wrong output dimensions accepted")
			}
			if observation.FPS > 0 {
				// GIF's time unit is one centisecond. This reports the historical
				// failure; protocol acceptance does not declare that clip correct.
				errorCS := float64(observation.Delay) - float64(observation.Frames)*100/float64(observation.FPS)
				t.Logf("%s: historical duration error %.4f centiseconds; required absolute error <= 0.5", value.ID, errorCS)
			}
		}
		changed := value
		changed.Inputs = []artifact.ID{value.Outputs[0]}
		if checkImageVideoCaseRun(changed, run) == nil {
			t.Fatal("substituted request accepted")
		}
		coverage[value.Recipe]++
	}
	for definition, count := range coverage {
		if count == 0 {
			t.Fatalf("required recipe %s has no review case", definition)
		}
	}
	for _, field := range []string{"cases", "required_checks", "review_questions", "rules", "inventory_census"} {
		t.Run("changed "+field, func(t *testing.T) {
			var changed map[string]any
			if err := json.Unmarshal(data, &changed); err != nil {
				t.Fatal(err)
			}
			delete(changed, field)
			encoded, err := json.Marshal(changed)
			if err != nil {
				t.Fatal(err)
			}
			if checkImageVideoProtocolIdentity(encoded) == nil {
				t.Fatal("changed frozen protocol accepted")
			}
		})
	}
	t.Logf("frozen review cases=%d, required recipe cells=%d; no model execution or performance acceptance", len(protocol.Cases), len(coverage))
}
