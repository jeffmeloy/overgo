package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
)

// personSettings are the fields of a request mediacapability.PersonRequest sets to the size a person asks for.
var personSettings = []string{"frames", "width", "height", "steps"}

// personSized is the template at the size a person asks for, the page's own defaults (mediacapability.PersonRequest),
// unless -render-set names one of its fields: a preset's evidence is a suite drawn at a person's size
// (judge-calibration/user-size), where a verification smoke ran at a size of its own.
func personSized(selected modelrecipe.CapabilityEvidenceSelection, directory string, template map[string]any, settings map[string]float64) (map[string]any, error) {
	stages := selected.Program.Stages()
	if len(stages) == 0 || slices.ContainsFunc(personSettings, func(name string) bool { _, named := settings[name]; return named }) {
		return template, nil
	}
	encoded, err := json.Marshal(template)
	if err != nil {
		return nil, err
	}
	sized, lifted, err := mediacapability.PersonRequest(stages[0].Module.ID, directory, string(encoded))
	if err != nil || !lifted {
		return template, err
	}
	var person map[string]any
	if err := json.Unmarshal([]byte(sized), &person); err != nil {
		return nil, err
	}
	return person, nil
}

// admitPresetReading reads a preset's judged suite against its model's reference (judge-calibration/user-size): the
// arms are read from the two render directories' manifests, the size a person asks for from the model's page through
// its active recipe's verified request (personSized), and each case's outcome from the judged evaluation record of each
// arm; it prints the verdict, which is the preset's admission or refusal.
func admitPresetReading(ctx context.Context, repository, modelPath, referenceDirectory, presetDirectory, referenceRecord, presetRecord string, said io.Writer) error {
	store, err := overgodb.OpenContext(ctx, repository)
	if err != nil {
		return err
	}
	defer store.Close()
	// read is one arm of the reading: the directory its media were drawn into and the record of their judged answers.
	type read struct {
		directory, record string
		arm               evaluation.PresetArm
		right             map[string]bool
	}
	reference, preset := &read{directory: referenceDirectory, record: referenceRecord}, &read{directory: presetDirectory, record: presetRecord}
	var drawn renderManifest
	for _, arm := range []*read{reference, preset} {
		if drawn, err = readRender(arm.directory); err != nil {
			return err
		}
		if arm.arm, err = drawnArm(drawn); err != nil {
			return err
		}
		record, err := artifact.ParseID(arm.record)
		if err != nil {
			return err
		}
		outcomes, found := evaluation.CaseOutcomes(ctx, store, record)
		if !found {
			return fmt.Errorf("evaluate: no case outcomes can be read from evaluation %s", record)
		}
		arm.right = outcomes
	}
	selected, err := modelrecipe.ResolveActiveExecution(ctx, store, drawn.Model, drawn.Task, modelrecipe.SessionWarm)
	if err != nil {
		return err
	}
	template, err := verifiedTemplate(ctx, store, selected.Activation)
	if err != nil {
		return err
	}
	if template, err = personSized(selected, modelPath, template, nil); err != nil {
		return err
	}
	person := map[string]float64{}
	for _, name := range personSettings {
		if value, stated := template[name].(float64); stated {
			person[name] = value
		}
	}
	verdict, err := evaluation.AdmitPreset(person, reference.arm, preset.arm, reference.right, preset.right)
	if err != nil {
		return err
	}
	encoded, err := json.MarshalIndent(verdict, "", "  ")
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(said, "%s\n", encoded)
	return err
}

// readRender is the manifest of the media a render drew into directory.
func readRender(directory string) (renderManifest, error) {
	var manifest renderManifest
	data, err := os.ReadFile(filepath.Join(directory, renderManifestFile))
	if err != nil {
		return renderManifest{}, err
	}
	return manifest, json.Unmarshal(data, &manifest)
}

// drawnArm is the arm a render's media make in a preset's reading: its preset, the size its draws were drawn at, read
// from the requests drawn and not from the settings asked, and its format. Draws at two sizes are no arm.
func drawnArm(manifest renderManifest) (evaluation.PresetArm, error) {
	arm := evaluation.PresetArm{Preset: manifest.Preset, Format: manifest.clipFormat(), Settings: map[string]float64{}}
	for _, key := range slices.Sorted(maps.Keys(manifest.Draws)) {
		var request map[string]any
		if err := json.Unmarshal(manifest.Draws[key].Request, &request); err != nil {
			return evaluation.PresetArm{}, fmt.Errorf("evaluate: the request of draw %s: %w", key, err)
		}
		for _, name := range personSettings {
			value, stated := request[name].(float64)
			if !stated {
				continue
			}
			if held, seen := arm.Settings[name]; seen && held != value {
				return evaluation.PresetArm{}, fmt.Errorf("evaluate: the render's draws were drawn at %s %v and %v, which is no one arm", name, held, value)
			}
			arm.Settings[name] = value
		}
	}
	return arm, nil
}
