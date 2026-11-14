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
// arm; it prints the verdict, which is the preset's admission or refusal. Two renders that are not one request but the
// preset are refused before any record is read (comparableArms), and a record that judged media its render did not
// draw, or by another judge or scorer than the other arm's, before any verdict (judgedRender).
func admitPresetReading(ctx context.Context, repository, modelPath, referenceDirectory, presetDirectory, referenceRecord, presetRecord string, said io.Writer) error {
	store, err := overgodb.OpenContext(ctx, repository)
	if err != nil {
		return err
	}
	defer store.Close()
	// read is one arm of the reading: the directory its media were drawn into and the record of their judged answers.
	type read struct {
		directory, record string
		drawn             renderManifest
		arm               evaluation.PresetArm
		right             map[string]bool
	}
	reference, preset := &read{directory: referenceDirectory, record: referenceRecord}, &read{directory: presetDirectory, record: presetRecord}
	for _, arm := range []*read{reference, preset} {
		if arm.drawn, err = readRender(arm.directory); err != nil {
			return err
		}
		if arm.arm, err = drawnArm(arm.drawn); err != nil {
			return err
		}
	}
	if err := comparableArms(reference.drawn, preset.drawn); err != nil {
		return err
	}
	var judge judging
	for _, arm := range []*read{reference, preset} {
		record, err := artifact.ParseID(arm.record)
		if err != nil {
			return err
		}
		inputs, scorer, err := evaluation.JudgedMedia(ctx, store, record)
		if err != nil {
			return err
		}
		if err := judgedRender(record, inputs, scorer, arm.drawn, &judge); err != nil {
			return err
		}
		outcomes, found := evaluation.CaseOutcomes(ctx, store, record)
		if !found {
			return fmt.Errorf("evaluate: no case outcomes can be read from evaluation %s", record)
		}
		arm.right = outcomes
	}
	selected, err := modelrecipe.ResolveActiveExecution(ctx, store, preset.drawn.Model, preset.drawn.Task, modelrecipe.SessionWarm)
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

// comparableArms refuses a reading of two renders that are not one request but the preset (SQA-379): the arms are of one
// model, task and recipe, and the requests the preset's arm drew, each without its preset, are the requests its
// reference drew, so their prompts and seeds pair and nothing else of them differs.
func comparableArms(reference, preset renderManifest) error {
	if reference.Model != preset.Model || reference.Task != preset.Task || reference.Recipe != preset.Recipe {
		return fmt.Errorf("evaluate: the preset's render is of model %s, task %s and recipe %s, its reference's of %s, %s and %s: no one reading",
			preset.Model, preset.Task, preset.Recipe, reference.Model, reference.Task, reference.Recipe)
	}
	// unpreset is the requests a render drew, each without its preset, as JSON with its keys in order.
	unpreset := func(drawn renderManifest) (map[string]bool, error) {
		requests := map[string]bool{}
		for key, media := range drawn.Draws {
			var request map[string]any
			if err := json.Unmarshal(media.Request, &request); err != nil {
				return nil, fmt.Errorf("evaluate: the request of draw %s: %w", key, err)
			}
			delete(request, "preset")
			encoded, err := json.Marshal(request)
			if err != nil {
				return nil, err
			}
			requests[string(encoded)] = true
		}
		return requests, nil
	}
	asked, err := unpreset(reference)
	if err != nil {
		return err
	}
	guided, err := unpreset(preset)
	if err != nil {
		return err
	}
	for request := range guided {
		if !asked[request] {
			return fmt.Errorf("evaluate: the preset's render drew %s, which its reference did not draw, the preset aside: no one reading", request)
		}
	}
	for request := range asked {
		if !guided[request] {
			return fmt.Errorf("evaluate: the reference drew %s, which the preset's render did not draw: no one reading", request)
		}
	}
	return nil
}

// judging is who judged a reading's media: the runtime that answered and the scorer that scored.
type judging struct {
	runtime evaluation.MediaRuntime
	scorer  artifact.ID
}

// judgedRender refuses an evaluation record that judged media its render did not draw, or by another judge or scorer
// than the reading's others (SQA-379): each media input its plan bound (evaluation.JudgedMedia) is one of the render's
// draws, the content it wrote, the request it was given and the run that drew it, and every input was answered by one
// runtime and scored by one scorer, which judge holds once the first arm is read.
func judgedRender(record artifact.ID, inputs []evaluation.MediaInput, scorer artifact.ID, drawn renderManifest, judge *judging) error {
	for _, input := range inputs {
		if !slices.ContainsFunc(slices.Collect(maps.Values(drawn.Draws)), func(media renderedMedia) bool {
			return media.Output == input.Content && media.Input == input.Request && media.Run == input.Producer
		}) {
			return fmt.Errorf("evaluate: evaluation %s judged %s, which its render did not draw", record, input.Content)
		}
		read := judging{runtime: input.Runtime, scorer: scorer}
		if *judge == (judging{}) {
			*judge = read
		}
		if read != *judge {
			return fmt.Errorf("evaluate: evaluation %s was judged by %+v, the reading's other judgments by %+v", record, read, *judge)
		}
	}
	return nil
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
