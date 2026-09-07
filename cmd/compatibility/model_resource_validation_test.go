package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

// Resource acceptance consumes the same serving observations used by the
// serving owner. A successful gate label or invented aggregate is insufficient.
func checkResourceValidationCell(ctx context.Context, store artifact.Reader, spec modelValidationSpecification, cell modelValidationCell) error {
	if cell.Plan.Valid() || cell.Dataset.Valid() || cell.Split.Valid() || len(cell.Shards) != 0 || len(cell.Cases) != 0 || len(cell.OracleSHA256) != sha256.Size*2 || strings.Trim(cell.OracleSHA256, "0123456789abcdef") != "" {
		return errors.New("model validation: media resource workload binding differs")
	}
	proof, err := runrecord.VerifyGateRun(ctx, store, cell.Recipe, cell.Evidence, cell.Run)
	if err != nil {
		return err
	}
	if proof.Gate.CodeCommit != spec.CodeCommit || proof.Gate.Environment != cell.Environment {
		return errors.New("model validation: media resource source or environment differs")
	}
	modes := []string{"text", "image", "audio", "multi-image", "video", "mixed"}
	mode := strings.TrimSuffix(strings.TrimPrefix(cell.Name, "resources/"), "/native")
	if mode == "mixed-history" {
		mode = "mixed"
	} // Native capture names this mixed-history request "mixed".
	if !slices.Contains(modes, mode) {
		return errors.New("model validation: foreign resource mode")
	}
	const reloads, cycles, actions = 3, 3, 5 // Executed recovery contract, not estimated counters.
	if len(proof.Gate.Steps) != 1+reloads+len(modes)*reloads*cycles*actions {
		return errors.New("model validation: incomplete media resource action denominator")
	}
	steps := make(map[string]runrecord.GateStep, len(proof.Gate.Steps))
	for _, step := range proof.Gate.Steps {
		if step.Outcome != runrecord.StepSucceeded || step.Phase != runrecord.PhaseTest || step.DurationNS == 0 {
			return errors.New("model validation: unexecuted recovery obligation")
		}
		steps[step.Name] = step
	}
	contract := fmt.Sprintf("contract=e4b-media-resources-v1;oracle_sha256=%s;model_definition=%s;modes=6;reloads=3;cycles=3;actions=5", cell.OracleSHA256, cell.ModelDefinition)
	if steps["media-resource-contract"].Evidence != contract {
		return errors.New("model validation: resource producer contract differs")
	}
	seen := map[artifact.ID]bool{}
	inputs, prompts, outputs := map[string]string{}, map[string]string{}, map[string]string{}
	var ceilingBytes, ceilingAllocations, peak, cancellations, consumerErrors, completedReloads uint64
	for reload := range reloads {
		var retained uint64
		for cycle := range cycles {
			for _, inputMode := range modes {
				for action := range actions {
					step := steps[fmt.Sprintf("media-r%d-c%d-%s-a%d", reload, cycle, inputMode, action)]
					var selection struct {
						Observation artifact.ID `json:"observation"`
						Input       string      `json:"input_sha256"`
						Prompt      string      `json:"prompt_sha256"`
						Output      string      `json:"output_sha256"`
						Mode        string      `json:"mode"`
						Reload      int         `json:"reload"`
						Cycle       int         `json:"cycle"`
						Action      int         `json:"action"`
					}
					if err := json.Unmarshal([]byte(step.Evidence), &selection); err != nil {
						return err
					}
					if selection.Mode != inputMode || selection.Reload != reload || selection.Cycle != cycle || selection.Action != action || seen[selection.Observation] {
						return errors.New("model validation: substituted or reused recovery observation")
					}
					seen[selection.Observation] = true
					for _, digest := range []string{selection.Input, selection.Prompt, selection.Output} {
						if len(digest) != sha256.Size*2 || strings.Trim(digest, "0123456789abcdef") != "" {
							return errors.New("model validation: missing recovery input or output identity")
						}
					}
					if old := inputs[inputMode]; old != "" && old != selection.Input {
						return errors.New("model validation: recovery request changed between repetitions")
					}
					if old := prompts[inputMode]; old != "" && old != selection.Prompt {
						return errors.New("model validation: recovery prepared prompt changed")
					}
					inputs[inputMode], prompts[inputMode] = selection.Input, selection.Prompt
					value, err := runrecord.RequireServingObservation(ctx, store, selection.Observation)
					if err != nil {
						return err
					}
					if value.Model != spec.Model || value.Recipe != cell.Recipe || value.Environment != cell.Environment || value.MeasuredNS != step.DurationNS || value.Usage.InputTokens == 0 || value.Usage.InputBytes == 0 || len(value.Hardware) != 3 {
						return errors.New("model validation: recovery observation authority or workload absent")
					}
					want := runrecord.OutcomeSucceeded
					if action == 1 {
						want = runrecord.OutcomeCancelled
					}
					if action == 3 {
						want = runrecord.OutcomeFailed
					}
					if value.Outcome != want || action == 3 && value.Failure != "execution_failed" {
						return errors.New("model validation: requested interruption or recovery did not execute")
					}
					if want == runrecord.OutcomeSucceeded {
						if old := outputs[inputMode]; old != "" && old != selection.Output {
							return errors.New("model validation: recovery complete output changed")
						}
						outputs[inputMode] = selection.Output
					}
					var measuredPeak uint64
					for i, sample := range value.Hardware {
						stages := [...]runrecord.ServingHardwareStage{runrecord.ServingHardwareStart, runrecord.ServingHardwarePrefill, runrecord.ServingHardwareFinish}
						if sample.Stage != stages[i] || sample.DeviceCurrentBytes == 0 || sample.DeviceAllocations == 0 {
							return errors.New("model validation: device phase observation absent")
						}
						measuredPeak = max(measuredPeak, sample.DevicePeakBytes)
					}
					finish := value.Hardware[len(value.Hardware)-1]
					if finish.ElapsedNS != value.MeasuredNS || measuredPeak == 0 || value.Resources.PeakDeviceBytes != measuredPeak {
						return errors.New("model validation: resource summary differs from observed phases")
					}
					if (reload != 0 || cycle != 0) && (finish.DeviceCurrentBytes > ceilingBytes || finish.DeviceAllocations > ceilingAllocations) {
						return errors.New("model validation: retained device memory or allocation count grew")
					}
					retained = finish.DeviceCurrentBytes
					if reload == 0 && cycle == 0 && inputMode == modes[len(modes)-1] && action == actions-1 {
						ceilingBytes, ceilingAllocations = finish.DeviceCurrentBytes, finish.DeviceAllocations
					}
					if inputMode == mode {
						peak = max(peak, measuredPeak)
						if action == 1 {
							cancellations++
						}
						if action == 3 {
							consumerErrors++
						}
					}
				}
			}
		}
		var release struct{ Before, After, Owned uint64 }
		if err := json.Unmarshal([]byte(steps[fmt.Sprintf("media-release-%d", reload)].Evidence), &release); err != nil {
			return err
		}
		if release.Owned != retained || release.After < release.Before || release.After-release.Before < release.Owned {
			return errors.New("model validation: reload did not release owned device memory")
		}
		completedReloads++
	}
	metrics := []runrecord.Metric{
		{Name: "peak-device-bytes", Unit: "bytes", Direction: runrecord.DirectionMinimize, Value: float64(peak)},
		{Name: "reload-recoveries", Unit: "count", Direction: runrecord.DirectionMaximize, Value: float64(completedReloads)},
		{Name: "cancellation-recoveries", Unit: "count", Direction: runrecord.DirectionMaximize, Value: float64(cancellations)},
		{Name: "error-recoveries", Unit: "count", Direction: runrecord.DirectionMaximize, Value: float64(consumerErrors)},
		{Name: "retained-allocation-growth", Unit: "count", Direction: runrecord.DirectionMinimize, Value: 0},
	}
	if len(cell.Bounds) != len(metrics) {
		return errors.New("model validation: incomplete resource acceptance bounds")
	}
	return checkValidationBounds(cell, metrics)
}

func testMediaResourceAcceptance(t *testing.T, store *overgodb.Store, base modelValidationSpecification, projection, environment, definition artifact.ID) {
	t.Helper()
	digest := func(value string) string { return fmt.Sprintf("%x", sha256.Sum256([]byte(value))) }
	oracle := digest("host recovery fixture, not a model measurement")
	makeSpec := func(t *testing.T, fault string) modelValidationSpecification {
		t.Helper()
		var steps []runrecord.GateStep
		batch := artifact.Batch{}
		steps = append(steps, runrecord.GateStep{Name: "media-resource-contract", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: 1,
			Evidence: fmt.Sprintf("contract=e4b-media-resources-v1;oracle_sha256=%s;model_definition=%s;modes=6;reloads=3;cycles=3;actions=5", oracle, definition)})
		var firstObservation artifact.ID
		for reload := range 3 {
			for cycle := range 3 {
				for modeIndex, mode := range []string{"text", "image", "audio", "multi-image", "video", "mixed"} {
					for action := range 5 {
						outcome, failure := runrecord.OutcomeSucceeded, ""
						if action == 1 {
							outcome = runrecord.OutcomeCancelled
						}
						if action == 3 {
							outcome, failure = runrecord.OutcomeFailed, "execution_failed"
						}
						value := runrecord.ServingObservation{Model: base.Model, Recipe: projection, Environment: environment, Task: recipe.TaskInference,
							Outcome: outcome, Failure: failure, StartedUnixNS: int64(1 + (((reload*3)+cycle)*6+modeIndex)*5 + action), MeasuredNS: 1,
							Usage: runrecord.ServingUsage{InputTokens: 4, InputBytes: 8}, Resources: runrecord.ServingResources{PeakDeviceBytes: 256},
							Hardware: []runrecord.ServingHardwareSample{
								{Stage: runrecord.ServingHardwareStart, DeviceCurrentBytes: 128, DevicePeakBytes: 128, DeviceAllocations: 4},
								{Stage: runrecord.ServingHardwarePrefill, DeviceCurrentBytes: 128, DevicePeakBytes: 256, DeviceAllocations: 4},
								{Stage: runrecord.ServingHardwareFinish, ElapsedNS: 1, DeviceCurrentBytes: 128, DevicePeakBytes: 256, DeviceAllocations: 4},
							},
						}
						target := reload == 1 && cycle == 1 && modeIndex == 1 && action == 4
						if target {
							switch fault {
							case "fabricated summary":
								value.Resources.PeakDeviceBytes++
							case "missing GPU":
								value.Hardware = nil
							case "retained bytes":
								value.Hardware[2].DeviceCurrentBytes++
							case "retained allocations":
								value.Hardware[2].DeviceAllocations++
							case "failed recovery":
								value.Outcome, value.Failure = runrecord.OutcomeFailed, "execution_failed"
							}
						}
						observation, err := runrecord.NewServingObservation(value)
						if err != nil {
							t.Fatal(err)
						}
						content, err := observation.Content()
						if err != nil {
							t.Fatal(err)
						}
						batch.Contents = append(batch.Contents, content)
						batch.Lineage = append(batch.Lineage, observation.Lineage()...)
						id := observation.ID
						if !firstObservation.Valid() {
							firstObservation = id
						}
						if target && fault == "reused observation" {
							id = firstObservation
						}
						input := digest("input/" + mode)
						if target && fault == "changed input" {
							input = digest("different request")
						}
						evidence, err := json.Marshal(map[string]any{"observation": id, "input_sha256": input, "prompt_sha256": digest("prompt/" + mode), "output_sha256": digest("output/" + mode), "mode": mode, "reload": reload, "cycle": cycle, "action": action})
						if err != nil {
							t.Fatal(err)
						}
						if target && fault == "omitted mode" {
							continue
						}
						steps = append(steps, runrecord.GateStep{Name: fmt.Sprintf("media-r%d-c%d-%s-a%d", reload, cycle, mode, action), Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: 1, Evidence: string(evidence)})
					}
				}
			}
			release := `{"before":1024,"after":1152,"owned":128}`
			if reload == 1 && fault == "failed release" {
				release = `{"before":1024,"after":1024,"owned":128}`
			}
			steps = append(steps, runrecord.GateStep{Name: fmt.Sprintf("media-release-%d", reload), Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: 1, Evidence: release})
		}
		record, err := runrecord.NewGateRecord(projection, environment, base.CodeCommit, runrecord.OutcomeSucceeded, "", uint64(len(steps)), steps)
		if err != nil {
			t.Fatal(err)
		}
		documents, err := record.Batch("fixture/media-resource/" + record.Result.ID.String())
		if err != nil {
			t.Fatal(err)
		}
		batch.Key = documents.Key
		batch.Contents = append(batch.Contents, documents.Contents...)
		batch.Lineage = append(batch.Lineage, documents.Lineage...)
		if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
			t.Fatal(err)
		}
		result := base
		result.Cells = nil
		for _, mode := range []string{"text", "image", "audio", "multi-image", "video", "mixed"} {
			name := mode
			if name == "mixed" {
				name = "mixed-history"
			}
			result.Cells = append(result.Cells, modelValidationCell{Name: "resources/" + name + "/native", Evidence: record.Result.ID, Run: record.Run.ID, OracleSHA256: oracle,
				Recipe: projection, Environment: environment, ModelDefinition: definition,
				Bounds: []modelValidationBound{
					{Metric: "peak-device-bytes", Unit: "bytes", Direction: runrecord.DirectionMinimize, Minimum: 1, Maximum: 256},
					{Metric: "reload-recoveries", Unit: "count", Direction: runrecord.DirectionMaximize, Minimum: 3, Maximum: 3},
					{Metric: "cancellation-recoveries", Unit: "count", Direction: runrecord.DirectionMaximize, Minimum: 9, Maximum: 9},
					{Metric: "error-recoveries", Unit: "count", Direction: runrecord.DirectionMaximize, Minimum: 9, Maximum: 9},
					{Metric: "retained-allocation-growth", Unit: "count", Direction: runrecord.DirectionMinimize, Minimum: 0, Maximum: 0},
				},
			})
		}
		return result
	}
	good := makeSpec(t, "")
	var names []string
	for _, cell := range good.Cells {
		names = append(names, cell.Name)
	}
	before, sequence := store.Head()
	if err := checkModelValidation(t.Context(), store, good, names); err != nil {
		t.Fatal(err)
	}
	if after, n := store.Head(); after != before || n != sequence {
		t.Fatal("resource acceptance mutated store")
	}
	for _, fault := range []string{"fabricated summary", "missing GPU", "retained bytes", "retained allocations", "failed recovery", "reused observation", "changed input", "omitted mode", "failed release"} {
		t.Run(fault, func(t *testing.T) {
			bad := makeSpec(t, fault)
			if err := checkModelValidation(t.Context(), store, bad, names); err == nil {
				t.Fatal("invalid resource observations accepted")
			}
		})
	}
	for _, fault := range []string{"stale source", "stale environment", "wrong oracle", "wrong recipe", "missing bounds"} {
		t.Run(fault, func(t *testing.T) {
			bad := good
			bad.Cells = slices.Clone(good.Cells)
			switch fault {
			case "stale source":
				bad.CodeCommit = strings.Repeat("b", 40)
			case "stale environment":
				bad.Cells[0].Environment = artifact.ID{}
			case "wrong oracle":
				bad.Cells[0].OracleSHA256 = digest("wrong oracle")
			case "wrong recipe":
				bad.Cells[0].Recipe = artifact.ID{}
			case "missing bounds":
				bad.Cells[0].Bounds = nil
			}
			if err := checkModelValidation(t.Context(), store, bad, names); err == nil {
				t.Fatal("invalid resource binding accepted")
			}
		})
	}
}
