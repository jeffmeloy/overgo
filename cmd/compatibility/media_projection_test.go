package main

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestTypedMediaProjection(t *testing.T) {
	scope := mediaReportScope{Tasks: []recipe.Task{recipe.TaskImageGen, recipe.TaskVideoGen}}
	row := reportRow{model: "fixture", task: string(recipe.TaskImageGen), tier: string(runrecord.TierRealArtifactSmoke),
		modelID:  testutil.ArtifactID(t, artifact.KindModel, "typed media"),
		recipeID: testutil.ArtifactID(t, artifact.KindRecipe, "typed media")}
	row.activation.Definition.ID = row.recipeID
	row.activation.Event.ID = testutil.ArtifactID(t, artifact.KindEvidence, "activation")
	previous := testutil.ArtifactID(t, artifact.KindEvidence, "previous activation")
	row.activation.Event.Supersedes = &previous
	row.verification.Run.ID = testutil.ArtifactID(t, artifact.KindRun, "historical verifier")
	row.verification.Run.Recipe = row.recipeID
	row.verification.Run.CodeCommit = strings.Repeat("ab", 20)
	row.verification.Run.Environment = testutil.ArtifactID(t, artifact.KindEvidence, "environment")
	row.verification.Run.MeasuredNS = uint64(time.Second)
	row.verification.Run.Phases = []runrecord.PhaseMetric{{Phase: runrecord.PhaseGenerate, DurationNS: uint64(time.Second)}}
	row.verification.Run.Inputs = []artifact.ID{testutil.ArtifactID(t, artifact.KindFile, "input")}
	row.verification.Run.Outputs = []artifact.ID{testutil.ArtifactID(t, artifact.KindOutput, "output")}
	row.verification.Gate.ID = testutil.ArtifactID(t, artifact.KindEvidence, "gate")
	row.verification.Gate.Recipe = row.recipeID
	row.verification.Gate.CodeCommit = row.verification.Run.CodeCommit
	row.verification.Gate.Environment = row.verification.Run.Environment
	row.verification.Gate.Steps = []runrecord.GateStep{{Evidence: "peak_device_bytes=1048576"}, {Evidence: "peak_device_bytes=1073741824"}}
	stale := reportRow{model: "fixture", modelID: row.modelID, task: string(recipe.TaskVideoGen), stale: "missing verification"}
	projection, err := projectMediaRows([]reportRow{row, stale}, scope, true)
	if err != nil {
		t.Fatal(err)
	}
	if projection.Counts != (mediaProjectionCounts{Activations: 2, Stale: 1, MeasuredVerifiers: 1}) || !projection.Truncated {
		t.Fatal("wrong denominator or completeness", projection)
	}
	if projection.Rows[0].Supersedes == nil || *projection.Rows[0].Supersedes != previous || projection.Rows[0].GenerationGap == "" || projection.Rows[0].RepairOwner == "" || projection.Rows[1].Verifier != nil {
		t.Fatal("lost history or invented generation acceptance")
	}
	data, err := json.Marshal(projection)
	if err != nil {
		t.Fatal(err)
	}
	var decoded mediaProjection
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(projection, decoded) {
		t.Fatal("typed values changed in JSON")
	}
	measurement := decoded.Rows[0].Verifier
	if measurement.Source != row.verification.Run.CodeCommit || measurement.Run != row.verification.Run.ID || measurement.Gate != row.verification.Gate.ID || measurement.PeakStepDeviceBytes == nil || *measurement.PeakStepDeviceBytes != 1073741824 {
		t.Fatal("lost exact historical identity or maximum step peak", measurement)
	}
	var headline bytes.Buffer
	writeMediaVerdict(&headline, decoded)
	if !strings.Contains(headline.String(), "2 activations; 1 are stale and 1 healthy") || !strings.Contains(mediaRunResult(measurement), "1073741824 B (1024.000 MiB; 1.000 GiB)") || mediaByteCell(1048576) != "1048576 B (1.000 MiB; 0.001 GiB)" {
		t.Fatal("human projection differs from typed values", headline.String(), mediaRunResult(measurement))
	}
	for name, mutate := range map[string]func(*reportRow){
		"source":            func(r *reportRow) { r.verification.Gate.CodeCommit = "foreign" },
		"environment":       func(r *reportRow) { r.verification.Gate.Environment = artifact.ID{} },
		"verifier recipe":   func(r *reportRow) { r.verification.Gate.Recipe = artifact.ID{} },
		"selected recipe":   func(r *reportRow) { r.recipeID = artifact.ID{} },
		"stale measurement": func(r *reportRow) { r.stale = "not current" },
		"scope":             func(r *reportRow) { r.task = string(recipe.TaskInference) },
	} {
		t.Run(name, func(t *testing.T) {
			changed := row
			mutate(&changed)
			if _, err := projectMediaRows([]reportRow{changed}, scope, false); err == nil {
				t.Fatal("contradiction accepted")
			}
		})
	}
	if _, err := projectMediaRows([]reportRow{row, row}, scope, false); err == nil {
		t.Fatal("duplicate denominator accepted")
	}
	for _, marker := range []string{"peak_device_bytes=", "peak_device_bytes=-1", "peak_device_bytes=42MiB", "peak_device_bytes=18446744073709551616", "peak_device_bytes=1 peak_device_bytes=2"} {
		changed := row.verification
		changed.Gate.Steps = []runrecord.GateStep{{Evidence: marker}}
		if _, err := projectMediaVerifier(changed); err == nil {
			t.Fatal("invalid or contradictory units accepted", marker)
		}
	}
	for _, marker := range []string{"", "peak_device_bytes=0", "peak_device_bytes=18446744073709551615"} {
		changed := row.verification
		changed.Gate.Steps = []runrecord.GateStep{{Evidence: marker}}
		value, err := projectMediaVerifier(changed)
		if err != nil {
			t.Fatal(err)
		}
		encoded, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		var roundtrip mediaVerifierProjection
		if err := json.Unmarshal(encoded, &roundtrip); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(*value, roundtrip) || (value.PeakStepDeviceBytes == nil) != (marker == "") {
			t.Fatal("absent, zero or exact large integer changed", marker)
		}
	}
	row.verification = runrecord.Verification{}
	activationOnly, err := projectMediaRows([]reportRow{row}, scope, false)
	if err != nil || activationOnly.Counts.MeasuredVerifiers != 0 || activationOnly.Rows[0].Verifier != nil || activationOnly.Rows[0].GenerationGap == "" {
		t.Fatal("activation-only gained measurement credit", err)
	}
}
