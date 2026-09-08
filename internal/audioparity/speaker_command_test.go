package audioparity

import (
	"encoding/json"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
	"overgo/internal/testutil"
)

func verifySpeakerCommand(t *testing.T, directory, commit, storePath string, definition artifact.ID, profile speechrecognition.SpeakerProfile, policy dataset.AudioInspectionPolicy, inputs []map[string]any, scores []evaluation.SpeechTurnScore, words [][]speechrecognition.AttributedWord) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "diarization.json")
	for _, partial := range []bool{false, true} {
		cases := inputs
		if partial {
			cases = []map[string]any{{"audio": inputs[0]["audio"], "reference": testutil.ArtifactID(t, artifact.KindOutput, "absent-speaker-reference"), "span": inputs[0]["span"]}, inputs[1]}
		}
		writeAudioMeasurementJSON(t, path, map[string]any{"recipe": definition, "profile": profile, "memory_bytes": adapterAcceptanceMemory, "inspection": policy, "inputs": cases})
		command := exec.CommandContext(t.Context(), "go", "run", "./cmd/evaluate", "-diarization-manifest", path, "-repo", storePath)
		command.Dir = directory
		output, err := command.CombinedOutput()
		if (err != nil) != partial {
			t.Fatalf("native command partial=%v: %v\n%s", partial, err, output)
		}
		var receipt struct {
			Report         artifact.ID `json:"report"`
			Inputs, Failed int
		}
		found := false
		for line := range strings.SplitSeq(string(output), "\n") {
			if !strings.HasPrefix(line, "{\"report\":") {
				continue
			}
			if found {
				t.Fatal("multiple command reports")
			}
			found = true
			if err := json.Unmarshal([]byte(line), &receipt); err != nil {
				t.Fatal(err)
			}
		}
		failed := 0
		if partial {
			failed = 1
		}
		if !found || receipt.Inputs != len(cases) || receipt.Failed != failed {
			t.Fatalf("command denominator: %s", output)
		}
		reader, err := overgodb.OpenReadOnly(storePath)
		if err != nil {
			t.Fatal(err)
		}
		func() {
			defer reader.Close()
			content, ok, err := artifact.ReadContent(t.Context(), reader, receipt.Report)
			if err != nil || !ok {
				t.Fatalf("report missing: %v", err)
			}
			var results []struct {
				Run, Output, Attribution artifact.ID
				Score                    *evaluation.SpeechTurnScore
				Failure                  string
			}
			if err := json.Unmarshal(content.Data, &results); err != nil || len(results) != len(cases) {
				t.Fatalf("retained denominator: %v", err)
			}
			for i, result := range results {
				if partial && i == 0 {
					if result.Failure == "" || result.Run.Valid() || result.Output.Valid() || result.Score != nil {
						t.Fatal("missing reference became a partial success")
					}
					continue
				}
				if result.Failure != "" || result.Score == nil || !reflect.DeepEqual(*result.Score, scores[i]) {
					t.Fatalf("command/library score differs: %+v", result)
				}
				run, err := runrecord.RequireExactRun(t.Context(), reader, result.Run)
				if err != nil || run.CodeCommit != commit || run.Recipe != definition || run.Outcome != runrecord.OutcomeSucceeded {
					t.Fatalf("command run binding: %+v %v", run, err)
				}
				attribution, ok, err := artifact.ReadContent(t.Context(), reader, result.Attribution)
				if err != nil || !ok {
					t.Fatalf("word evidence absent: %v", err)
				}
				var actual struct {
					Alignment, Turns, Activity artifact.ID
					Words                      []speechrecognition.AttributedWord
				}
				if err := json.Unmarshal(attribution.Data, &actual); err != nil || actual.Alignment != cases[i]["alignment"] || actual.Turns != result.Output || !reflect.DeepEqual(actual.Words, words[i]) {
					t.Fatalf("stored word attribution differs: %v", err)
				}
			}
		}()
		t.Logf("native diarization command: inputs=%d scored=%d failed=%d; all attempts retained; report=%s", receipt.Inputs, receipt.Inputs-receipt.Failed, receipt.Failed, receipt.Report)
	}
}
