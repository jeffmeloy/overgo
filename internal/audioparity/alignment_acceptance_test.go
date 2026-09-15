package audioparity

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
	"overgo/internal/testskip"
	"overgo/internal/trainingdata"
)

func TestSpeechAlignmentAcceptance(t *testing.T) {
	t.Parallel()
	if testing.Short() {
		t.Skip(testskip.ShortIntegration + ": real CTC alignment and independent corpus annotations run as exact acceptance")
	}
	if os.Getenv(testskip.StoreAcceptanceEnv) == "" {
		t.Skip(testskip.StoreAcceptance)
	}
	fixture := loadCTCTrainingFixture(t)
	source := loadAlignmentFixture(t, fixture.root, fixture.storeRoot)
	sourceDirectory, commit := snapshotAudioSource(t)
	storePath := filepath.Join(t.TempDir(), "store")
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	publication := audioPublication{store: store}
	base, _ := publication.publishTranscriptionBase(t, fixture)
	profile, err := speechrecognition.NewAlignmentProfile(speechrecognition.AlignmentProfile{
		FrameMapping: "pooled-hop-cells", WordMapping: "whitespace-token-spans", BlankBoundary: "excluded", Confidence: "geometric-mean-target-probability"})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := profile.Batch("alignment/profile")
	publication.commit(t, batch, err)
	definition, err := modelrecipe.AlignmentDefinition(base, profile.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(t.Context(), store, "alignment/recipe", definition); err != nil {
		t.Fatal(err)
	}
	transform, err := trainingdata.NewTextTransform(true)
	if err != nil {
		t.Fatal(err)
	}
	content, err := transform.Content()
	if err != nil {
		t.Fatal(err)
	}
	shard, annotations := alignmentFileID(t, source.ShardSHA256), alignmentFileID(t, source.AnnotationSHA256)
	publication.commit(t, artifact.Batch{Key: "alignment/sources", Contents: []artifact.Content{content}, Artifacts: []artifact.Descriptor{
		{ID: shard, Size: source.ShardBytes}, {ID: annotations, Size: source.AnnotationBytes},
	}}, nil)
	environment, err := runrecord.CurrentEnvironment("cpu", "go-host")
	if err != nil {
		t.Fatal(err)
	}
	batch, err = environment.Batch("alignment/environment")
	publication.commit(t, batch, err)
	policyData, err := os.ReadFile("../dataset/testdata/audio_inspection_policy.json")
	if err != nil {
		t.Fatal(err)
	}
	var policy dataset.AudioInspectionPolicy
	if err := json.Unmarshal(policyData, &policy); err != nil {
		t.Fatal(err)
	}
	for _, test := range source.Cases {
		policy.MaximumSamples = max(policy.MaximumSamples, test.Samples)
		policy.MaximumEncodedBytes = max(policy.MaximumEncodedBytes, test.EncodedBytes)
	}
	rows, err := dataset.OpenParquetRows(t.Context(), source.path, []string{"audio.bytes", "audio_id", "text"}, adapterAcceptanceMemory)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	session, err := speechrecognition.LoadSession(t.Context(), store, definition.ID, adapterAcceptanceMemory)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close(t.Context())
	completed, words, boundaries := 0, 0, 0
	var commandInputs []map[string]any
	var expectedScores []evaluation.AlignmentScore
	for _, test := range source.Cases {
		t.Run(test.ID, func(t *testing.T) {
			row, err := rows.Read(t.Context(), test.Row)
			if err != nil || row["audio.bytes"] == nil || row["audio_id"] == nil || *row["audio_id"] != test.ID || row["text"] == nil || *row["text"] != test.Text {
				t.Fatalf("physical source row: %v", err)
			}
			payload := []byte(*row["audio.bytes"])
			id, err := artifact.IdentifyBytes(artifact.KindFile, payload)
			if err != nil || id.DigestHex() != test.AudioSHA256 || uint64(len(payload)) != test.EncodedBytes {
				t.Fatal("source audio bytes differ")
			}
			origin := dataset.AudioPayloadOrigin{Container: shard, Column: "audio.bytes", Row: &test.Row}
			inspection, err := dataset.InspectAudio(t.Context(), store, payload, origin, policy)
			if err != nil || inspection.Decision.Outcome != recipecontract.AudioAdmissionAccepted || uint64(len(inspection.Samples)) != test.Samples {
				t.Fatalf("source signal admission: %+v %v", inspection.Decision, err)
			}
			target, err := transform.Apply(test.Text)
			if err != nil {
				t.Fatal(err)
			}
			transcript := recipecontract.Transcription{Source: inspection.Signal.Source, Text: target, Language: "en"}
			condition, err := artifact.JSONContent(artifact.JSONContract(artifact.KindOutput, speechrecognition.TranscriptionSchema), transcript)
			if err != nil {
				t.Fatal(err)
			}
			publication.commit(t, artifact.Batch{Key: "alignment/condition/" + test.ID, Contents: []artifact.Content{condition},
				Lineage: artifact.DependencyLineage(condition.Descriptor.ID, annotations, shard, inspection.Signal.Source.Audio, inspection.Signal.Source.Profile, transform.ID)}, nil)
			request := speechrecognition.AlignmentRequest{Transcription: condition.Descriptor.ID, Span: recipecontract.SampleSpan{End: test.Samples}}
			lease, err := session.Lease(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			defer lease.Release()
			binding := speechrecognition.RunBinding{Key: "alignment/run/" + test.ID, CodeCommit: commit, Environment: environment.ID}
			predicted, run, err := lease.Align(t.Context(), payload, origin, policy, request, binding)
			if err != nil {
				t.Fatal(err)
			}
			if run.Recipe != definition.ID || run.CodeCommit != commit || run.Environment != environment.ID || run.Outcome != runrecord.OutcomeSucceeded || len(run.Outputs) != 1 {
				t.Fatalf("run binding differs: %+v", run)
			}
			persisted, err := speechrecognition.RequireAlignment(t.Context(), store, run.Outputs[0])
			if err != nil || !reflect.DeepEqual(persisted, predicted) {
				t.Fatalf("persisted alignment differs: %v", err)
			}
			reference := recipecontract.TimestampedAlignment{Source: transcript.Source, Transcription: condition.Descriptor.ID,
				Items: source.referenceWords(t, test, inspection.Signal.Format.SampleRate, transform)}
			score, err := evaluation.ScoreAlignment(reference, predicted)
			if err != nil {
				t.Fatal(err)
			}
			if score.Words == 0 || score.Boundaries != 2*score.Words {
				t.Fatal("incomplete scoring denominator")
			}
			referenceContent, err := artifact.JSONContent(artifact.JSONContract(artifact.KindOutput, "overgo/audio-alignment/v1"), reference)
			if err != nil {
				t.Fatal(err)
			}
			publication.commit(t, artifact.Batch{Key: "alignment/reference/" + test.ID, Contents: []artifact.Content{referenceContent}, Lineage: artifact.DependencyLineage(referenceContent.Descriptor.ID, condition.Descriptor.ID, annotations, shard)}, nil)
			commandInputs = append(commandInputs, map[string]any{
				"audio":   dataset.AudioPayloadReference{Path: source.path, Audio: id, Origin: origin},
				"request": request, "reference": referenceContent.Descriptor.ID,
			})
			expectedScores = append(expectedScores, score)
			if test.Row == source.Cases[0].Row {
				// A failed invocation cannot contaminate the reusable lease.
				bad := request
				bad.Span.End++
				binding.Key += "/bad-span"
				if _, failed, err := lease.Align(t.Context(), payload, origin, policy, bad, binding); err == nil || failed.Outcome != runrecord.OutcomeFailed {
					t.Fatalf("out-of-source interval did not fail durably: run=%+v error=%v", failed, err)
				}
				binding.Key += "/recovery"
				again, _, err := lease.Align(t.Context(), payload, origin, policy, request, binding)
				if err != nil || !reflect.DeepEqual(again, predicted) {
					t.Fatalf("lease recovery changed output: %v", err)
				}
			}
			rate := float64(inspection.Signal.Format.SampleRate)
			t.Logf("annotation agreement: words=%d boundaries=%d median_ms=%.3f mean_ms=%.3f max_ms=%.3f source=%s output=%s; official forced-alignment-derived timing, not human boundary truth", score.Words, score.Boundaries, score.MedianAbsoluteSamples*1000/rate, score.MeanAbsoluteSamples*1000/rate, float64(score.MaximumAbsoluteSamples)*1000/rate, id, run.Outputs[0])
			completed++
			words += score.Words
			boundaries += score.Boundaries
		})
	}
	if completed != len(source.Cases) || words == 0 || boundaries != words*2 {
		t.Fatalf("alignment coverage: %d/%d clips, %d words %d boundaries", completed, len(source.Cases), words, boundaries)
	}
	if err := session.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	manifest, err := json.Marshal(map[string]any{"base_recipe": base.ID, "profile": profile, "memory_bytes": adapterAcceptanceMemory, "inspection": policy, "inputs": commandInputs})
	if err != nil {
		t.Fatal(err)
	}
	manifestPath := filepath.Join(t.TempDir(), "alignment.json")
	if err := os.WriteFile(manifestPath, manifest, 0o600); err != nil {
		t.Fatal(err)
	}
	output := baselineCommand(t, sourceDirectory, "go", "run", "./cmd/evaluate", "-alignment-manifest", manifestPath, "-repo", storePath)
	lines := strings.Split(strings.TrimSpace(output), "\n")
	var receipt struct {
		Report         artifact.ID `json:"report"`
		Inputs, Failed int
	}
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &receipt); err != nil || receipt.Inputs != len(commandInputs) || receipt.Failed != 0 {
		t.Fatalf("native alignment command: %s: %v", output, err)
	}
	reader, err := overgodb.OpenReadOnly(storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	report, present, err := artifact.ReadContent(t.Context(), reader, receipt.Report)
	if err != nil || !present {
		t.Fatalf("command report absent: %v", err)
	}
	var results []struct {
		Run     artifact.ID               `json:"run"`
		Score   evaluation.AlignmentScore `json:"score"`
		Failure string                    `json:"failure"`
	}
	if err := json.Unmarshal(report.Data, &results); err != nil || len(results) != len(expectedScores) {
		t.Fatalf("command denominator: %v", err)
	}
	for index, result := range results {
		if result.Score != expectedScores[index] || result.Failure != "" {
			t.Fatalf("command/library score differs: %+v", result)
		}
		run, err := runrecord.RequireExactRun(t.Context(), reader, result.Run)
		if err != nil || run.CodeCommit != commit || run.Recipe != definition.ID {
			t.Fatalf("command execution binding: %+v %v", run, err)
		}
	}
	t.Logf("native evaluate command: %d/%d inputs rescored from fresh inference, exact library scores; report=%s", receipt.Inputs-receipt.Failed, receipt.Inputs, receipt.Report)
	if err := reader.Close(); err != nil {
		t.Fatal(err)
	}
	// One invalid interval alongside one valid input must retain both attempts
	// and exit nonzero, without treating the surviving score as a complete pass.
	badRequest := commandInputs[0]["request"].(speechrecognition.AlignmentRequest)
	badRequest.Span.End++
	commandInputs[0]["request"] = badRequest
	manifest, err = json.Marshal(map[string]any{"base_recipe": base.ID, "profile": profile, "memory_bytes": adapterAcceptanceMemory, "inspection": policy, "inputs": commandInputs[:2]})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(manifestPath, manifest, 0o600); err != nil {
		t.Fatal(err)
	}
	command := exec.CommandContext(t.Context(), "go", "run", "./cmd/evaluate", "-alignment-manifest", manifestPath, "-repo", storePath)
	command.Dir = sourceDirectory
	refused, err := command.CombinedOutput()
	if err == nil {
		t.Fatal("partial alignment command reported success")
	}
	found := false
	for line := range strings.SplitSeq(string(refused), "\n") {
		if !strings.HasPrefix(line, "{\"report\":") {
			continue
		}
		if err := json.Unmarshal([]byte(line), &receipt); err != nil || receipt.Inputs != 2 || receipt.Failed != 1 {
			t.Fatalf("partial command denominator: %s %v", refused, err)
		}
		found = true
	}
	if !found {
		t.Fatalf("partial command lost report: %s", refused)
	}
	reader, err = overgodb.OpenReadOnly(storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	report, present, err = artifact.ReadContent(t.Context(), reader, receipt.Report)
	if err != nil || !present {
		t.Fatalf("partial report absent: %v", err)
	}
	results = nil
	if err := json.Unmarshal(report.Data, &results); err != nil || len(results) != 2 || results[0].Failure == "" || results[1].Failure != "" || results[1].Score != expectedScores[1] {
		t.Fatalf("partial report lost attempts: %s %v", report.Data, err)
	}
	failedRun, err := runrecord.RequireExactRun(t.Context(), reader, results[0].Run)
	if err != nil || failedRun.Outcome != runrecord.OutcomeFailed || len(failedRun.Outputs) != 0 {
		t.Fatalf("partial command failure provenance: %+v %v", failedRun, err)
	}
	t.Log("native evaluator refusal: 1 failed and 1 scored input retained; nonzero exit, no partial pass")
	t.Log(fmt.Sprintf("CPU word alignment: %d/%d real clips, %d words, %d boundaries; separate pinned CTC recurrence oracle; no GPU, human-ground-truth, diarization or production promotion claim", completed, len(source.Cases), words, boundaries))
}
