package audioparity

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
)

func verifySpeakerSessionControls(t *testing.T, root string, session *speechrecognition.Session, publication *audioPublication, policy dataset.AudioInspectionPolicy, commit string, environment artifact.ID) {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(root, "clip-partial.wav"))
	if err != nil {
		t.Fatal(err)
	}
	id, err := artifact.IdentifyBytes(artifact.KindFile, data)
	if err != nil || id.DigestHex() != "ff407d029c95a37cd986d6f496c4780c58efa8afe4760ab84150942c25eba8ce" {
		t.Fatal("partial signal differs")
	}
	const samples = 1985 * 160
	publication.commit(t, artifact.Batch{Key: "speaker/partial-source", Artifacts: []artifact.Descriptor{{ID: id, Size: uint64(len(data))}}, Lineage: artifact.DependencyLineage(id, alignmentFileID(t, "8a6f043585e8343dc030e700ad819b866d7e632886736512e289aa7f9a12921a"))}, nil)
	origin := dataset.AudioPayloadOrigin{Container: id}
	policy.MaximumEncodedBytes = uint64(len(data))
	policy.MaximumSamples = samples
	lease, err := session.Lease(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	binding := speechrecognition.RunBinding{Key: "speaker/partial-run", CodeCommit: commit, Environment: environment}
	first, run, err := lease.Diarize(t.Context(), data, origin, policy, binding)
	if err != nil {
		t.Fatal(err)
	}
	var oracle []recipecontract.SpeechTurn
	readAudioFixtureJSON(t, "testdata/speaker_partial_turns.json", &oracle)
	clipped := 0
	for i := range oracle {
		if oracle[i].Span.End > samples {
			clipped++
			oracle[i].Span.End = samples
		}
	}
	if clipped != 1 || len(first.Turns) != 10 || !reflect.DeepEqual(first.Turns, oracle) {
		t.Fatal("partial final boundary policy differs")
	}
	if run.Outcome != runrecord.OutcomeSucceeded {
		t.Fatal("partial run absent")
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	if _, canceled, err := lease.Diarize(ctx, data, origin, policy, binding); !errors.Is(err, context.Canceled) || canceled.ID.Valid() {
		t.Fatalf("canceled execution: %v", err)
	}
	if _, _, err := lease.Transcribe(t.Context(), data, origin, policy, binding); err == nil {
		t.Fatal("wrong task admitted")
	}
	refusedPolicy := policy
	refusedPolicy.Admission.MaximumDurationNanoseconds = uint64(samples)*uint64(time.Second)/16000 - 1
	binding.Key = "speaker/refused-duration"
	_, failed, err := lease.Diarize(t.Context(), data, origin, refusedPolicy, binding)
	if !errors.Is(err, speechrecognition.ErrAudioAdmissionRefused) || failed.Outcome != runrecord.OutcomeFailed || len(failed.Outputs) != 0 || failed.CodeCommit != commit || failed.Environment != environment {
		t.Fatalf("admission failure evidence: %+v %v", failed, err)
	}
	if _, err := runrecord.RequireExactRun(t.Context(), publication.store, failed.ID); err != nil {
		t.Fatal(err)
	}
	binding.Key = "speaker/reuse-after-refusal"
	second, _, err := lease.Diarize(t.Context(), data, origin, policy, binding)
	if err != nil || !reflect.DeepEqual(second, first) {
		t.Fatalf("reuse after refusal differs: %v", err)
	}
	if err := lease.Release(); err != nil {
		t.Fatal(err)
	}
	if err := lease.Release(); err != nil {
		t.Fatal(err)
	}
	if _, _, err := lease.Diarize(t.Context(), data, origin, policy, binding); err == nil {
		t.Fatal("released lease executed")
	}
	if _, err := session.OpenStream(t.Context(), speechrecognition.StreamRequest{Source: first.Source}); err == nil {
		t.Fatal("offline speaker recipe admitted as streaming")
	}
	t.Log("partial source: 1985 valid features, 249 output frames, 10 turns; exactly one reference endpoint clipped to actual source; cancellation, task, duration, release and reuse controls passed")
}
