package dataset

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
)

func inspectionPolicy() AudioInspectionPolicy {
	return AudioInspectionPolicy{MaximumEncodedBytes: 1024, MaximumSamples: 16, ClipThreshold: 1,
		Admission: recipecontract.AudioAdmissionPolicy{MinimumChannels: 1, MaximumChannels: 1, SilenceRMSThreshold: 1.0 / 32768, MaximumAbsoluteDCOffset: 1}}
}

func TestAudioDecodeAdmissionAndExactLineage(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	for _, test := range []struct {
		name      string
		samples   []float32
		violation recipecontract.AudioSignalViolation
	}{
		{"accepted", []float32{.5, -.5}, ""},
		{"silent", []float32{0, 0}, recipecontract.AudioViolationSilent},
		{"near-silent", []float32{1.0 / 65536, -1.0 / 65536}, recipecontract.AudioViolationSilent},
		{"clipping", []float32{1, -1}, recipecontract.AudioViolationClipped},
		{"nan", []float32{float32(math.NaN()), .5}, recipecontract.AudioViolationNonFiniteSamples},
		{"infinite", []float32{float32(math.Inf(1)), .5}, recipecontract.AudioViolationNonFiniteSamples},
	} {
		t.Run(test.name, func(t *testing.T) {
			data := floatWAV(test.samples)
			source := publishInspectionSource(t, store, data)
			result, err := InspectAudio(t.Context(), store, data, AudioPayloadOrigin{Container: source}, inspectionPolicy())
			if err != nil {
				t.Fatal(err)
			}
			if test.violation == "" {
				if result.Decision.Outcome != recipecontract.AudioAdmissionAccepted || !slices.Equal(result.Samples, test.samples) {
					t.Fatalf("accepted result=%+v", result)
				}
			} else if result.Decision.Outcome != recipecontract.AudioAdmissionQuarantined || !slices.Contains(result.Decision.Violations, test.violation) || result.Samples != nil {
				t.Fatalf("quarantine=%+v", result)
			}
			if _, err := audioSignalProfileCodec.RequireExactLineage(t.Context(), store, result.SignalID, func(v AudioSignalProfileDocument) []artifact.Lineage {
				return artifact.DependencyLineage(v.ID, v.Profile.Source.Audio, v.Profile.Source.Profile)
			}); err != nil {
				t.Fatal(err)
			}
			loaded, err := audioAdmissionDecisionCodec.RequireExactLineage(t.Context(), store, result.DecisionID, func(v AudioAdmissionDecisionDocument) []artifact.Lineage {
				return artifact.DependencyLineage(v.ID, v.Signal, v.Policy)
			})
			if err != nil || loaded.Signal != result.SignalID || loaded.Policy != result.PolicyID {
				t.Fatalf("decision=%+v %v", loaded, err)
			}
			head, _ := store.Head()
			replay, err := InspectAudio(t.Context(), store, data, AudioPayloadOrigin{Container: source}, inspectionPolicy())
			after, _ := store.Head()
			if err != nil || replay.DecisionID != result.DecisionID || head != after {
				t.Fatalf("non-idempotent inspection: %v", err)
			}
		})
	}
}

func TestAudioDecodeQuarantinesFailuresAndRejectsUnboundSources(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	valid := floatWAV([]float32{.5, -.5})
	for _, test := range []struct {
		name   string
		data   []byte
		status recipecontract.AudioDecodeStatus
		bound  uint64
	}{
		{"truncated", valid[:len(valid)-1], recipecontract.AudioDecodeTruncated, 16},
		{"unsupported", []byte("OggSunsupported"), recipecontract.AudioDecodeUnsupported, 16},
		{"budget", valid, recipecontract.AudioDecodeResourceLimit, 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			source := publishInspectionSource(t, store, test.data)
			policy := inspectionPolicy()
			policy.MaximumSamples = test.bound
			result, err := InspectAudio(t.Context(), store, test.data, AudioPayloadOrigin{Container: source}, policy)
			if err != nil {
				t.Fatal(err)
			}
			if result.Signal.DecodeStatus != test.status || result.Decision.Outcome != recipecontract.AudioAdmissionQuarantined || result.Samples != nil || result.Signal.SampleCount != 0 || result.DecodeError == "" {
				t.Fatalf("failure=%+v", result)
			}
		})
	}
	missing := testutil.ArtifactID(t, artifact.KindFile, "missing")
	if _, err := InspectAudio(t.Context(), store, valid, AudioPayloadOrigin{Container: missing}, inspectionPolicy()); err == nil {
		t.Fatal("missing container accepted")
	}
	source := publishInspectionSource(t, store, valid)
	if _, err := InspectAudio(t.Context(), store, []byte("wrong"), AudioPayloadOrigin{Container: source}, inspectionPolicy()); err == nil {
		t.Fatal("different source bytes accepted")
	}
	head, _ := store.Head()
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(errors.New("stop"))
	if _, err := InspectAudio(ctx, store, valid, AudioPayloadOrigin{Container: source}, inspectionPolicy()); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	after, _ := store.Head()
	if head != after {
		t.Fatal("canceled operation changed store")
	}
}

func TestAudioDecodeEmbeddedSelectorLineage(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	container := publishInspectionSource(t, store, []byte("parquet-container"))
	data := floatWAV([]float32{.5, -.5})
	first, err := InspectAudio(t.Context(), store, data, AudioPayloadOrigin{Container: container, Column: "bytes", ValueIndex: 0}, inspectionPolicy())
	if err != nil {
		t.Fatal(err)
	}
	head, _ := store.Head()
	replay, err := InspectAudio(t.Context(), store, data, AudioPayloadOrigin{Container: container, Column: "bytes", ValueIndex: 0}, inspectionPolicy())
	after, _ := store.Head()
	if err != nil || replay.DecisionID != first.DecisionID || head != after {
		t.Fatalf("embedded replay changed publication: %v", err)
	}
	second, err := InspectAudio(t.Context(), store, data, AudioPayloadOrigin{Container: container, Column: "bytes", ValueIndex: 1}, inspectionPolicy())
	if err != nil {
		t.Fatal(err)
	}
	if first.SignalID == second.SignalID || first.Signal.Source.Audio != second.Signal.Source.Audio {
		t.Fatal("selector or payload identity lost")
	}
	parents, err := store.Parents(t.Context(), first.Signal.Source.Profile)
	if err != nil {
		t.Fatal(err)
	}
	if len(parents) != 2 {
		t.Fatalf("decode profile parents=%v", parents)
	}
	content, found, err := artifact.ReadContent(t.Context(), store, first.Signal.Source.Audio)
	if err != nil || found || len(content.Data) != 0 {
		t.Fatal("source corpus bytes were copied into store")
	}
}

func publishInspectionSource(t *testing.T, store artifact.Repository, data []byte) artifact.ID {
	t.Helper()
	id := testutil.ArtifactBytesID(t, artifact.KindFile, data)
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: "fixture/audio/" + id.DigestHex(), Artifacts: []artifact.Descriptor{{ID: id, Size: uint64(len(data))}}}); err != nil {
		t.Fatal(err)
	}
	return id
}

func floatWAV(samples []float32) []byte {
	data := testutil.MonoPCM16WAV(16000, make([]int16, len(samples)*2))
	binary.LittleEndian.PutUint16(data[20:], 3)
	binary.LittleEndian.PutUint16(data[32:], 4)
	binary.LittleEndian.PutUint16(data[34:], 32)
	binary.LittleEndian.PutUint32(data[28:], 16000*4)
	for i, v := range samples {
		binary.LittleEndian.PutUint32(data[44+i*4:], math.Float32bits(v))
	}
	return data
}
