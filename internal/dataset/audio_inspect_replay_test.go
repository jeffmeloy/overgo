package dataset

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
)

type inspectionBatchCapture struct {
	artifact.Repository
	batch artifact.Batch
}

func (capture *inspectionBatchCapture) Commit(ctx context.Context, batch artifact.Batch) (artifact.CommitID, error) {
	capture.batch = batch
	return capture.Repository.Commit(ctx, batch)
}

func TestAudioInspectionCanonicalReplayAcceptance(t *testing.T) {
	for _, signal := range []struct {
		name    string
		samples []float32
		outcome recipecontract.AudioAdmissionOutcome
	}{
		{"accepted", []float32{.5, -.5}, recipecontract.AudioAdmissionAccepted},
		{"silent-quarantine", []float32{0, 0}, recipecontract.AudioAdmissionQuarantined},
	} {
		t.Run(signal.name, func(t *testing.T) {
			data := floatWAV(signal.samples)
			fresh, err := overgodb.Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			defer fresh.Close()
			container := publishInspectionSource(t, fresh, []byte("replay-container"))
			origin := AudioPayloadOrigin{Container: container, Column: "bytes"}
			capture := &inspectionBatchCapture{Repository: fresh}
			expected, err := InspectAudio(t.Context(), capture, data, origin, inspectionPolicy())
			if err != nil || expected.Decision.Outcome != signal.outcome {
				t.Fatalf("fresh inspection: outcome=%s err=%v", expected.Decision.Outcome, err)
			}
			for _, shape := range []string{"fresh", "historical-without-source", "historical-with-source"} {
				t.Run(shape, func(t *testing.T) {
					path := t.TempDir()
					store, err := overgodb.Open(path)
					if err != nil {
						t.Fatal(err)
					}
					t.Cleanup(func() { _ = store.Close() })
					publishInspectionSource(t, store, []byte("replay-container"))
					publishInspectionSource(t, store, data)
					batch := capture.batch
					if shape == "fresh" {
						if _, err := InspectAudio(t.Context(), store, data, origin, inspectionPolicy()); err != nil {
							t.Fatal(err)
						}
					} else {
						// Immutable historical envelopes used this unversioned key,
						// both before and after including the source descriptor.
						batch.Key = "audio/inspection/" + expected.DecisionID.DigestHex()
						if shape == "historical-without-source" {
							batch.Artifacts = nil
						}
						if _, err := store.Commit(t.Context(), batch); err != nil {
							t.Fatal(err)
						}
					}
					head, sequence := store.Head()
					receipt, found, err := store.CommitDeltaAt(t.Context(), sequence)
					if err != nil || !found {
						t.Fatalf("receipt: found=%t err=%v", found, err)
					}
					for range 2 {
						replayed, err := InspectAudio(t.Context(), store, data, origin, inspectionPolicy())
						if err != nil || !reflect.DeepEqual(replayed, expected) {
							t.Fatalf("replay changed evidence or samples: %v", err)
						}
						after, afterSequence := store.Head()
						if after != head || afterSequence != sequence {
							t.Fatal("identical inspection changed store head")
						}
						unchanged, found, err := store.CommitDeltaAt(t.Context(), sequence)
						if err != nil || !found || !reflect.DeepEqual(receipt, unchanged) {
							t.Fatalf("historical receipt changed: %v", err)
						}
						if err := store.Close(); err != nil {
							t.Fatal(err)
						}
						reopened, err := overgodb.Open(path)
						if err != nil {
							t.Fatal(err)
						}
						store = reopened
					}
					conflict := capture.batch
					conflict.Key = batch.Key
					if shape != "historical-without-source" {
						conflict.Artifacts = nil
					}
					if _, err := store.Commit(t.Context(), conflict); !errors.Is(err, overgodb.ErrBatchKeyConflict) {
						t.Fatalf("same-key conflict was not refused: %v", err)
					}
					if _, err := InspectAudio(t.Context(), store, []byte("different bytes"), AudioPayloadOrigin{Container: expected.Signal.Source.Audio}, inspectionPolicy()); err == nil {
						t.Fatal("changed whole-file source was accepted")
					}
					ctx, cancel := context.WithCancelCause(t.Context())
					cancel(errors.New("inspection canceled"))
					if _, err := InspectAudio(ctx, store, data, origin, inspectionPolicy()); !errors.Is(err, context.Canceled) {
						t.Fatalf("cancellation: %v", err)
					}
					after, afterSequence := store.Head()
					if after != head || afterSequence != sequence {
						t.Fatal("refused request changed store head")
					}
				})
			}
			t.Run("immutable-source-size", func(t *testing.T) {
				store, err := overgodb.Open(t.TempDir())
				if err != nil {
					t.Fatal(err)
				}
				defer store.Close()
				publishInspectionSource(t, store, []byte("replay-container"))
				if _, err := store.Commit(t.Context(), artifact.Batch{
					Key:       "fixture/wrong-source-size",
					Artifacts: []artifact.Descriptor{{ID: expected.Signal.Source.Audio, Size: uint64(len(data) + 1)}},
				}); err != nil {
					t.Fatal(err)
				}
				head, sequence := store.Head()
				if _, err := InspectAudio(t.Context(), store, data, origin, inspectionPolicy()); err == nil {
					t.Fatal("immutable source size mismatch was accepted")
				}
				after, afterSequence := store.Head()
				if after != head || afterSequence != sequence {
					t.Fatal("source size mismatch changed store head")
				}
			})
		})
	}
	t.Log("executed accepted and quarantined audio across fresh and both historical envelopes; repeated/reopened replay, immutable receipts, conflict refusal and cancellation; no model or GPU execution")
}
