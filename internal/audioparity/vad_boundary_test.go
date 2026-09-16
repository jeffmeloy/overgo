package audioparity

import (
	"encoding/json"
	"reflect"
	"testing"

	"overgo/internal/speechactivity"
)

type vadBoundaryCase struct {
	Name          string                        `json:"name"`
	Config        speechactivity.BoundaryConfig `json:"config"`
	Probabilities []float32                     `json:"probabilities"`
	Stream        []speechactivity.Boundary     `json:"stream"`
	Offline       []int                         `json:"offline"`
	MergeSilence  int                           `json:"merge_silence"`
	ExtendSpeech  int                           `json:"extend_speech"`
}

func vadBoundaryCases(t *testing.T) []vadBoundaryCase {
	t.Helper()
	var reference struct {
		Source string            `json:"source_commit"`
		Cases  []vadBoundaryCase `json:"cases"`
	}
	readAudioFixtureJSON(t, "testdata/vad_boundaries.json", &reference)
	if reference.Source != "c30ec49e8cc69642b0ee65362eba11b9d11c6e54" || len(reference.Cases) != 22 {
		t.Fatal("boundary source or denominator differs")
	}
	return reference.Cases
}

func TestVADStreamBoundaryParity(t *testing.T) {
	t.Parallel()
	for _, fixture := range vadBoundaryCases(t) {
		t.Run(fixture.Name, func(t *testing.T) {
			policy, err := speechactivity.NewBoundaryPolicy(fixture.Config, vadReferenceBytes)
			if err != nil {
				t.Fatal(err)
			}
			var state speechactivity.BoundaryState
			for index, probability := range fixture.Probabilities {
				calls := 0
				err := policy.Process(t.Context(), []float32{probability}, &state, false, func(got speechactivity.Boundary) error {
					calls++
					got.Probability, got.Smoothed = 0, 0
					if got != fixture.Stream[index] {
						t.Fatalf("frame %d: got %+v want %+v", index, got, fixture.Stream[index])
					}
					return nil
				})
				if err != nil || calls != 1 {
					t.Fatalf("boundary execution: %v calls=%d", err, calls)
				}
				encoded, err := json.Marshal(state)
				if err != nil {
					t.Fatal(err)
				}
				var restored speechactivity.BoundaryState
				if err := json.Unmarshal(encoded, &restored); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(state, restored) {
					t.Fatal("boundary checkpoint differs")
				}
				state = restored
			}
			closed := 0
			err = policy.Process(t.Context(), nil, &state, true, func(event speechactivity.Boundary) error {
				closed++
				if !event.Ended || event.End != uint64(len(fixture.Probabilities)) || event.Start == 0 {
					t.Fatal("invalid final boundary")
				}
				return nil
			})
			if err != nil || !state.Final || closed > 1 {
				t.Fatalf("finalization: %v", err)
			}
		})
	}
}

func TestVADOfflineBoundaryParity(t *testing.T) {
	t.Parallel()
	for _, fixture := range vadBoundaryCases(t) {
		t.Run(fixture.Name, func(t *testing.T) {
			c := fixture.Config
			config := speechactivity.OfflineConfig{Smoothing: c.Smoothing, Threshold: c.Threshold, MinSpeech: int(c.MinSpeech), MaxSpeech: int(c.MaxSpeech), MinSilence: int(c.MinSilence), MergeSilence: fixture.MergeSilence, ExtendSpeech: fixture.ExtendSpeech}
			got, err := speechactivity.OfflineDecisions(t.Context(), fixture.Probabilities, config, vadReferenceBytes)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(fixture.Offline) {
				t.Fatal("offline frame count differs")
			}
			for frame, value := range got {
				if value != (fixture.Offline[frame] != 0) {
					t.Fatalf("offline frame %d differs: %v want %d", frame, value, fixture.Offline[frame])
				}
			}
		})
	}
}
