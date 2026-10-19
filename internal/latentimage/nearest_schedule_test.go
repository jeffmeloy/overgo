package latentimage

import (
	"fmt"
	"math"
	"testing"
)

func TestMOSSNearestGridScheduleReference(t *testing.T) {
	t.Parallel()
	source := mossGolden(t, "primitives")
	contract := mossContract(t)
	reference := contract.Cases[0]
	for _, candidate := range contract.Cases {
		if candidate.Name == "primitives" {
			reference = candidate
		}
	}
	for _, steps := range []int{1, 2, 50, 1001} {
		t.Run(fmt.Sprint(steps), func(t *testing.T) {
			schedule, err := compileNearestGridFlowSchedule(steps, 1000, 5, 1)
			if err != nil {
				t.Fatal(err)
			}
			for suffix, values := range map[string][]float64{"sigmas": schedule.Sigmas, "timesteps": schedule.Timesteps} {
				name := fmt.Sprintf("schedule.%d.%s", steps, suffix)
				expected := mossF32(t, source, name)
				actual := make([]float32, len(values))
				for i, v := range values {
					actual[i] = float32(v)
				}
				if err := contract.compare("primitives", reference.Artifact, reference.Backend, name, "F32", []uint64{uint64(len(values))}, actual, expected); err != nil {
					t.Fatal(err)
				}
			}
			if schedule.Steps != len(schedule.Timesteps) || len(schedule.Sigmas) != schedule.Steps+1 || len(schedule.Deltas) != schedule.Steps {
				t.Fatal("inconsistent schedule extents")
			}
			sample := mossF32(t, source, fmt.Sprintf("schedule.%d.initial", steps))
			prediction := mossF32(t, source, fmt.Sprintf("schedule.%d.prediction", steps))
			for i, delta := range schedule.Deltas {
				for j := range sample {
					sample[j] += float32(float32(delta) * prediction[j])
				}
				if i != 0 && i != schedule.Steps-1 {
					continue
				}
				name := fmt.Sprintf("schedule.%d.latent.%d", steps, i)
				expected := mossF32(t, source, name)
				if err := contract.compare("primitives", reference.Artifact, reference.Backend, name, "F32", source.Tensors[name].Shape, sample, expected); err != nil {
					t.Fatal(err)
				}
			}
			if steps == 1001 && schedule.Steps != 616 {
				t.Fatalf("deduplicated count=%d want 616", schedule.Steps)
			}
		})
	}
	for _, settings := range []struct {
		steps, total    int
		shift, training float64
	}{{0, 1000, 5, 1}, {1, 0, 5, 1}, {1, 1000, 0, 1}, {1, 1000, 5, -1}, {1, 1000, math.Inf(1), 1}, {1, 1000, math.NaN(), 1}} {
		if _, err := compileNearestGridFlowSchedule(settings.steps, settings.total, settings.shift, settings.training); err == nil {
			t.Fatal("invalid schedule accepted")
		}
	}
}
