package latentimage

import (
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/torchrng"
)

type samplingCase struct {
	Name        string
	Steps       int
	Guidance    float32
	Skip        []int
	Start, Stop float64
}

func samplingFixture(t *testing.T) (*safetensors.Source, []samplingCase) {
	t.Helper()
	mossFixtureBytes(t, "sampling.safetensors", "d333b4bfff71f54bbdf3719366888803f557c9d8cb271dc838f58ce0de75690a")
	s, e := safetensors.OpenFile(filepath.Join("..", "..", "fixtures", "moss", "sampling.safetensors"))
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() { s.Close() })
	var cases []samplingCase
	for _, meta := range s.Metadata {
		if e := json.Unmarshal([]byte(meta["cases"]), &cases); e != nil {
			t.Fatal(e)
		}
	}
	if len(cases) != 5 {
		t.Fatal("native cases missing")
	}
	return s, cases
}

func samplingSettings(t *testing.T, c samplingCase) rotarySamplingSpec {
	t.Helper()
	schedule, e := compileNearestGridFlowSchedule(c.Steps, 1000, 5, 1)
	if e != nil {
		t.Fatal(e)
	}
	return rotarySamplingSpec{schedule: schedule, guidance: c.Guidance, skip: c.Skip, start: c.Start, stop: c.Stop}
}

func samplingReferenceName(name string) string {
	if name == "initial_latents" {
		return "initial"
	}
	return strings.Replace(name, "latents.", "latent.", 1)
}

func TestMOSSSamplingGolden(t *testing.T) {
	oracle, cases := samplingFixture(t)
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			settings := samplingSettings(t, c)
			prefix := "cpu-f32." + c.Name + "."
			latent, e := torchrng.CPUNormal(42, 24)
			if e != nil {
				t.Fatal(e)
			}
			compareRefinement(t, latent, refinementValues(t, oracle, prefix+"initial"), 0)
			step, e := compileFlowGuidanceProgram(tensor.MustShape(6, 4), settings.guidance)
			if e != nil {
				t.Fatal(e)
			}
			predictionIndex := 0
			for i, time := range settings.schedule.Timesteps {
				forward := func(context []float32, skip []int) []float32 {
					p, e := compileRotaryDenoiserProgram(denoiserSpec(), 1e-6, 2, 3, len(context)/8, skip, dtype.F32)
					if e != nil {
						t.Fatal(e)
					}
					feeds := denoiserFeeds(t, p, 0, dtype.F32)
					dynamic, e := p.feeds(latent, context, float32(time))
					if e != nil {
						t.Fatal(e)
					}
					maps.Copy(feeds, dynamic)
					got, e := reference.Execute([]*tensor.Tensor{p.output}, feeds)
					if e != nil {
						t.Fatal(e)
					}
					values := got[p.output].Data
					compareRefinement(t, values, refinementValues(t, oracle, fmt.Sprintf("%sprediction.%d", prefix, predictionIndex)), 1.3e-6)
					predictionIndex++
					return values
				}
				positive := forward(refinementValues(t, oracle, prefix+"positive"), nil)
				feeds := map[*tensor.Tensor]reference.Value{step.latent: {Shape: step.latent.Shape, Data: latent}, step.positive: {Shape: step.positive.Shape, Data: positive}, step.delta: {Shape: step.delta.Shape, Data: []float32{float32(settings.schedule.Deltas[i])}}}
				if step.negative != nil {
					skip := []int(nil)
					if settings.skipped(i) {
						skip = settings.skip
					}
					negative := forward(refinementValues(t, oracle, prefix+"negative"), skip)
					feeds[step.negative] = reference.Value{Shape: step.negative.Shape, Data: negative}
				}
				got, e := reference.Execute([]*tensor.Tensor{step.velocity, step.next}, feeds)
				if e != nil {
					t.Fatal(e)
				}
				compareRefinement(t, got[step.velocity].Data, refinementValues(t, oracle, fmt.Sprintf("%svelocity.%d", prefix, i)), 1.3e-6)
				latent = got[step.next].Data
				compareRefinement(t, latent, refinementValues(t, oracle, fmt.Sprintf("%slatent.%d", prefix, i)), 1.3e-6)
			}
		})
	}
}

func TestMOSSSamplingRefusesInvalidSettings(t *testing.T) {
	valid := samplingSettings(t, samplingCase{Steps: 2, Guidance: 3, Stop: 1})
	for _, change := range []func(*rotarySamplingSpec){func(s *rotarySamplingSpec) { s.guidance = -1 }, func(s *rotarySamplingSpec) { s.guidance = float32(math.Inf(1)) }, func(s *rotarySamplingSpec) { s.start = math.NaN() }, func(s *rotarySamplingSpec) { s.start = 1; s.stop = 0 }, func(s *rotarySamplingSpec) { s.skip = []int{2} }, func(s *rotarySamplingSpec) { s.schedule.Steps = 0 }} {
		v := valid
		change(&v)
		if v.validate(2) == nil {
			t.Fatal("invalid sampling settings accepted")
		}
	}
}
