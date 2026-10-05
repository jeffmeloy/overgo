package loratrain

import (
	"maps"
	"math"
	"math/rand"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/peft"
)

// factorGradients are the gradients of the loss on the periodic window with
// respect to each factor of the adapter the config starts, its up factors made
// nonzero so both factors have one, as the finite-difference check makes them.
func factorGradients(t *testing.T, config Config) map[string][]float32 {
	t.Helper()
	model, _, _ := tinyModel(t)
	trainer, err := New(model, config)
	if err != nil {
		t.Fatal(err)
	}
	random := rand.New(rand.NewSource(2))
	for _, name := range slices.Sorted(maps.Keys(trainer.adapter.Modules)) {
		up := trainer.adapter.Modules[name].Up
		for index := range up {
			up[index] = float32(random.NormFloat64()) / 4
		}
	}
	if err := trainer.apply(); err != nil {
		t.Fatal(err)
	}
	_, _, grads, err := model.LossAndGrads(periodic)
	if err != nil {
		t.Fatal(err)
	}
	if err := projectAll(trainer, grads); err != nil {
		t.Fatal(err)
	}
	copied := map[string][]float32{}
	for name, values := range trainer.gradients {
		copied[name] = slices.Clone(values)
	}
	return copied
}

// TestRSLoRAAndLoRAPlus checks the two scalings an adapter can be asked for:
// rank-stabilized LoRA scales each module by alpha over the root of its rank,
// and LoRA+ makes the up factor learn the given number of times faster by
// training it over that rate under a scale that carries it; both give factor
// gradients that match central differences, both survive a PEFT round trip, and
// a rate below one is refused.
func TestRSLoRAAndLoRAPlus(t *testing.T) {
	t.Run("rank-stabilized scale", func(t *testing.T) {
		model, _, _ := tinyModel(t)
		for _, test := range []struct {
			rs   bool
			want float64
		}{{false, 8.0 / 4}, {true, 8.0 / 2}} {
			trainer, err := New(model, Config{Rank: 4, Alpha: 8, Seed: 1, RSLoRA: test.rs})
			if err != nil {
				t.Fatal(err)
			}
			for name := range trainer.Adapter().Modules {
				if scale, err := trainer.Adapter().Scale(name); err != nil || math.Abs(scale-test.want) > 1e-12 {
					t.Fatalf("rslora=%t: %s scale %v (%v), want %v", test.rs, name, scale, err, test.want)
				}
			}
		}
	})

	t.Run("gradients are the plain adapter's at the same scale", func(t *testing.T) {
		// Alphas chosen so each module trains at the scale of 3 the plain adapter does: its gradients are
		// checked against central differences by TestFactorGradientsMatchFiniteDifferences, and a
		// variant whose scale reaches the projection correctly has exactly those gradients.
		plain := factorGradients(t, Config{Rank: 2, Alpha: 6, Seed: 1})
		for name, config := range map[string]Config{
			"rank-stabilized": {Rank: 2, Alpha: 3 * math.Sqrt2, Seed: 1, RSLoRA: true},
			"LoRA+":           {Rank: 2, Alpha: 6.0 / 4, Seed: 1, UpRate: 4},
		} {
			for factor, want := range plain {
				got := factorGradients(t, config)[factor]
				for index, value := range want {
					if math.Abs(float64(got[index]-value)) > 1e-5+1e-4*math.Abs(float64(value)) {
						t.Fatalf("%s: %s[%d] gradient %v, the plain adapter's %v", name, factor, index, got[index], value)
					}
				}
			}
		}
	})

	t.Run("the up factor's first step is the rate times longer", func(t *testing.T) {
		const rate = 4
		deltas := make([]map[string][]float32, 2)
		for index, up := range []float64{0, rate} {
			model, _, _ := tinyModel(t)
			trainer, err := New(model, Config{Rank: 2, Alpha: 6, Seed: 1, UpRate: up})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := trainer.Step(periodic, nil, 0.02, 0.9); err != nil {
				t.Fatal(err)
			}
			deltas[index] = map[string][]float32{}
			for name := range trainer.Adapter().Modules {
				if deltas[index][name], err = trainer.Adapter().Delta(name); err != nil {
					t.Fatal(err)
				}
			}
		}
		compared := 0
		for name, plain := range deltas[0] {
			for position, value := range plain {
				if got, want := float64(deltas[1][name][position]), rate*float64(value); math.Abs(got-want) > 1e-6+1e-4*math.Abs(want) {
					t.Fatalf("%s[%d]: LoRA+ moved the weight by %v, %d times plain's %v is %v", name, position, got, rate, value, want)
				}
				if value != 0 {
					compared++
				}
			}
		}
		if compared == 0 {
			t.Fatal("the first step moved no weight: the comparison proves nothing")
		}
	})

	t.Run("a saved adapter says how it was scaled", func(t *testing.T) {
		model, _, _ := tinyModel(t)
		trainer, err := New(model, Config{Rank: 2, Alpha: 6, Seed: 1, RSLoRA: true, UpRate: 4})
		if err != nil {
			t.Fatal(err)
		}
		directory := filepath.Join(t.TempDir(), "adapter")
		if err := peft.Save(directory, trainer.Adapter()); err != nil {
			t.Fatal(err)
		}
		loaded, err := peft.Load(directory)
		if err != nil {
			t.Fatal(err)
		}
		if !loaded.Config.RSLoRA || loaded.Config.Alpha != 24 {
			t.Fatalf("loaded adapter is rslora=%t with alpha %v, want rank-stabilized at alpha 6 times the rate of 4", loaded.Config.RSLoRA, loaded.Config.Alpha)
		}
	})

	t.Run("a rate below one is refused", func(t *testing.T) {
		model, _, _ := tinyModel(t)
		for _, rate := range []float64{0.5, -1, math.NaN(), math.Inf(1)} {
			if _, err := New(model, Config{Rank: 2, Alpha: 4, UpRate: rate}); err == nil {
				t.Errorf("an up-factor rate of %v was accepted", rate)
			}
		}
	})
}
