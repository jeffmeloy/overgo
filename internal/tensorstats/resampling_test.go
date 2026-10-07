package tensorstats

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// resamplingFixture is a plan over five units in two strata, three in the
// first and two in the second, with unequal member counts.
func resamplingFixture(t *testing.T) ResamplingPlan {
	t.Helper()
	value := ResamplingPlan{
		Population: testutil.ArtifactID(t, artifact.KindDataset, "declared source population"),
		Assumption: testutil.ArtifactID(t, artifact.KindProfile, "independent units exchangeable within strata"),
		Evidence:   testutil.ArtifactID(t, artifact.KindEvidence, "grouping and sampling-model evidence"),
		Method:     BlockBootstrap, SeedHigh: 10, SeedLow: 20, Draws: 6,
	}
	for index, count := range []int{1, 3, 1, 2, 1} {
		stratum := "first stratum"
		if index >= 3 {
			stratum = "second stratum"
		}
		unit := SamplingUnit{
			ID:      testutil.ArtifactID(t, artifact.KindEvidence, fmt.Sprintf("source event %d", index)),
			Stratum: testutil.ArtifactID(t, artifact.KindProfile, stratum),
		}
		for member := range count {
			unit.Members = append(unit.Members, testutil.ArtifactID(t, artifact.KindEvidence, fmt.Sprintf("event %d ordered member %d", index, member)))
		}
		value.Units = append(value.Units, unit)
	}
	plan, err := NewResamplingPlan(value)
	if err != nil {
		t.Fatal(err)
	}
	return plan
}

// TestBlockBootstrapDrawsWholeUnitsWithinStrata holds the ported block
// resampling to its contract: a plan is canonical and commits with every
// authority it names; its draws are reproducible from its seeds; each draw
// fills every stratum's slots with whole units of that stratum, repeats
// allowed; and a plan with a stray unit, a shared member or no draws is
// refused.
func TestBlockBootstrapDrawsWholeUnitsWithinStrata(t *testing.T) {
	t.Parallel()
	plan := resamplingFixture(t)
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	batch, err := plan.Batch("resampling/plan")
	if err != nil {
		t.Fatal(err)
	}
	for _, edge := range batch.Lineage {
		batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: edge.Parent})
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	if held, err := store.HasContent(t.Context(), plan.ID); err != nil || !held {
		t.Fatalf("committed plan held %t (%v)", held, err)
	}

	strata := map[artifact.ID]artifact.ID{}
	for _, unit := range plan.Units {
		strata[unit.ID] = unit.Stratum
	}
	collect := func() [][]artifact.ID {
		var draws [][]artifact.ID
		err := ResampleBlocks(t.Context(), plan, func(draw BlockDraw) error {
			if draw.Plan != plan.ID || draw.Ordinal != uint64(len(draws)) || len(draw.Units) != len(plan.Units) {
				t.Fatalf("draw %d of plan %s holds %d units", draw.Ordinal, draw.Plan, len(draw.Units))
			}
			for slot, unit := range draw.Units {
				if strata[unit] != plan.Units[slot].Stratum {
					t.Fatalf("draw %d put a unit of another stratum in slot %d", draw.Ordinal, slot)
				}
			}
			draws = append(draws, draw.Units)
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		return draws
	}
	first, second := collect(), collect()
	if uint64(len(first)) != plan.Draws || !slices.EqualFunc(first, second, slices.Equal[[]artifact.ID]) {
		t.Fatalf("%d draws, repeated %d: the seeds do not reproduce them", len(first), len(second))
	}
	repeated := slices.ContainsFunc(first, func(units []artifact.ID) bool {
		return len(slices.Compact(slices.SortedFunc(slices.Values(units), artifact.CompareID))) < len(units)
	})
	if !repeated {
		t.Fatal("six bootstrap draws of five units never repeated one: they were not drawn with replacement")
	}
	stop := errors.New("stop")
	visited := 0
	if err := ResampleBlocks(t.Context(), plan, func(BlockDraw) error { visited++; return stop }); !errors.Is(err, stop) || visited != 1 {
		t.Fatalf("a consumer's error = %v after %d draws, want it to stop the stream", err, visited)
	}

	for name, mutate := range map[string]func(*ResamplingPlan){
		"no draws":          func(value *ResamplingPlan) { value.Draws = 0 },
		"a repeated unit":   func(value *ResamplingPlan) { value.Units[1].ID = value.Units[0].ID },
		"a shared member":   func(value *ResamplingPlan) { value.Units[1].Members[0] = value.Units[0].Members[0] },
		"an unnamed method": func(value *ResamplingPlan) { value.Method = "" },
	} {
		invalid := cloneResamplingPlan(plan)
		mutate(&invalid)
		if _, err := NewResamplingPlan(invalid); err == nil {
			t.Fatalf("a plan with %s was admitted", name)
		}
	}
}

// TestSymmetricEigenReturnsOrderedOrthonormalVectors holds the accumulated
// Jacobi sweep to what an eigendecomposition is: values descending, vectors
// of unit length and orthogonal, and each vector scaled by its value under
// the matrix.
func TestSymmetricEigenReturnsOrderedOrthonormalVectors(t *testing.T) {
	t.Parallel()
	const k = 4
	matrix := []float64{
		4, 1, 0.5, 0,
		1, 3, 0.2, 0.1,
		0.5, 0.2, 2, 0.3,
		0, 0.1, 0.3, 1,
	}
	values, vectors, ok := SymmetricEigen(matrix, k)
	if !ok || !slices.IsSortedFunc(values, func(left, right float64) int { return int(math.Copysign(1, right-left)) }) {
		t.Fatalf("eigenvalues %v (ok %t), want them descending", values, ok)
	}
	const tolerance = 1e-12
	for column := range k {
		for other := range k {
			var dot float64
			for row := range k {
				dot += vectors[row*k+column] * vectors[row*k+other]
			}
			want := 0.0
			if column == other {
				want = 1
			}
			if math.Abs(dot-want) > tolerance {
				t.Fatalf("vectors %d and %d have inner product %v, want %v", column, other, dot, want)
			}
		}
		for row := range k {
			var image float64
			for inner := range k {
				image += matrix[row*k+inner] * vectors[inner*k+column]
			}
			if math.Abs(image-values[column]*vectors[row*k+column]) > tolerance {
				t.Fatalf("vector %d is not scaled by its value %v under the matrix", column, values[column])
			}
		}
	}
	unordered := symmetricEigenvalues(matrix, k)
	slices.Sort(unordered)
	slices.Reverse(unordered)
	for index, value := range unordered {
		if math.Abs(value-values[index]) > tolerance {
			t.Fatalf("the values-only sweep reads %v where the accumulated one reads %v", unordered, values)
		}
	}
	if _, _, ok := SymmetricEigen(matrix, k+1); ok {
		t.Fatal("a matrix of the wrong extent was decomposed")
	}
}
