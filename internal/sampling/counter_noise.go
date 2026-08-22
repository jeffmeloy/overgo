package sampling

import (
	"fmt"
	"math"
)

const (
	counterNoiseBlock  = 256
	counterNoiseUnroll = 4
)

// CounterNoisePlan is the launch geometry for a counter-based normal source.
type CounterNoisePlan struct {
	Seed   uint64
	Offset uint64
	Grid   int
	Block  int
	Unroll int
}

// CompileCounterNoisePlan derives the reference launch from an extent and a
// measured device occupancy profile.
func CompileCounterNoisePlan(seed, offset uint64, elements, multiprocessors, threadsPerMultiprocessor int) (CounterNoisePlan, uint64, error) {
	if elements <= 0 || multiprocessors <= 0 || threadsPerMultiprocessor < counterNoiseBlock {
		return CounterNoisePlan{}, 0, fmt.Errorf("counter noise extent or device profile is invalid")
	}
	grid := min((elements+counterNoiseBlock-1)/counterNoiseBlock, multiprocessors*(threadsPerMultiprocessor/counterNoiseBlock))
	grid = max(grid, 1)
	stride := counterNoiseBlock * grid * counterNoiseUnroll
	advance := uint64((elements+stride-1)/stride) * counterNoiseUnroll
	return CounterNoisePlan{
		Seed: seed, Offset: offset, Grid: grid, Block: counterNoiseBlock, Unroll: counterNoiseUnroll,
	}, advance, nil
}

// FillCounterNormalNoise fills dst with Philox 4x32-10 / Box-Muller standard
// normals. Thread idx owns subsequence idx; each generation yields four
// normals distributed with a block*grid stride.
func FillCounterNormalNoise(dst []float32, plan CounterNoisePlan) error {
	if plan.Grid <= 0 || plan.Block <= 0 || plan.Unroll != counterNoiseUnroll {
		return fmt.Errorf("counter noise plan grid=%d block=%d unroll=%d is invalid", plan.Grid, plan.Block, plan.Unroll)
	}
	numel := int64(len(dst))
	span := int64(plan.Block * plan.Grid)
	stride := span * int64(plan.Unroll)
	rounded := ((numel-1)/stride + 1) * stride
	for idx := int64(0); idx < span; idx++ {
		counter := philoxInit(plan.Seed, uint64(idx), plan.Offset)
		for linear := idx; linear < rounded; linear += stride {
			x, y, z, w := philox4x32x10(counter, plan.Seed)
			counter = philoxIncrement(counter)
			n0, n1 := boxMullerNormalPair(x, y)
			n2, n3 := boxMullerNormalPair(z, w)
			for lane, value := range [4]float32{n0, n1, n2, n3} {
				linearIndex := linear + int64(lane)*span
				if linearIndex < numel {
					dst[linearIndex] = value
				}
			}
		}
	}
	return nil
}

type philoxCounter struct{ x, y, z, w uint32 }

func philoxInit(seed, subsequence, offset uint64) philoxCounter {
	counter := philoxCounter{
		z: uint32(subsequence),
		w: uint32(subsequence >> 32),
	}
	advance := offset / 4
	counter.x = uint32(advance)
	counter.y = uint32(advance >> 32)
	return counter
}

func philoxIncrement(counter philoxCounter) philoxCounter {
	counter.x++
	if counter.x == 0 {
		counter.y++
		if counter.y == 0 {
			counter.z++
			if counter.z == 0 {
				counter.w++
			}
		}
	}
	return counter
}

func philox4x32x10(counter philoxCounter, seed uint64) (uint32, uint32, uint32, uint32) {
	const (
		multiplierA = 0xD2511F53
		multiplierB = 0xCD9E8D57
		bumpA       = 0x9E3779B9
		bumpB       = 0xBB67AE85
	)
	keyA, keyB := uint32(seed), uint32(seed>>32)
	for round := 0; round < 10; round++ {
		productA := uint64(multiplierA) * uint64(counter.x)
		productB := uint64(multiplierB) * uint64(counter.z)
		hiA, loA := uint32(productA>>32), uint32(productA)
		hiB, loB := uint32(productB>>32), uint32(productB)
		counter = philoxCounter{
			x: hiB ^ counter.y ^ keyA,
			y: loB,
			z: hiA ^ counter.w ^ keyB,
			w: loA,
		}
		keyA += bumpA
		keyB += bumpB
	}
	return counter.x, counter.y, counter.z, counter.w
}

const (
	uniformScale  = float32(2.3283064e-10)
	uniformOffset = uniformScale / 2
	angularScale  = float32(2.3283064e-10 * 6.2831855)
	angularOffset = angularScale / 2
)

func boxMullerNormalPair(x, y uint32) (float32, float32) {
	u := float32(x)*uniformScale + uniformOffset
	v := float32(y)*angularScale + angularOffset
	scale := float32(math.Sqrt(float64(float32(-2) * float32(math.Log(float64(u))))))
	return scale * float32(math.Sin(float64(v))), scale * float32(math.Cos(float64(v)))
}
