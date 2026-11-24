package media

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"overgo/internal/testcheck"
)

// TestTheQwenImageGridIsTakenWhereItKeepsItsPhase holds krea-degrid's finish to its measure: a grid of 0.3 of a level
// whose phase turns over between the halves of a smooth picture is taken from each half, where one pattern for the
// whole picture would have inverted one of them; a grid-free texture of single-pixel noise keeps its 1-pixel steps
// within 1%; no value moves more than 2 levels; the pixels given are not changed; and a size that is not theirs is
// refused.
func TestTheQwenImageGridIsTakenWhereItKeepsItsPhase(t *testing.T) {
	t.Parallel()
	const side, level = 96, 1.0 / 255
	// halfGrid is the grid's signed strength over one half's interior, columns and rows, read off one channel.
	halfGrid := func(pixels []float32, left int) (columns, rows float64) {
		count := 0.0
		for row := 16; row < side-16; row++ {
			for column := left + 12; column < left+side/2-12; column++ {
				value := float64(pixels[(row*side+column)*RGBChannels])
				columns += value * parity(column)
				rows += value * parity(row)
				count++
			}
		}
		return columns / count / level, rows / count / level
	}
	// Each half is flat, its own grey, as the flat areas the grid shows in are: a ramp read against alternating signs
	// reads half its slope as a grid.
	gridded, ramp := make([]float32, side*side*RGBChannels), make([]float32, side*side*RGBChannels)
	for row := range side {
		for column := range side {
			sign, flat := 1.0, 0.4
			if column >= side/2 {
				sign, flat = -1, 0.6
			}
			for channel := range RGBChannels {
				gridded[(row*side+column)*RGBChannels+channel] = float32(flat + sign*0.3*level*(parity(column)+parity(row))/2)
				ramp[(row*side+column)*RGBChannels+channel] = float32(0.3 + 0.4*float64(row+column)/(2*side))
			}
		}
	}
	// A ramp is left as it is, but at the border, where the mirror bends it and the finish reads a little of the bend.
	smoothed := testcheck.Must(RemoveQwenImageGrid(ramp, side, side))(t)
	for index := range ramp {
		row, column := index/RGBChannels/side, index/RGBChannels%side
		bound := 0.1 * level
		if inner := qwenImageGridTaps + qwenImageGridRadius; row >= inner && row < side-inner && column >= inner && column < side-inner {
			bound = 0.001 * level
		}
		if moved := math.Abs(float64(smoothed[index] - ramp[index])); moved > bound {
			t.Fatalf("a smooth ramp moved %.4f levels at row %d, column %d; want under %.3f", moved/level, row, column, bound/level)
		}
	}
	before := slices.Clone(gridded)
	finished := testcheck.Must(RemoveQwenImageGrid(gridded, side, side))(t)
	if !slices.Equal(gridded, before) {
		t.Fatal("the finish changed the pixels it was given")
	}
	for _, left := range []int{0, side / 2} {
		if columns, rows := halfGrid(gridded, left); math.Abs(columns) < 0.1 || math.Abs(rows) < 0.1 {
			t.Fatalf("the half at %d holds a grid of %.3f and %.3f levels; want the 0.15 it was drawn with", left, columns, rows)
		}
		if columns, rows := halfGrid(finished, left); math.Abs(columns) > 0.01 || math.Abs(rows) > 0.01 {
			t.Errorf("the half at %d keeps a grid of %.3f and %.3f levels; want none", left, columns, rows)
		}
	}
	random := rand.New(rand.NewPCG(1, 2))
	texture := make([]float32, side*side*RGBChannels)
	for index := range texture {
		texture[index] = float32(0.5 + 20*level*(random.Float64()*2-1))
	}
	textured := testcheck.Must(RemoveQwenImageGrid(texture, side, side))(t)
	step := func(pixels []float32) float64 {
		sum := 0.0
		for row := range side {
			for column := range side - 1 {
				d := float64(pixels[(row*side+column+1)*RGBChannels] - pixels[(row*side+column)*RGBChannels])
				sum += d * d
			}
		}
		return math.Sqrt(sum)
	}
	if loss := 1 - step(textured)/step(texture); math.Abs(loss) > 0.01 {
		t.Errorf("the finish moved a grid-free texture's 1-pixel steps by %.2f%%; want under 1%%", 100*loss)
	}
	for _, pair := range [][2][]float32{{gridded, finished}, {texture, textured}} {
		for index := range pair[0] {
			if moved := math.Abs(float64(pair[1][index] - pair[0][index])); moved > 2*level+1e-6 {
				t.Fatalf("a value moved %.3f levels; want 2 at most", moved/level)
			}
		}
	}
	if _, err := RemoveQwenImageGrid(gridded, side, side+1); err == nil {
		t.Fatal("pixels of another size were finished")
	}
}
