package media

import (
	"cmp"
	"image"
	"image/color"
	"slices"

	"overgo/internal/tensor"
)

const (
	// paletteChannelBits is how finely a picture's colours are counted before
	// a palette is cut from them: five bits a channel, 32768 cells, each
	// remembering the sum of the full colours that fell in it, so a palette
	// colour is a mean of real colours and not a cell's corner.
	paletteChannelBits = 5
	// paletteCells is the count of those cells.
	paletteCells = 1 << (RGBChannels * paletteChannelBits)
)

// paletteBox is a set of colour cells a palette colour will stand for: the
// cells, how many pixels they hold, and the cell range they span on each
// channel.
type paletteBox struct {
	cells      []int
	population int
	low, high  [RGBChannels]int
}

// AdaptivePalette chooses up to colors colours for a picture by median cut:
// the picture's colours start as one box, and the box whose pixels spread
// furthest, counted by pixels times its longest side, is cut across that side
// where half its pixels lie to each hand, until there are colors boxes or no
// box holds more than one cell. Each box gives the mean colour of its pixels.
// A palette cut from the picture it will draw keeps a dithered frame far
// closer to the picture than a fixed palette does, and a picture of no more
// than colors colours keeps every one of them. The cut is deterministic.
func AdaptivePalette(picture *image.RGBA, colors int) color.Palette {
	counts := make([]int, paletteCells)
	sums := make([][RGBChannels]int, paletteCells)
	bounds := picture.Bounds()
	const drop = bitsPerByte - paletteChannelBits
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			at := picture.PixOffset(x, y)
			cell := 0
			for channel := range RGBChannels {
				cell = cell<<paletteChannelBits | int(picture.Pix[at+channel]>>drop)
			}
			counts[cell]++
			for channel := range RGBChannels {
				sums[cell][channel] += int(picture.Pix[at+channel])
			}
		}
	}
	coordinate := func(cell, channel int) int {
		return cell >> ((RGBChannels - 1 - channel) * paletteChannelBits) & (1<<paletteChannelBits - 1)
	}
	measure := func(cells []int) paletteBox {
		box := paletteBox{cells: cells}
		for index, cell := range cells {
			box.population += counts[cell]
			for channel := range RGBChannels {
				value := coordinate(cell, channel)
				if index == 0 || value < box.low[channel] {
					box.low[channel] = value
				}
				if index == 0 || value > box.high[channel] {
					box.high[channel] = value
				}
			}
		}
		return box
	}
	longest := func(box paletteBox) (channel, side int) {
		for candidate := range RGBChannels {
			if extent := box.high[candidate] - box.low[candidate]; extent > side {
				channel, side = candidate, extent
			}
		}
		return channel, side
	}
	var occupied []int
	for cell, count := range counts {
		if count > 0 {
			occupied = append(occupied, cell)
		}
	}
	if len(occupied) == 0 || colors <= 0 {
		return nil
	}
	boxes := []paletteBox{measure(occupied)}
	for len(boxes) < colors {
		chosen, weight := -1, 0
		for index, box := range boxes {
			if _, side := longest(box); len(box.cells) > 1 && box.population*max(side, 1) > weight {
				chosen, weight = index, box.population*max(side, 1)
			}
		}
		if chosen < 0 {
			break
		}
		box := boxes[chosen]
		channel, _ := longest(box)
		slices.SortStableFunc(box.cells, func(left, right int) int {
			return cmp.Or(cmp.Compare(coordinate(left, channel), coordinate(right, channel)), cmp.Compare(left, right))
		})
		// The cut falls after the cell that carries the box's middle pixel, and
		// never after the last cell: both halves hold at least one.
		cut, passed := 0, 0
		for cut < len(box.cells)-1 {
			passed += counts[box.cells[cut]]
			cut++
			if passed*tensor.PairedExtent >= box.population {
				break
			}
		}
		boxes[chosen] = measure(box.cells[:cut:cut])
		boxes = append(boxes, measure(box.cells[cut:]))
	}
	palette := make(color.Palette, len(boxes))
	for index, box := range boxes {
		var total [RGBChannels]int
		for _, cell := range box.cells {
			for channel := range RGBChannels {
				total[channel] += sums[cell][channel]
			}
		}
		mean := func(channel int) uint8 {
			return uint8((total[channel] + box.population/tensor.PairedExtent) / box.population)
		}
		palette[index] = color.RGBA{R: mean(tensor.FirstOffset), G: mean(tensor.SingletonExtent), B: mean(tensor.PairedExtent), A: ^uint8(tensor.FirstOffset)}
	}
	return palette
}
