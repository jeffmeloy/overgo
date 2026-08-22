package media

import (
	"fmt"
	"math"

	"overgo/internal/checked"
)

// SquarePatchExtent derives a square patch side from flattened vector width
// and channel count.
func SquarePatchExtent(vectorWidth, channels int) (int, error) {
	area, ok := checked.DivExactInt(vectorWidth, channels)
	if !ok {
		return 0, fmt.Errorf("media: patch vector is not channel-aligned")
	}
	patch := int(math.Sqrt(float64(area)))
	want, err := PatchVectorWidth(channels, patch)
	if err != nil || !checked.Equal(want, vectorWidth) {
		return 0, fmt.Errorf("media: patch vector does not describe a square patch")
	}
	return patch, nil
}

// PatchChannelOrder selects the per-patch element order.
type PatchChannelOrder uint8

const (
	PatchChannelsFirst PatchChannelOrder = iota
	PatchChannelsLast
)

// PatchGrid validates spatial patch divisibility and returns the patch grid.
func PatchGrid(height, width, patch int) (int, int, error) {
	if height <= 0 || width <= 0 || patch <= 0 || height%patch != 0 || width%patch != 0 {
		return 0, 0, fmt.Errorf("media: %dx%d is incompatible with patch %d", height, width, patch)
	}
	return height / patch, width / patch, nil
}

// PatchVectorWidth returns the flattened per-token width for planar patches.
func PatchVectorWidth(channels, patch int) (int, error) {
	patchArea, ok := checked.MulInt(patch, patch)
	if !ok || channels <= 0 || patch <= 0 {
		return 0, fmt.Errorf("media: invalid patch width channels=%d patch=%d", channels, patch)
	}
	width, ok := checked.MulInt(channels, patchArea)
	if !ok {
		return 0, fmt.Errorf("media: patch width overflows channels=%d patch=%d", channels, patch)
	}
	return width, nil
}

// PackPlanar packs planar [channels,height,width] values into row-major patch
// tokens and returns the patch-grid dimensions.
func PackPlanar[T ~float32 | ~float64](planar []T, channels, height, width, patch int, order PatchChannelOrder) ([]T, int, int, error) {
	if order != PatchChannelsFirst && order != PatchChannelsLast {
		return nil, 0, 0, fmt.Errorf("media: invalid patch channel order %d", order)
	}
	gridHeight, gridWidth, err := PatchGrid(height, width, patch)
	if err != nil {
		return nil, 0, 0, err
	}
	if len(planar) != channels*height*width {
		return nil, 0, 0, fmt.Errorf("media: planar length=%d want=%d", len(planar), channels*height*width)
	}
	inputChannels, err := PatchVectorWidth(channels, patch)
	if err != nil {
		return nil, 0, 0, err
	}
	output := make([]T, gridHeight*gridWidth*inputChannels)
	for row := range gridHeight {
		for column := range gridWidth {
			token := output[(row*gridWidth+column)*inputChannels : (row*gridWidth+column+1)*inputChannels]
			if order == PatchChannelsFirst {
				for channel := range channels {
					for patchRow := range patch {
						for patchColumn := range patch {
							token[(channel*patch+patchRow)*patch+patchColumn] = planar[(channel*height+row*patch+patchRow)*width+column*patch+patchColumn]
						}
					}
				}
			} else {
				for patchRow := range patch {
					for patchColumn := range patch {
						for channel := range channels {
							token[(patchRow*patch+patchColumn)*channels+channel] = planar[(channel*height+row*patch+patchRow)*width+column*patch+patchColumn]
						}
					}
				}
			}
		}
	}
	return output, gridHeight, gridWidth, nil
}

// UnpackPlanar reverses PackPlanar.
func UnpackPlanar[T ~float32 | ~float64](patches []T, channels, gridHeight, gridWidth, patch int, order PatchChannelOrder) ([]T, error) {
	if order != PatchChannelsFirst && order != PatchChannelsLast {
		return nil, fmt.Errorf("media: invalid patch channel order %d", order)
	}
	inputChannels, err := PatchVectorWidth(channels, patch)
	if err != nil {
		return nil, err
	}
	if len(patches) != gridHeight*gridWidth*inputChannels {
		return nil, fmt.Errorf("media: patch length=%d want=%d", len(patches), gridHeight*gridWidth*inputChannels)
	}
	height, width := gridHeight*patch, gridWidth*patch
	output := make([]T, channels*height*width)
	for row := range gridHeight {
		for column := range gridWidth {
			token := patches[(row*gridWidth+column)*inputChannels : (row*gridWidth+column+1)*inputChannels]
			if order == PatchChannelsFirst {
				for channel := range channels {
					for patchRow := range patch {
						for patchColumn := range patch {
							output[(channel*height+row*patch+patchRow)*width+column*patch+patchColumn] = token[(channel*patch+patchRow)*patch+patchColumn]
						}
					}
				}
			} else {
				for patchRow := range patch {
					for patchColumn := range patch {
						for channel := range channels {
							output[(channel*height+row*patch+patchRow)*width+column*patch+patchColumn] = token[(patchRow*patch+patchColumn)*channels+channel]
						}
					}
				}
			}
		}
	}
	return output, nil
}
