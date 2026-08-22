package hostmath

import (
	"errors"

	"overgo/internal/checked"
)

// Downsample2DChannelsF64Into applies a channel-mixing 3x3 stride-2
// convolution independently to each frame with float64 accumulation.
func Downsample2DChannelsF64Into(out, input, weight, bias []float32, channels, frames, height, width int) error {
	if !checked.PositiveInts(channels, frames) || !checked.EvenInt(height) || !checked.EvenInt(width) {
		return errors.New("hostmath: invalid downsample geometry")
	}
	if err := checked.Length(input, channels, frames, height, width); err != nil {
		return errors.New("hostmath: invalid downsample input")
	}
	if err := checked.Length(weight, channels, channels, 3, 3); err != nil {
		return errors.New("hostmath: invalid downsample weights")
	}
	if err := checked.Length(bias, channels); err != nil {
		return errors.New("hostmath: invalid downsample bias")
	}
	if err := checked.Length(out, channels, frames, height/2, width/2); err != nil {
		return errors.New("hostmath: invalid downsample output")
	}
	outHeight, outWidth := height/2, width/2
	for outputChannel := range channels {
		for frame := range frames {
			for outputY := range outHeight {
				for outputX := range outWidth {
					sum := float64(bias[outputChannel])
					for inputChannel := range channels {
						for kernelY := range 3 {
							inputY := 2*outputY + kernelY
							if inputY >= height {
								continue
							}
							for kernelX := range 3 {
								inputX := 2*outputX + kernelX
								if inputX >= width {
									continue
								}
								inputIndex := ((inputChannel*frames+frame)*height+inputY)*width + inputX
								weightIndex := ((outputChannel*channels+inputChannel)*3+kernelY)*3 + kernelX
								sum += float64(input[inputIndex]) * float64(weight[weightIndex])
							}
						}
					}
					out[((outputChannel*frames+frame)*outHeight+outputY)*outWidth+outputX] = float32(sum)
				}
			}
		}
	}
	return nil
}
