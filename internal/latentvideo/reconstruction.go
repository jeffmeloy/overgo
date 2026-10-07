package latentvideo

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"image/color"
	// A picture arrives as a PNG or a JPEG.
	_ "image/jpeg"
	_ "image/png"
	"path/filepath"

	"overgo/internal/checked"
	"overgo/internal/media"
	"overgo/internal/tensor"
)

// AutoencoderWeights is the file a Wan release keeps its autoencoder in: the
// encoder and the decoder the generator, the editor and a round trip all use.
func AutoencoderWeights(directory string) string {
	return filepath.Join(directory, "Wan2.1_VAE.pth")
}

// ReconstructionRequest is one encoded picture, PNG or JPEG, to pass through
// the autoencoder: into its latent and back.
type ReconstructionRequest struct {
	Image []byte `json:"image"`
}

// ValidateReconstructionRequest refuses a request without a picture before
// any model is loaded.
func ValidateReconstructionRequest(request ReconstructionRequest) error {
	if len(request.Image) == 0 {
		return errors.New("latent video: a reconstruction needs an encoded picture")
	}
	return nil
}

// SessionKey names the one resident autoencoder: it takes a picture of any
// size, so every request shares it.
func (ReconstructionRequest) SessionKey() (string, error) { return "autoencoder", nil }

// declaredLatentStats reads the latent's mean and spread for each channel
// from the profile catalog, for an autoencoder with that many latent
// channels: the encoder normalizes by them and the decoder undoes it.
func declaredLatentStats(channels int) (VAELatentStats, error) {
	profile, err := declaredProfile()
	if err != nil {
		return VAELatentStats{}, err
	}
	if err := media.ValidateChannelMoments(profile.LatentStats.Mean, profile.LatentStats.Std, channels); err != nil {
		return VAELatentStats{}, fmt.Errorf("latent video: the declared latent statistics do not fit the autoencoder: %w", err)
	}
	return profile.LatentStats, nil
}

// sourcePicture decodes an encoded picture into one planar signed-unit frame
// and returns the picture's own height and width. The autoencoder halves each
// side a fixed number of times, so a side that is not a multiple of its
// stride, rows down and columns across, is extended by repeating the last
// row or column; cropPicture takes the extension off what comes back.
func sourcePicture(data []byte, rows, columns int) (SourceVideo, int, int, error) {
	decoded, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return SourceVideo{}, 0, 0, fmt.Errorf("latent video: decode the picture: %w", err)
	}
	bounds := decoded.Bounds()
	height, width := bounds.Dy(), bounds.Dx()
	if !checked.PositiveInts(height, width, rows, columns) {
		return SourceVideo{}, 0, 0, errors.New("latent video: the picture or the autoencoder's stride is empty")
	}
	source, err := SourceVideoFromFrames([]image.Image{extendedPicture{
		Image: decoded, extent: image.Rect(bounds.Min.X, bounds.Min.Y, bounds.Min.X+(width+columns-1)/columns*columns, bounds.Min.Y+(height+rows-1)/rows*rows),
	}})
	return source, height, width, err
}

// extendedPicture shows a picture at a larger extent by repeating its last
// row and column.
type extendedPicture struct {
	image.Image
	extent image.Rectangle
}

// Bounds is the larger extent.
func (picture extendedPicture) Bounds() image.Rectangle { return picture.extent }

// At is the picture's own pixel where it has one, and its nearest edge pixel
// in the extension.
func (picture extendedPicture) At(x, y int) color.Color {
	inner := picture.Image.Bounds()
	return picture.Image.At(min(x, inner.Max.X-tensor.SingletonExtent), min(y, inner.Max.Y-tensor.SingletonExtent))
}

// SourceVideoFromFrames lays decoded frames of one size out as the planar
// signed-unit clip an autoencoder takes: channel, then frame, then row.
func SourceVideoFromFrames(frames []image.Image) (SourceVideo, error) {
	first, ok := checked.First(frames)
	if !ok {
		return SourceVideo{}, errors.New("latent video: a clip has no frames")
	}
	height, width := first.Bounds().Dy(), first.Bounds().Dx()
	if !checked.PositiveInts(height, width) {
		return SourceVideo{}, errors.New("latent video: a clip's frames are empty")
	}
	plane := height * width
	pixels := make([]float32, media.RGBChannels*len(frames)*plane)
	for index, frame := range frames {
		bounds := frame.Bounds()
		if bounds.Dy() != height || bounds.Dx() != width {
			return SourceVideo{}, fmt.Errorf("latent video: frame %d is %dx%d in a clip of %dx%d", index, bounds.Dy(), bounds.Dx(), height, width)
		}
		for y := range height {
			for x := range width {
				red, green, blue, _ := frame.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
				for channel, value := range [media.RGBChannels]uint32{red, green, blue} {
					pixels[(channel*len(frames)+index)*plane+y*width+x] = media.U8ToSignedUnitF32(media.RGBAChannel8(value))
				}
			}
		}
	}
	return SourceVideo{Pixels: pixels, Channels: media.RGBChannels, Frames: len(frames), Height: height, Width: width}, nil
}

// cropPicture keeps the top-left height by width of one planar frame.
func cropPicture(frame SourceVideo, height, width int) []float32 {
	if height == frame.Height && width == frame.Width {
		return frame.Pixels
	}
	cropped := make([]float32, frame.Channels*height*width)
	for channel := range frame.Channels {
		for y := range height {
			from := (channel*frame.Height + y) * frame.Width
			copy(cropped[(channel*height+y)*width:][:width], frame.Pixels[from:from+width])
		}
	}
	return cropped
}
