//go:build windows

package latentimage

import (
	"context"
	"errors"
	"path/filepath"

	"overgo/internal/tensor"
	"overgo/internal/vitencoder"
)

// pictureSections are what a request's pictures are read with on the card, each built the first time a request asks
// for it and held across a session's requests (moss-depth-card): the picture encoder at the canvas, which encodes the
// picture a sampling starts from and a depth control's map, and Depth Anything V2 Small, which maps a depth control's
// picture, as ComfyUI's VAE and preprocessor stay loaded between its runs. Building the encoder took 0.4 to 1 s a map,
// and loading the depth model 0.4 s, each request.
type pictureSections struct {
	root          string
	width, height int
	encoder       *PictureEncoder
	depth         *vitencoder.DepthCard
}

// newPictureSections are the sections of a canvas of width by height of the model at root, nothing built yet.
func newPictureSections(root string, width, height int) *pictureSections {
	return &pictureSections{root: root, width: width, height: height}
}

// sections are the generator's picture sections, at its request's canvas, which a session's requests share.
func (g *Generator) sections() *pictureSections {
	if g.pictures == nil {
		g.pictures = newPictureSections(g.modelDir, g.request.Width, g.request.Height)
	}
	return g.pictures
}

// pictureEncoder is the sections' picture encoder, built the first time it is asked for.
func (s *pictureSections) pictureEncoder(ctx context.Context) (*PictureEncoder, error) {
	if s.encoder == nil {
		encoder, err := NewPictureEncoder(ctx, s.root, s.height, s.width)
		if err != nil {
			return nil, err
		}
		s.encoder = encoder
	}
	return s.encoder, nil
}

// depthCard is the sections' depth model on the card, read from the root's checkpoint the first time it is asked for.
func (s *pictureSections) depthCard() (*vitencoder.DepthCard, error) {
	if s.depth == nil {
		model, err := vitencoder.LoadDepthAnything(filepath.Join(s.root, depthDirectory, depthCheckpoint))
		if err != nil {
			return nil, err
		}
		if s.depth, err = model.OpenCard(tensor.FirstOffset); err != nil {
			return nil, err
		}
	}
	return s.depth, nil
}

// close releases what the sections built.
func (s *pictureSections) close(ctx context.Context) error {
	if s == nil {
		return nil
	}
	var err error
	if s.encoder != nil {
		err = s.encoder.Close(ctx)
	}
	if s.depth != nil {
		err = errors.Join(err, s.depth.Close())
	}
	s.encoder, s.depth = nil, nil
	return err
}
