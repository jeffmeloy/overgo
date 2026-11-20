//go:build windows

package latentimage

import (
	"context"
	"errors"
	"path/filepath"

	"overgo/internal/graphruntime"
	"overgo/internal/tensor"
	"overgo/internal/vitencoder"
)

// pictureSections are what a request's pictures are read with on the card, each built the first time a request asks
// for it and held across a session's requests (moss-depth-card): the picture encoder at the canvas, which encodes the
// picture a sampling starts from and a depth control's map, and Depth Anything V2 Small, which maps a depth control's
// picture, as ComfyUI's VAE and preprocessor stay loaded between its runs. Building the encoder took 0.4 to 1 s a map,
// and loading the depth model 0.4 s, each request. Both run in the generator's own session where it lends one, so its
// device ledger counts what they hold beside its own (SQA-434); sections lent none open a session of their own.
type pictureSections struct {
	root          string
	width, height int
	// session is the session the sections run in, and owned whether they opened it.
	session *graphruntime.ResidentSession
	owned   bool
	encoder *PictureEncoder
	depth   *vitencoder.DepthCard
	// depthContent is the identity of the checkpoint bytes depth was read from: a checkpoint replaced under its name
	// is read again, and one unchanged is not (SQA-406).
	depthContent string
}

// newPictureSections are the sections of a canvas of width by height of the model at root, nothing built yet, run in
// session where it is given.
func newPictureSections(root string, width, height int, session *graphruntime.ResidentSession) *pictureSections {
	return &pictureSections{root: root, width: width, height: height, session: session}
}

// sections are the generator's picture sections, at its request's canvas, which a session's requests share, run in
// the rotary generation's session.
func (g *Generator) sections() *pictureSections {
	if g.pictures == nil {
		var session *graphruntime.ResidentSession
		if g.rotary != nil {
			session = g.rotary.runtime
		}
		g.pictures = newPictureSections(g.modelDir, g.request.Width, g.request.Height, session)
	}
	return g.pictures
}

// deviceSession is the session the sections run in, opened the first time it is asked for where none was lent.
func (s *pictureSections) deviceSession() (*graphruntime.ResidentSession, error) {
	if s.session == nil {
		session, err := graphruntime.NewResidentSession(tensor.FirstOffset)
		if err != nil {
			return nil, err
		}
		s.session, s.owned = session, true
	}
	return s.session, nil
}

// pictureEncoder is the sections' picture encoder, built the first time it is asked for.
func (s *pictureSections) pictureEncoder(ctx context.Context) (*PictureEncoder, error) {
	if s.encoder == nil {
		session, err := s.deviceSession()
		if err != nil {
			return nil, err
		}
		if s.encoder, err = newPictureEncoderIn(ctx, session, s.root, s.height, s.width); err != nil {
			return nil, err
		}
	}
	return s.encoder, nil
}

// depthCard is the sections' depth model on the card, read from the root's checkpoint where none is held or the one
// held was read from other bytes than the checkpoint holds now: the checkpoint is identified by its content each time
// a map is asked for, as an adapter is (fileContent), and a checkpoint that cannot be read refuses the map.
func (s *pictureSections) depthCard(ctx context.Context) (*vitencoder.DepthCard, error) {
	path := filepath.Join(s.root, depthDirectory, depthCheckpoint)
	content, err := fileContent(path)
	if err != nil {
		return nil, err
	}
	if s.depth != nil && content == s.depthContent {
		return s.depth, nil
	}
	if s.depth != nil {
		held := s.depth
		s.depth, s.depthContent = nil, ""
		if err := held.Close(ctx); err != nil {
			return nil, err
		}
	}
	model, err := vitencoder.LoadDepthAnything(path)
	if err != nil {
		return nil, err
	}
	session, err := s.deviceSession()
	if err != nil {
		return nil, err
	}
	s.depth, s.depthContent = model.OpenCard(session), content
	return s.depth, nil
}

// close releases what the sections built, and the session where they opened it.
func (s *pictureSections) close(ctx context.Context) error {
	if s == nil {
		return nil
	}
	var err error
	if s.encoder != nil {
		err = s.encoder.Close(ctx)
	}
	if s.depth != nil {
		err = errors.Join(err, s.depth.Close(ctx))
	}
	if s.owned {
		err = errors.Join(err, s.session.Close(ctx))
	}
	s.encoder, s.depth, s.depthContent, s.session, s.owned = nil, nil, "", nil, false
	return err
}
