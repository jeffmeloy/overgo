package projector

import (
	"context"
	"errors"
	"image"

	"overgo/internal/tensor/reference"
)

// encodedProjector is a projector whose images another program encodes, as a composite's image encoder and bridge do:
// it holds no file or device resource of its own.
type encodedProjector struct {
	prompt promptDispatch
}

// Close releases nothing: the encoder and the text model are their owners'.
func (*encodedProjector) Close() error { return nil }

func (p *encodedProjector) setPrompt(dispatch promptDispatch) { p.prompt = dispatch }

func (p *encodedProjector) compiledPrompt() promptDispatch { return p.prompt }

// NewEncodedImageSession is a session whose images encode turns into rows of the text model's width, a placeholder
// token for each row, under family's name: the placeholder path every projector's prompt takes, so a composite's rows
// reach the text model where a native projector's would.
func NewEncodedImageSession(
	family, placeholder string,
	width int,
	encode func(context.Context, image.Image) (reference.Value, error),
) (Session, error) {
	if family == "" || placeholder == "" || width <= 0 || encode == nil {
		return nil, errors.New("projector: an encoded image session needs a family, a placeholder, a width and an encoder")
	}
	source := &encodedProjector{}
	source.setPrompt(promptPrograms(func(*encodedProjector) compiledImagePromptProgram {
		return compileDelimitedImagePromptProgram(family, placeholder, width, referenceImageEncoder(encode))
	}, nil)(source))
	return NewSession(source)
}
