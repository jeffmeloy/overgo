package projector

import (
	"context"
	"errors"
	"image"

	"overgo/internal/artifact"
)

// SessionCapabilities: the media prompt surfaces a compiled projection
// session offers. Compiled once at session construction.
type SessionCapabilities struct {
	Image        bool
	MultiImage   bool
	Video        bool
	Audio        bool
	MediaHistory bool
}

// Session: the compiled projection session — the sole prompt authority over an
// opened projector artifact. Families declare unexported prompt providers;
// callers never touch family-specific entry points.
type Session interface {
	BuildImagePrompt(context.Context, ImageTokenizer, image.Image, string, string, bool) (MultimodalPrompt, error)
	BuildImagesPrompt(context.Context, ImageTokenizer, []image.Image, []string, PromptOptions) (MultimodalPrompt, error)
	BuildVideoPrompt(context.Context, ImageTokenizer, []image.Image, string, string, float64, bool) (MultimodalPrompt, error)
	BuildAudioPrompt(context.Context, ImageTokenizer, []float32, string, string) (MultimodalPrompt, error)
	BuildMediaHistoryPrompt(context.Context, ImageTokenizer, []MediaInput, []string) (MultimodalPrompt, error)
	Capabilities() SessionCapabilities
	Close() error
}

// Family prompt providers. Unexported by design: the session is the only
// public route to prompt construction.
type imagesPromptProvider interface {
	imagesPrompt(context.Context, ImageTokenizer, []image.Image, []string, PromptOptions) (MultimodalPrompt, error)
}

type videoPromptProvider interface {
	videoPrompt(context.Context, ImageTokenizer, []image.Image, string, string, float64, bool) (MultimodalPrompt, error)
}

type audioPromptProvider interface {
	audioPrompt(context.Context, ImageTokenizer, []float32, string, string) (MultimodalPrompt, error)
}

type mediaHistoryPromptProvider interface {
	mediaHistoryPrompt(context.Context, ImageTokenizer, []MediaInput, []string) (MultimodalPrompt, error)
}

type compiledSession struct {
	source Projector
	images imagesPromptProvider
	video  videoPromptProvider
	audio  audioPromptProvider
	media  mediaHistoryPromptProvider
}

// NewSession compiles the projection session for an opened projector.
func NewSession(source Projector) (Session, error) {
	if source == nil {
		return nil, errors.New("projector: session source is nil")
	}
	session := &compiledSession{source: source}
	session.images, _ = source.(imagesPromptProvider)
	session.video, _ = source.(videoPromptProvider)
	session.audio, _ = source.(audioPromptProvider)
	session.media, _ = source.(mediaHistoryPromptProvider)
	return session, nil
}

// OpenSession: open a projector artifact and compile its prompt session.
func OpenSession(ctx context.Context, path string, options OpenOptions) (Session, error) {
	opened, err := OpenAs[Projector](ctx, path, options)
	if err != nil {
		return nil, err
	}
	return NewSession(opened)
}

// OpenActiveSession: OpenActiveAs admission followed by session compilation.
func OpenActiveSession(
	ctx context.Context,
	store artifact.Reader,
	modelID artifact.ID,
	path string,
	options OpenOptions,
) (Session, error) {
	opened, err := OpenActiveAs[Projector](ctx, store, modelID, path, options)
	if err != nil {
		return nil, err
	}
	return NewSession(opened)
}

func (s *compiledSession) Capabilities() SessionCapabilities {
	return SessionCapabilities{
		Image: s.images != nil, MultiImage: s.images != nil,
		Video: s.video != nil, Audio: s.audio != nil, MediaHistory: s.media != nil,
	}
}

func (s *compiledSession) Close() error { return s.source.Close() }

func (s *compiledSession) BuildImagePrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	source image.Image,
	beforeImage, afterImage string,
	thinking bool,
) (MultimodalPrompt, error) {
	if s.images == nil {
		return MultimodalPrompt{}, errors.New("projector: session does not build image prompts")
	}
	return s.images.imagesPrompt(
		ctx, tokenizer, []image.Image{source},
		[]string{beforeImage, afterImage}, PromptOptions{Thinking: thinking},
	)
}

func (s *compiledSession) BuildImagesPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	options PromptOptions,
) (MultimodalPrompt, error) {
	if s.images == nil {
		return MultimodalPrompt{}, errors.New("projector: session does not build image prompts")
	}
	return s.images.imagesPrompt(ctx, tokenizer, sources, text, options)
}

func (s *compiledSession) BuildVideoPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	frames []image.Image,
	beforeVideo, afterVideo string,
	fps float64,
	thinking bool,
) (MultimodalPrompt, error) {
	if s.video == nil {
		return MultimodalPrompt{}, errors.New("projector: session does not build video prompts")
	}
	return s.video.videoPrompt(ctx, tokenizer, frames, beforeVideo, afterVideo, fps, thinking)
}

func (s *compiledSession) BuildAudioPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	samples []float32,
	beforeAudio, afterAudio string,
) (MultimodalPrompt, error) {
	if s.audio == nil {
		return MultimodalPrompt{}, errors.New("projector: session does not build audio prompts")
	}
	return s.audio.audioPrompt(ctx, tokenizer, samples, beforeAudio, afterAudio)
}

func (s *compiledSession) BuildMediaHistoryPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	media []MediaInput,
	text []string,
) (MultimodalPrompt, error) {
	if s.media == nil {
		return MultimodalPrompt{}, errors.New("projector: session does not build media history prompts")
	}
	return s.media.mediaHistoryPrompt(ctx, tokenizer, media, text)
}
