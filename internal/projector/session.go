package projector

import (
	"context"
	"errors"
	"image"

	"overgo/internal/artifact"
)

// SessionCapabilities: compiled prompt surfaces.
type SessionCapabilities struct {
	Image        bool
	MultiImage   bool
	Video        bool
	Audio        bool
	MediaHistory bool
}

// Session: compiled projector prompt authority.
type Session interface {
	BuildImagePrompt(context.Context, ImageTokenizer, image.Image, string, string, bool) (MultimodalPrompt, error)
	BuildImagesPrompt(context.Context, ImageTokenizer, []image.Image, []string, PromptOptions) (MultimodalPrompt, error)
	BuildVideoPrompt(context.Context, ImageTokenizer, []image.Image, string, string, float64, bool) (MultimodalPrompt, error)
	BuildAudioPrompt(context.Context, ImageTokenizer, []float32, string, string) (MultimodalPrompt, error)
	BuildMediaHistoryPrompt(context.Context, ImageTokenizer, []MediaInput, []string) (MultimodalPrompt, error)
	Capabilities() SessionCapabilities
	Close() error
}

// Prompt callbacks: compiled once per session.
type imagesPromptFunc func(context.Context, ImageTokenizer, []image.Image, []string, PromptOptions) (MultimodalPrompt, error)
type videoPromptFunc func(context.Context, ImageTokenizer, []image.Image, string, string, float64, bool) (MultimodalPrompt, error)
type audioPromptFunc func(context.Context, ImageTokenizer, []float32, string, string) (MultimodalPrompt, error)
type mediaHistoryPromptFunc func(context.Context, ImageTokenizer, []MediaInput, []string) (MultimodalPrompt, error)

type promptDispatch struct {
	images          imagesPromptFunc
	video           videoPromptFunc
	audio           audioPromptFunc
	audioSampleRate func() (int, error)
	media           mediaHistoryPromptFunc
}

func compilePromptDispatch(source Projector) promptDispatch {
	dispatch := promptDispatch{}
	if provider, ok := source.(interface {
		imagesPrompt(context.Context, ImageTokenizer, []image.Image, []string, PromptOptions) (MultimodalPrompt, error)
	}); ok {
		dispatch.images = provider.imagesPrompt
	}
	if provider, ok := source.(interface {
		videoPrompt(context.Context, ImageTokenizer, []image.Image, string, string, float64, bool) (MultimodalPrompt, error)
	}); ok {
		dispatch.video = provider.videoPrompt
	}
	if provider, ok := source.(interface {
		audioPrompt(context.Context, ImageTokenizer, []float32, string, string) (MultimodalPrompt, error)
		audioSampleRate() (int, error)
	}); ok {
		dispatch.audio = provider.audioPrompt
		dispatch.audioSampleRate = provider.audioSampleRate
	}
	if provider, ok := source.(interface {
		mediaHistoryPrompt(context.Context, ImageTokenizer, []MediaInput, []string) (MultimodalPrompt, error)
	}); ok {
		dispatch.media = provider.mediaHistoryPrompt
	}
	return dispatch
}

type compiledSession struct {
	source Projector
	prompt promptDispatch
}

// NewSession compiles prompt dispatch.
func NewSession(source Projector) (Session, error) {
	if source == nil {
		return nil, errors.New("projector: session source is nil")
	}
	return &compiledSession{source: source, prompt: compilePromptDispatch(source)}, nil
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
		Image: s.prompt.images != nil, MultiImage: s.prompt.images != nil,
		Video: s.prompt.video != nil, Audio: s.prompt.audio != nil,
		MediaHistory: s.prompt.media != nil,
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
	if s.prompt.images == nil {
		return MultimodalPrompt{}, errors.New("projector: session does not build image prompts")
	}
	return s.prompt.images(
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
	if s.prompt.images == nil {
		return MultimodalPrompt{}, errors.New("projector: session does not build image prompts")
	}
	return s.prompt.images(ctx, tokenizer, sources, text, options)
}

func (s *compiledSession) BuildVideoPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	frames []image.Image,
	beforeVideo, afterVideo string,
	fps float64,
	thinking bool,
) (MultimodalPrompt, error) {
	if s.prompt.video == nil {
		return MultimodalPrompt{}, errors.New("projector: session does not build video prompts")
	}
	return s.prompt.video(ctx, tokenizer, frames, beforeVideo, afterVideo, fps, thinking)
}

func (s *compiledSession) BuildAudioPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	samples []float32,
	beforeAudio, afterAudio string,
) (MultimodalPrompt, error) {
	if s.prompt.audio == nil {
		return MultimodalPrompt{}, errors.New("projector: session does not build audio prompts")
	}
	return s.prompt.audio(ctx, tokenizer, samples, beforeAudio, afterAudio)
}

func (s *compiledSession) audioInputSampleRate() (int, error) {
	if s.prompt.audioSampleRate == nil {
		return 0, errors.New("projector: session does not build audio prompts")
	}
	return s.prompt.audioSampleRate()
}

// AudioSampleRate returns the artifact-declared rate for a compiled session.
func AudioSampleRate(session Session) (int, error) {
	provider, ok := session.(interface{ audioInputSampleRate() (int, error) })
	if !ok {
		return 0, errors.New("projector: session has no audio sample-rate contract")
	}
	return provider.audioInputSampleRate()
}

func (s *compiledSession) BuildMediaHistoryPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	media []MediaInput,
	text []string,
) (MultimodalPrompt, error) {
	if s.prompt.media == nil {
		return MultimodalPrompt{}, errors.New("projector: session does not build media history prompts")
	}
	return s.prompt.media(ctx, tokenizer, media, text)
}
