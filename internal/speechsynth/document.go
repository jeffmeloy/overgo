package speechsynth

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"unicode"
	"unicode/utf8"

	"overgo/internal/artifact"
)

// DocumentRequest preserves edited text independently of the optional original
// file artifact. Unlike SynthesisRequest, it never asks for a truncated preview.
type DocumentRequest struct {
	Text        string      `json:"text"`
	Voice       string      `json:"voice"`
	Seed        int64       `json:"seed"`
	Temperature float64     `json:"temperature,omitzero"`
	Source      artifact.ID `json:"source,omitzero"`
}

// DocumentSegment identifies an exact, contiguous UTF-8 source span.
type DocumentSegment struct {
	Start      int `json:"start"`
	End        int `json:"end"`
	Tokens     int `json:"tokens"`
	MaxFrames  int `json:"max_frames"`
	MaxSamples int `json:"max_samples"`
}

// DocumentPlan binds source coverage and the reference generation policy before
// execution. Every segment must subsequently reach the native EOS stop.
type DocumentPlan struct {
	Request    DocumentRequest   `json:"request"`
	Segments   []DocumentSegment `json:"segments"`
	SampleRate int               `json:"sample_rate"`
}

// RIFF/WAVE PCM16 mono: RIFF + fmt + data headers, followed by two bytes
// per sample. Keep the artifact ceiling separate from working-memory admission.
const documentWAVHeaderBytes, documentSampleBytes = 44, 2

// Pocket-TTS 2.1.0 declares a 50-token chunk policy and estimates duration as
// tokens/3 + 2 seconds in tts_model.py. This is a source policy, not a model
// architecture limit. Oversized words are split at UTF-8 boundaries instead of
// accepting the reference implementation's warning-only oversized chunks.
const documentSegmentTokens = 50

// PlanDocument uses this model's actual tokenizer and codec frame rate. The
// caller supplies immutable original-file identity; the edited bytes stay exact.
func (s *Synthesizer) PlanDocument(ctx context.Context, request DocumentRequest) (DocumentPlan, error) {
	if ctx == nil || !s.complete() {
		return DocumentPlan{}, errors.New("speech document: incomplete context or synthesizer")
	}
	if err := context.Cause(ctx); err != nil {
		return DocumentPlan{}, err
	}
	if !utf8.ValidString(request.Text) || strings.TrimSpace(request.Text) == "" || strings.ContainsRune(request.Text, 0) {
		return DocumentPlan{}, errors.New("speech document: text must be nonempty UTF-8 without NUL")
	}
	if len(request.Text) > artifact.MaxContentBytes {
		return DocumentPlan{}, errors.New("speech document: source exceeds the artifact limit")
	}
	if request.Source != (artifact.ID{}) && request.Source.Kind() != artifact.KindFile {
		return DocumentPlan{}, errors.New("speech document: original source must be a file artifact")
	}
	if strings.TrimSpace(request.Voice) == "" || request.Temperature < 0 || math.IsNaN(request.Temperature) || math.IsInf(request.Temperature, 0) {
		return DocumentPlan{}, errors.New("speech document: a voice and finite nonnegative temperature are required")
	}
	if _, err := ResolveVoicePath(s.directory, request.Voice); err != nil {
		return DocumentPlan{}, err
	}
	if s.model.Codec.FrameRate <= 0 || math.IsNaN(s.model.Codec.FrameRate) || math.IsInf(s.model.Codec.FrameRate, 0) {
		return DocumentPlan{}, errors.New("speech document: invalid codec frame rate")
	}
	segments, err := splitDocument(ctx, s.tokenizer, request.Text, documentSegmentTokens)
	if err != nil {
		return DocumentPlan{}, err
	}
	remaining := (artifact.MaxContentBytes - documentWAVHeaderBytes) / documentSampleBytes
	for i := range segments {
		segments[i].MaxFrames = int(math.Ceil((float64(segments[i].Tokens)/3 + 2) * s.model.Codec.FrameRate))
		samples := segments[i].MaxFrames
		strides := []int{s.model.Codec.upsampleK / 2}
		for _, conv := range s.model.Codec.convs {
			if conv.transpose {
				strides = append(strides, conv.strd)
			} else if conv.strd != 1 {
				return DocumentPlan{}, errors.New("speech document: unsupported strided codec output bound")
			}
		}
		for _, stride := range strides {
			if samples <= 0 || stride <= 0 || samples > remaining/stride {
				return DocumentPlan{}, errors.New("speech document: text exceeds the recording artifact limit")
			}
			samples *= stride
		}
		segments[i].MaxSamples = samples
		remaining -= samples
	}
	return DocumentPlan{Request: request, Segments: segments, SampleRate: s.model.Codec.SampleRate}, nil
}

func splitDocument(ctx context.Context, tokenizer *Unigram, text string, limit int) ([]DocumentSegment, error) {
	var result []DocumentSegment
	for start := 0; start < len(text); {
		end, word, sentence := start, start, start
		for cursor := start; cursor < len(text); {
			if err := context.Cause(ctx); err != nil {
				return nil, err
			}
			r, size := utf8.DecodeRuneInString(text[cursor:])
			cursor += size
			if unicode.IsSpace(r) {
				for cursor < len(text) {
					next, size := utf8.DecodeRuneInString(text[cursor:])
					if !unicode.IsSpace(next) {
						break
					}
					cursor += size
				}
			}
			ids, err := tokenizer.Encode(text[start:cursor])
			if err != nil {
				return nil, err
			}
			if len(ids) > limit {
				break
			}
			end = cursor
			if unicode.IsSpace(r) && strings.TrimSpace(text[start:cursor]) != "" {
				word = cursor
			}
			next, _ := utf8.DecodeRuneInString(text[cursor:])
			if strings.ContainsRune(".!?", r) && (cursor == len(text) || unicode.IsSpace(next)) {
				sentence = cursor
			}
		}
		if end == start {
			return nil, fmt.Errorf("speech document: a Unicode character exceeds the token policy at byte %d", start)
		}
		cut := end
		if end < len(text) {
			if sentence > start {
				cut = sentence
			} else if word > start {
				cut = word
			}
		}
		// Trailing whitespace belongs to the preceding segment. Preserve those
		// bytes without creating a speech request containing only separators.
		if strings.TrimSpace(text[cut:]) == "" {
			cut = len(text)
		}
		ids, err := tokenizer.Encode(text[start:cut])
		if err != nil {
			return nil, err
		}
		if len(ids) > limit || strings.TrimSpace(text[start:cut]) == "" {
			return nil, errors.New("speech document: invalid segment bound")
		}
		result = append(result, DocumentSegment{Start: start, End: cut, Tokens: len(ids)})
		start = cut
	}
	return result, nil
}
