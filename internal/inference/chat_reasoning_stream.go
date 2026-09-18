package inference

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// reasoningChannelStream retains undecided marker/UTF-8 fragments and trim
// whitespace. The caller owns completed output and tool parsing.
type reasoningChannelStream struct {
	markers   reasoningMarkers
	pending   string
	atStart   bool
	thinking  bool
	trimStart bool
}

func newReasoningChannelStream(template, prompt string) (*reasoningChannelStream, error) {
	markers, err := chatReasoningMarkers(template)
	if err != nil {
		return nil, err
	}
	stream := &reasoningChannelStream{markers: markers}
	declared := strings.Contains(template, markers.open) && strings.Contains(template, markers.close)
	if !declared {
		return stream, nil
	}
	tail := strings.TrimRight(prompt, " \t\r\n")
	switch {
	case strings.HasSuffix(tail, markers.open):
		stream.thinking, stream.trimStart = true, true
	case strings.HasSuffix(tail, markers.close):
		// The close belongs to the prompt. Generated visible bytes are intact.
	default:
		stream.atStart = true
	}
	return stream, nil
}

func (s *reasoningChannelStream) accept(piece string, final bool) (reasoning, content string) {
	s.pending += piece
	if s.atStart {
		if !final {
			known := reasoningUTF8Prefix(s.pending)
			if known < len(s.pending) && strings.TrimLeftFunc(s.pending[:known], unicode.IsSpace) == "" {
				return "", ""
			}
		}
		trimmed := strings.TrimLeftFunc(s.pending, unicode.IsSpace)
		if !final && strings.HasPrefix(s.markers.open, trimmed) && trimmed != s.markers.open {
			return "", ""
		}
		s.atStart = false
		if strings.HasPrefix(trimmed, s.markers.open) {
			s.pending = trimmed[len(s.markers.open):]
			s.thinking, s.trimStart = true, true
		}
	}
	if s.thinking {
		if s.trimStart {
			s.pending = strings.TrimLeft(s.pending, "\r\n")
			s.trimStart = s.pending == ""
		}
		if index := strings.Index(s.pending, s.markers.close); index >= 0 {
			reasoning = strings.TrimRight(s.pending[:index], "\r\n")
			s.pending = s.pending[index+len(s.markers.close):]
			s.thinking, s.trimStart = false, true
		} else {
			end := len(s.pending)
			if !final {
				for size := min(end, len(s.markers.close)-1); size > 0; size-- {
					if strings.HasSuffix(s.pending, s.markers.close[:size]) {
						end -= size
						break
					}
				}
				end = reasoningUTF8Prefix(s.pending[:end])
			}
			end = len(strings.TrimRight(s.pending[:end], "\r\n"))
			reasoning, s.pending = s.pending[:end], s.pending[end:]
			if final {
				s.pending = ""
			}
			return reasoning, ""
		}
	}
	if s.trimStart {
		s.pending = strings.TrimLeft(s.pending, "\r\n")
		s.trimStart = s.pending == ""
	}
	end := len(s.pending)
	if !final {
		end = reasoningUTF8Prefix(s.pending)
	}
	content, s.pending = s.pending[:end], s.pending[end:]
	return reasoning, content
}

func reasoningUTF8Prefix(text string) int {
	end := 0
	for end < len(text) && utf8.FullRuneInString(text[end:]) {
		_, size := utf8.DecodeRuneInString(text[end:])
		end += size
	}
	return end
}
