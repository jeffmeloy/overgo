package evaluation

import (
	"fmt"
	"strings"
)

// MediaKind is the kind of media a case's prompt rides.
type MediaKind string

// MediaVideo is a case that asks about one video clip.
const MediaVideo MediaKind = "video"

// mediaPromptOpening and mediaPromptClosing frame the one line that names a
// case's media: a suite stays a plain text document, and a runtime that holds
// the media resolves the name against its own root.
const (
	mediaPromptOpening = "<overgo-media "
	mediaPromptClosing = ">"
	// mediaPromptFields is the kind and the name the line holds.
	mediaPromptFields = 2
)

// mediaPrompt is the prompt of a case that asks its question of the named
// media: the line that names it, then the question.
func mediaPrompt(kind MediaKind, name, question string) string {
	return fmt.Sprintf("%skind=%s name=%s%s\n%s", mediaPromptOpening, kind, name, mediaPromptClosing, question)
}

// ParseMediaPrompt reads what mediaPrompt wrote, and reports false for a
// prompt that names no media. A name holds no whitespace, so the line
// holds exactly the kind and the name.
func ParseMediaPrompt(prompt string) (kind MediaKind, name, question string, ok bool) {
	line, question, found := strings.Cut(prompt, "\n")
	line, opened := strings.CutPrefix(line, mediaPromptOpening)
	line, closed := strings.CutSuffix(line, mediaPromptClosing)
	if !found || !opened || !closed {
		return "", "", "", false
	}
	fields := strings.Fields(line)
	kindField, hasKind := strings.CutPrefix(fieldAt(fields, 0), "kind=")
	nameField, hasName := strings.CutPrefix(fieldAt(fields, 1), "name=")
	if len(fields) != mediaPromptFields || !hasKind || !hasName || nameField == "" || MediaKind(kindField) != MediaVideo {
		return "", "", "", false
	}
	return MediaKind(kindField), nameField, question, true
}

func fieldAt(fields []string, index int) string {
	if index < len(fields) {
		return fields[index]
	}
	return ""
}
