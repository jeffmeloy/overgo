package trainingworkflow

import (
	"errors"
	"slices"
	"strings"

	"overgo/internal/dataset"
)

// assistantRole is the speaker whose messages a conversation run learns.
const assistantRole = "assistant"

// conversationSegment is a stretch of a conversation as the chat template
// renders it, and whether a run learns it.
type conversationSegment struct {
	Text    string `json:"text"`
	Trained bool   `json:"trained,omitzero"`
}

// unusableConversation is why one well-formed conversation cannot be cut into the
// stretches a run learns and reads under this model's chat template and tokenizer.
// A run skips such a conversation and says so; a file that is no conversation at
// all is the dataset's fault and stays an error.
type unusableConversation struct{ error }

// Unwrap is the reason the conversation cannot be learned from.
func (unusable unusableConversation) Unwrap() error { return unusable.error }

// SkippedConversation is a conversation of the dataset a run left out, and why.
type SkippedConversation struct {
	File   string
	Reason string
}

// conversationRecord is one conversation as a run reads it: its rendering, in
// order, cut into the stretches a run learns and the stretches it only reads.
type conversationRecord struct {
	Segments []conversationSegment `json:"segments"`
}

// segmentConversation renders a conversation and cuts it where the replies
// begin and end. With prompts set the whole rendering is one learned segment.
// Otherwise each assistant message's text, after the generation prompt that
// introduces it, is learned and the rest, the user's turns, the template's
// markers and the generation prompt itself, which serving supplies, is only
// read. The cut is the rendering of the messages up to the reply, with the
// generation prompt, and up to and including it, without: a template whose
// renderings do not extend one another that way cannot be cut, and says so.
func segmentConversation(messages []dataset.ChatMessage, render ConversationRenderer, prompts bool) ([]conversationSegment, error) {
	full, err := render(messages, false)
	if err != nil {
		return nil, err
	}
	if full == "" {
		return nil, errors.New("renders as no text")
	}
	if prompts {
		return []conversationSegment{{Text: full, Trained: true}}, nil
	}
	var segments []conversationSegment
	keep := func(text string, trained bool) {
		if text != "" {
			segments = append(segments, conversationSegment{Text: text, Trained: trained})
		}
	}
	done := 0
	for index, message := range messages {
		if message.Role != assistantRole {
			continue
		}
		prompt, err := render(messages[:index], true)
		if err != nil {
			return nil, err
		}
		through, err := render(messages[:index+1], false)
		if err != nil {
			return nil, err
		}
		if !strings.HasPrefix(through, prompt) || !strings.HasPrefix(full, through) || len(prompt) < done {
			return nil, errors.New("the chat template does not render the conversation turn by turn, so a reply cannot be told from its prompt: train on the prompts too")
		}
		keep(full[done:len(prompt)], false)
		keep(full[len(prompt):len(through)], true)
		done = len(through)
	}
	if done == 0 {
		return nil, errors.New("has no assistant reply to learn")
	}
	keep(full[done:], false)
	return segments, nil
}

// encodeConversation tokenizes a record's segments in order, the first as the
// tokenizer encodes a conversation and each later one as a continuation of the
// text before it, and says per token whether its segment is learned. The tokens
// must be those the whole text encodes to, or a segment boundary falls inside a
// token and no token can be said to be a reply's: the tokenizer and the
// template then do not agree where a reply starts.
func encodeConversation(tokens trainableTokenizer, segments []conversationSegment) ([]int, []bool, error) {
	var ids []int
	var trained []bool
	var whole strings.Builder
	for index, segment := range segments {
		encode := tokens.extend
		if index == 0 {
			encode = tokens.encode
		}
		piece, err := encode(segment.Text)
		if err != nil {
			return nil, nil, err
		}
		ids = append(ids, piece...)
		trained = append(trained, slices.Repeat([]bool{segment.Trained}, len(piece))...)
		whole.WriteString(segment.Text)
	}
	if len(segments) > 1 {
		again, err := tokens.encode(whole.String())
		if err != nil {
			return nil, nil, err
		}
		if !slices.Equal(again, ids) {
			return nil, nil, errors.New("the tokenizer encodes the conversation turn by turn to other tokens than it encodes it whole, so a reply's tokens cannot be told from its prompt's: train on the prompts too")
		}
	}
	return ids, trained, nil
}
