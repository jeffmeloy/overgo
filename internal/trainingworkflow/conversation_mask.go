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
		tail, err := turnTail(messages, index, prompt, render)
		if err != nil {
			return nil, err
		}
		learnedEnd := len(through)
		if strings.HasSuffix(through, tail) {
			learnedEnd -= len(tail)
		}
		keep(full[done:len(prompt)], false)
		keep(full[len(prompt):learnedEnd], true)
		done = learnedEnd
	}
	if done == 0 {
		return nil, errors.New("has no assistant reply to learn")
	}
	keep(full[done:], false)
	return segments, nil
}

// turnWhitespace is what a template may write after the marker that closes a turn.
const turnWhitespace = " \t\r\n"

// turnTail is the whitespace a template writes after the marker that closes an
// assistant turn, a newline after an end-of-turn token for one: the learned span of a
// reply stops before it, since the model would be rewarded for a token only the
// template has. It is found by rendering the turn with no content, so it is what the
// template adds after any reply, and is empty for a template that closes a turn with
// no marker, where whatever follows the reply is the reply's own.
func turnTail(messages []dataset.ChatMessage, index int, prompt string, render ConversationRenderer) (string, error) {
	emptied := slices.Clone(messages[:index+1])
	emptied[index].Content = ""
	rendered, err := render(emptied, false)
	if err != nil {
		return "", err
	}
	closing, found := strings.CutPrefix(rendered, prompt)
	if !found {
		return "", nil
	}
	marker := strings.TrimRight(closing, turnWhitespace)
	if marker == "" {
		return "", nil
	}
	return closing[len(marker):], nil
}

// encodeConversation tokenizes a record's segments as the tokenizer encodes the whole
// text, and says per token whether its segment is learned: a segment ends where the
// encoding of the text up to its end does, which must be the start of the whole
// text's encoding, or a segment boundary falls inside a token and no token can be
// said to be a reply's: the tokenizer and the template then do not agree where a
// reply starts. Every boundary is placed in text that begins at the start, so a
// vocabulary that adds a space at the start of what it encodes, or a token at the
// end, is read as it reads the whole.
func encodeConversation(tokens trainableTokenizer, segments []conversationSegment) ([]int, []bool, error) {
	var whole strings.Builder
	for _, segment := range segments {
		whole.WriteString(segment.Text)
	}
	ids, err := tokens.encode(whole.String())
	if err != nil {
		return nil, nil, err
	}
	trained := make([]bool, len(ids))
	var prefix strings.Builder
	start := 0
	for at, segment := range segments {
		prefix.WriteString(segment.Text)
		end := len(ids)
		if at < len(segments)-1 {
			head, err := tokens.encode(prefix.String())
			if err != nil {
				return nil, nil, err
			}
			// A token the vocabulary adds at the end of what it encodes belongs to the end of the text.
			if tokens.ended && len(head) > 0 && head[len(head)-1] == tokens.end && (len(head) > len(ids) || ids[len(head)-1] != tokens.end) {
				head = head[:len(head)-1]
			}
			if len(head) > len(ids) || !slices.Equal(head, ids[:len(head)]) {
				return nil, nil, errors.New("the tokenizer encodes the conversation up to a turn to other tokens than it begins its whole encoding with, so a reply's tokens cannot be told from its prompt's: train on the prompts too")
			}
			end = len(head)
		}
		for position := start; position < end; position++ {
			trained[position] = segment.Trained
		}
		start = end
	}
	return ids, trained, nil
}
