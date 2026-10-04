package dataset

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/fsatomic"
)

// The common chat formats a dataset arrives in -- alpaca {instruction, input,
// output}, ShareGPT {conversations: [{from, value}]} and OpenAI messages
// {messages: [{role, content}]} -- import through one converter into the
// OpenAI messages form, one conversation per record, which the store
// registers as a structured dataset.

// ModalityStructured classes the JSON, CSV and Parquet records a dataset
// holds.
const ModalityStructured = "structured"

// ChatMessage is one turn of an imported conversation.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Conversation is one imported record in the OpenAI messages form.
type Conversation struct {
	Messages []ChatMessage `json:"messages"`
}

// shareGPTRoles maps a ShareGPT speaker to its role.
var shareGPTRoles = map[string]string{"system": "system", "human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}

// messageRoles are the roles a conversation keeps.
var messageRoles = map[string]bool{"system": true, "user": true, "assistant": true}

// commonRecord holds the fields any of the three formats sets.
type commonRecord struct {
	Messages      []commonMessage `json:"messages"`
	Conversations []struct {
		From  string `json:"from"`
		Value string `json:"value"`
	} `json:"conversations"`
	System      string  `json:"system"`
	Instruction *string `json:"instruction"`
	Input       string  `json:"input"`
	Output      *string `json:"output"`
}

// commonMessage is an OpenAI message whose content is text or text parts.
type commonMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// convertCommonFormat reads a JSON Lines file or a JSON array of records in
// any of the three formats and returns each as a conversation, with the
// formats it met in the order it met them.
func convertCommonFormat(source io.Reader) ([]Conversation, []string, error) {
	reader := bufio.NewReader(source)
	decoder := json.NewDecoder(reader)
	if first, err := firstByte(reader); err != nil {
		return nil, nil, err
	} else if first == '[' {
		if _, err := decoder.Token(); err != nil {
			return nil, nil, err
		}
	}
	var conversations []Conversation
	var formats []string
	for decoder.More() {
		var record commonRecord
		if err := decoder.Decode(&record); err != nil {
			return nil, nil, fmt.Errorf("dataset: record %d: %w", len(conversations)+1, err)
		}
		conversation, format, err := record.conversation()
		if err != nil {
			return nil, nil, fmt.Errorf("dataset: record %d: %w", len(conversations)+1, err)
		}
		conversations = append(conversations, conversation)
		if !slices.Contains(formats, format) {
			formats = append(formats, format)
		}
	}
	if len(conversations) == 0 {
		return nil, nil, errors.New("dataset: the file holds no records")
	}
	return conversations, formats, nil
}

// byteOrderMark opens a UTF-8 file some editors save.
const byteOrderMark = "\ufeff"

// firstByte peeks the first byte past a byte order mark and white space.
func firstByte(reader *bufio.Reader) (byte, error) {
	if peeked, _ := reader.Peek(len(byteOrderMark)); string(peeked) == byteOrderMark {
		_, _ = reader.Discard(len(byteOrderMark))
	}
	for {
		next, err := reader.ReadByte()
		if err != nil {
			return 0, fmt.Errorf("dataset: the file holds no records: %w", err)
		}
		if !strings.ContainsRune(" \t\r\n", rune(next)) {
			return next, reader.UnreadByte()
		}
	}
}

// conversation normalizes one record, naming the format it was in.
func (record commonRecord) conversation() (Conversation, string, error) {
	var conversation Conversation
	add := func(role, content string) {
		conversation.Messages = append(conversation.Messages, ChatMessage{Role: role, Content: content})
	}
	var format string
	switch {
	case len(record.Messages) > 0:
		format = "openai-messages"
		for _, message := range record.Messages {
			content, err := messageText(message.Content)
			if err != nil {
				return Conversation{}, "", err
			}
			if !messageRoles[message.Role] {
				return Conversation{}, "", fmt.Errorf("message role %q is not system, user or assistant", message.Role)
			}
			add(message.Role, content)
		}
	case len(record.Conversations) > 0:
		format = "sharegpt"
		for _, turn := range record.Conversations {
			role, known := shareGPTRoles[turn.From]
			if !known {
				return Conversation{}, "", fmt.Errorf("conversation speaker %q is not system, human or gpt", turn.From)
			}
			add(role, turn.Value)
		}
	case record.Instruction != nil && record.Output != nil:
		format = "alpaca"
		if record.System != "" {
			add("system", record.System)
		}
		prompt := *record.Instruction
		if record.Input != "" {
			prompt += "\n\n" + record.Input
		}
		add("user", prompt)
		add("assistant", *record.Output)
	default:
		return Conversation{}, "", errors.New("the record is none of alpaca {instruction, output}, ShareGPT {conversations} or OpenAI {messages}")
	}
	return conversation, format, nil
}

// messageText reads an OpenAI content: a string, or text parts joined.
func messageText(content json.RawMessage) (string, error) {
	var text string
	if err := json.Unmarshal(content, &text); err == nil {
		return text, nil
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(content, &parts); err != nil {
		return "", errors.New("message content is neither text nor text parts")
	}
	texts := make([]string, 0, len(parts))
	for _, part := range parts {
		if part.Type != "text" {
			return "", fmt.Errorf("message content part %q is not text", part.Type)
		}
		texts = append(texts, part.Text)
	}
	return strings.Join(texts, ""), nil
}

// ImportCommonFormat converts a file in any of the common chat formats into
// a new directory under root, one conversation per JSON file named by its
// position, and registers the directory under name as a structured dataset.
// An existing directory is refused, never overwritten.
func ImportCommonFormat(ctx context.Context, repository artifact.Repository, name, source, root string) (RegisteredDataset, []string, error) {
	name = strings.TrimSpace(name)
	if name == "" || name != filepath.Base(name) || name == "." || name == ".." {
		return RegisteredDataset{}, nil, fmt.Errorf("dataset: import name %q must be one path element", name)
	}
	file, err := os.Open(source)
	if err != nil {
		return RegisteredDataset{}, nil, err
	}
	conversations, formats, err := convertCommonFormat(file)
	_ = file.Close()
	if err != nil {
		return RegisteredDataset{}, nil, err
	}
	directory, err := fsatomic.PublishDirectory(root, name, clioptions.OutputDirectoryMode, func(stage string) error {
		return writeConversations(stage, conversations)
	})
	if errors.Is(err, fs.ErrExist) {
		return RegisteredDataset{}, nil, fmt.Errorf("dataset: %s already exists; import under another name", filepath.Join(root, name))
	}
	if err != nil {
		return RegisteredDataset{}, nil, err
	}
	registered, err := RegisterDirectoryDatasetAs(ctx, repository, name, directory, ModalityStructured)
	if err != nil {
		return RegisteredDataset{}, nil, errors.Join(err, os.RemoveAll(directory))
	}
	return registered, formats, nil
}

// writeConversations writes each conversation as its own JSON file in
// directory, the names padded to sort in record order.
func writeConversations(directory string, conversations []Conversation) error {
	width := len(strconv.Itoa(len(conversations)))
	for index, conversation := range conversations {
		data, err := json.Marshal(conversation)
		if err != nil {
			return err
		}
		file, err := os.Create(filepath.Join(directory, fmt.Sprintf("%0*d.json", width, index+1)))
		if err != nil {
			return err
		}
		_, err = file.Write(data)
		if err := errors.Join(err, file.Close()); err != nil {
			return err
		}
	}
	return nil
}
