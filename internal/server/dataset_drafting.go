package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/fsatomic"
	"overgo/internal/inference"
)

// draftPairForm closes every drafting instruction, so one reader takes the pair whatever the template.
const draftPairForm = ` Reply with one JSON object and nothing else: {"question": "...", "answer": "..."}`

// datasetDraftTemplates holds, by the id a request names, the fixed instruction the served model drafts one pair
// under: a page never types an instruction.
var datasetDraftTemplates = map[string]string{
	"grounded-qa": "Read the passage the user gives. Write one question that the passage answers, and the answer to it using only what the passage says." + draftPairForm,
	"text-to-sql": "The user gives rows of a table, under a header row that names its columns when the passage has one. Write one question a person would ask of that table, and as the answer one SQL query over a table named data that answers it." + draftPairForm,
}

const (
	// draftPassageRunes bounds one passage: a few paragraphs, about five hundred tokens of English, enough to ask
	// one question of and small beside a chat model's context, where the instruction and the answer must fit too.
	draftPassageRunes = 2000
	// draftMaxPassages bounds one request, which holds the served model for one generation a passage: sixty-four
	// passages is a long report, and a longer text is drafted in parts under more than one name.
	draftMaxPassages = 64
)

// datasetDraftRequest names the dataset to make, the template, and exactly one source: pasted text, or the id of a
// document the artifact intake stored.
type datasetDraftRequest struct {
	Name     string `json:"name"`
	Template string `json:"template"`
	Text     string `json:"text,omitzero"`
	Document string `json:"document,omitzero"`
}

// draftDataset answers POST /datasets/draft: the served model drafts one question and answer from each passage of a
// text under a fixed template, and the pairs register as a conversation dataset. A request that cannot be drafted
// is refused before any generation, and the answer is the only record of how the dataset was made.
func (h *Handler) draftDataset(response http.ResponseWriter, request *http.Request) {
	formatter, ok := h.generator.(chatFormatter)
	if !ok {
		reason, action := workspaceRefusal("conversation")
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, reason+" "+action)
		return
	}
	var body datasetDraftRequest
	if !h.decodeBoundedJSON(response, request, &body) || !requireLibraryStore(response, request, h.repository) {
		return
	}
	if h.config.LibraryIntake.DatasetRoot == "" {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "dataset drafting needs the data root's datasets directory")
		return
	}
	body.Name = strings.TrimSpace(body.Name)
	instruction, passages, err := h.draftPassages(request.Context(), body)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	prompts := make([]nativePrompt, len(passages))
	for index, passage := range passages {
		prompts[index], err = h.normalizeChatPrompt(request.Context(), formatter, chatCompletionRequest{Messages: []inference.ChatMessage{
			{Role: inference.ChatRoleSystem, Content: instruction}, {Role: inference.ChatRoleUser, Content: string(passage)},
		}}, nil, false)
		if err != nil {
			writeInvalidRequest(response, err)
			return
		}
	}
	// Each answer may run to the server's ceiling on a request's output, as an agent's turn may: the default
	// completion budget is sized for a short completion and cuts a pair off before its closing brace, and a model
	// ends its turn where the pair ends. Every passage is admitted on the one session before any generates, so a
	// passage too long for the model refuses the whole draft.
	plan, ok := h.prepareProtocolGenerationPlan(response, request, prompts[0], samplingParameters{}, h.config.MaxTokens, nil)
	if !ok {
		return
	}
	defer plan.release()
	turns := []*protocolGenerationPlan{plan}
	for _, prompt := range prompts[1:] {
		turn, err := plan.turn(prompt)
		if err != nil {
			writePromptRefusal(response, err)
			return
		}
		turns = append(turns, turn)
	}
	var conversations []dataset.Conversation
	for index, turn := range turns {
		result, err := turn.run(nil)
		if err != nil {
			writeGenerationError(response, err)
			return
		}
		// An answer that holds no pair is skipped and counted, never guessed at.
		if conversation, drafted := draftedConversation(string(passages[index]), result.pump.text()); drafted {
			conversations = append(conversations, conversation)
		}
	}
	if len(conversations) == 0 {
		writeError(response, http.StatusInternalServerError, "generation_error", "the served model drafted no question and answer pair")
		return
	}
	registered, err := h.registerDrafted(request.Context(), body.Name, conversations)
	if err != nil {
		writeError(response, http.StatusUnprocessableEntity, "library_refused", err.Error())
		return
	}
	writeJSON(response, http.StatusOK, map[string]any{
		"name": body.Name, "template": body.Template, "model": h.config.ModelID, "dataset": registered.Dataset, "commit": registered.Commit.String(),
		"files": registered.Files, "bytes": registered.Bytes, "skipped": len(turns) - len(conversations),
	})
}

// draftPassages refuses a request that cannot be drafted, before the served model is held, and answers the
// template's instruction with the source text cut into passages.
func (h *Handler) draftPassages(ctx context.Context, body datasetDraftRequest) (string, [][]rune, error) {
	instruction, known := datasetDraftTemplates[body.Template]
	if !known {
		return "", nil, fmt.Errorf("template must be one of %s", strings.Join(slices.Sorted(maps.Keys(datasetDraftTemplates)), ", "))
	}
	if err := fsatomic.PathComponent(body.Name); err != nil {
		return "", nil, fmt.Errorf("dataset name %w", err)
	}
	// The import refuses a taken name too, but only after every passage has generated.
	if _, err := os.Lstat(filepath.Join(h.config.LibraryIntake.DatasetRoot, body.Name)); err == nil {
		return "", nil, fmt.Errorf("dataset %s already exists; draft under another name", body.Name)
	}
	if (body.Text == "") == (body.Document == "") {
		return "", nil, errors.New("give exactly one of text or document")
	}
	text := body.Text
	if body.Document != "" {
		id, err := artifact.ParseID(body.Document)
		if err != nil {
			return "", nil, err
		}
		content, found, err := artifact.ReadContent(ctx, h.repository, id)
		if err != nil || !found {
			return "", nil, errors.Join(fmt.Errorf("document %s is not stored", body.Document), err)
		}
		if text, err = documentText(content.Descriptor.MediaType, content.Data); err != nil {
			return "", nil, fmt.Errorf("document %w", err)
		}
	}
	passages := slices.Collect(slices.Chunk([]rune(strings.TrimSpace(text)), draftPassageRunes))
	if len(passages) == 0 || len(passages) > draftMaxPassages {
		return "", nil, fmt.Errorf("the text must make between 1 and %d passages of %d characters, and it makes %d", draftMaxPassages, draftPassageRunes, len(passages))
	}
	return instruction, passages, nil
}

// draftedConversation reads the pair from a model's reply as the JSON object between its first and last brace, so
// prose around the object is ignored, and answers the conversation that asks the question of the passage.
func draftedConversation(passage, reply string) (dataset.Conversation, bool) {
	var pair struct {
		Question string `json:"question"`
		Answer   string `json:"answer"`
	}
	start, end := strings.IndexByte(reply, '{'), strings.LastIndexByte(reply, '}')
	if start < 0 || end < start || json.Unmarshal([]byte(reply[start:end+1]), &pair) != nil || pair.Question == "" || pair.Answer == "" {
		return dataset.Conversation{}, false
	}
	return dataset.Conversation{Messages: []dataset.ChatMessage{
		{Role: "user", Content: passage + "\n\n" + pair.Question}, {Role: "assistant", Content: pair.Answer},
	}}, true
}

// registerDrafted registers the conversations as the dataset name through the common-format import, which checks
// the name, publishes the directory whole and refuses a taken one. The import reads a file, so the conversations
// pass through a working file that is removed.
func (h *Handler) registerDrafted(ctx context.Context, name string, conversations []dataset.Conversation) (dataset.RegisteredDataset, error) {
	file, err := os.CreateTemp("", "overgo-dataset-draft-*.json")
	if err != nil {
		return dataset.RegisteredDataset{}, err
	}
	defer os.Remove(file.Name())
	if err := errors.Join(json.NewEncoder(file).Encode(conversations), file.Close()); err != nil {
		return dataset.RegisteredDataset{}, err
	}
	registered, _, err := dataset.ImportCommonFormat(ctx, h.repository, name, file.Name(), h.config.LibraryIntake.DatasetRoot)
	return registered, err
}
