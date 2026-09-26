package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/runrecord"
	"overgo/internal/tokenizer"
)

// continuationParameters: the sampling and stops a turn generated with, kept
// in its token context so a continuation decodes the same way.
type continuationParameters struct {
	Sampling samplingParameters `json:"sampling"`
	Stop     []string           `json:"stop,omitempty"`
}

// turnSession: a response turn's continuation facts: the parameters it
// decodes with, the text a continued turn showed before it stopped, the
// stopped turn it continues, and the text a stop filter held back there.
// keep: a plain-text turn keeps its token context when it can be continued.
type turnSession struct {
	parameters continuationParameters
	prefix     string
	continues  artifact.ID
	held       string
	keep       bool
}

// turnRecord: what an interaction stores beside its messages: the token
// context of a turn that can be continued, and the stopped turn it continues.
type turnRecord struct {
	context   *runrecord.InteractionContext
	continues artifact.ID
}

// record: a turn that stopped or reached its output limit keeps its token
// context (every id, the last one pending) with the text its stop filter
// held back; a finished turn keeps none, as nothing continues it.
func (session turnSession) record(ids []tokenizer.TokenID, held string, continuable bool) (turnRecord, error) {
	record := turnRecord{continues: session.continues}
	if !session.keep || !continuable || len(ids) < runrecord.InteractionContextMinimumTokens {
		return record, nil
	}
	sampling, err := json.Marshal(session.parameters)
	if err != nil {
		return turnRecord{}, err
	}
	tokens := make([]uint32, len(ids))
	for index, id := range ids {
		tokens[index] = uint32(id)
	}
	record.context = &runrecord.InteractionContext{Tokens: tokens, Sampling: sampling, Held: held}
	return record, nil
}

// publishTurn stores a response turn with what it can be continued from.
func (h *Handler) publishTurn(
	ctx context.Context,
	responseID string,
	parent artifact.ID,
	messages []inference.ChatMessage,
	terminal runrecord.Outcome,
	reason runrecord.InteractionTerminalReason,
	session turnSession,
	ids []tokenizer.TokenID,
	held string,
	continuable bool,
) error {
	record, err := session.record(ids, held, continuable)
	if err != nil {
		return err
	}
	return h.publishResponseInteraction(ctx, responseID, parent, messages, terminal, reason, record)
}

// continuedTurn: a continue request's resolved turn: the stopped turn's
// messages and parent, its token context as the prompt, and its session.
type continuedTurn struct {
	turn    []inference.ChatMessage
	parent  artifact.ID
	prompt  nativePrompt
	session turnSession
}

// sessionRefusal answers a continuation that cannot resume, with its reason.
func sessionRefusal(response http.ResponseWriter, reason string) {
	failure := errorEnvelope("invalid_request_error", reason)
	failure.Error.Code = "session_mismatch"
	writeJSON(response, http.StatusConflict, failure)
}

// continuedResponse resolves a continue request: the previous response must
// be a stopped turn this model generated with a kept token context, and a
// request that names sampling must name the sampling the turn ran with.
func (h *Handler) continuedResponse(response http.ResponseWriter, request *http.Request, body responsesRequest) (continuedTurn, bool) {
	input := bytes.TrimSpace(body.Input)
	if body.PreviousResponseID == "" || (len(input) != 0 && !bytes.Equal(input, []byte("null"))) ||
		body.Instructions != "" || len(bytes.TrimSpace(body.Tools)) != 0 || body.Reasoning != nil {
		writeInvalidRequestMessage(response, "continue resumes previous_response_id and takes no input, instructions, tools or reasoning")
		return continuedTurn{}, false
	}
	if h.repository == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "continuing a response needs the response store")
		return continuedTurn{}, false
	}
	id := body.PreviousResponseID
	if running, found := h.inflight.lookup(id); found {
		if _, done, _, _, _ := running.snapshot(); !done {
			sessionRefusal(response, "response "+id+" is still generating")
			return continuedTurn{}, false
		}
	}
	ctx := request.Context()
	stopped, found, err := runrecord.ResolveInteraction(ctx, h.repository, id)
	if err != nil {
		writeGenerationError(response, err)
		return continuedTurn{}, false
	}
	if !found {
		writeError(response, http.StatusNotFound, "not_found", "response "+id+" is not stored")
		return continuedTurn{}, false
	}
	description, admitted := h.interactionDescription()
	if !admitted {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "this server's model keeps no response identity")
		return continuedTurn{}, false
	}
	if stopped.Model != description.Identity.Model || stopped.Recipe != description.Identity.Recipe {
		sessionRefusal(response, fmt.Sprintf("response %s was generated by model %s (recipe %s); this server serves model %s (recipe %s)",
			id, stopped.Model, stopped.Recipe, description.Identity.Model, description.Identity.Recipe))
		return continuedTurn{}, false
	}
	if !stopped.Context.Valid() {
		sessionRefusal(response, "response "+id+" kept no token context: only a stopped or output-limited plain-text turn continues")
		return continuedTurn{}, false
	}
	tokens, err := runrecord.RequireInteractionContext(ctx, h.repository, stopped.Context)
	if err != nil {
		writeGenerationError(response, err)
		return continuedTurn{}, false
	}
	var parameters continuationParameters
	if err := json.Unmarshal(tokens.Sampling, &parameters); err != nil {
		writeGenerationError(response, err)
		return continuedTurn{}, false
	}
	requested, askedErr := json.Marshal(body.samplingParameters)
	unset, unsetErr := json.Marshal(samplingParameters{})
	saved, savedErr := json.Marshal(parameters.Sampling)
	if err := errors.Join(askedErr, unsetErr, savedErr); err != nil {
		writeGenerationError(response, err)
		return continuedTurn{}, false
	}
	if !bytes.Equal(requested, unset) && !bytes.Equal(requested, saved) {
		sessionRefusal(response, fmt.Sprintf("response %s sampled with %s; the request asks %s", id, saved, requested))
		return continuedTurn{}, false
	}
	transcript, err := runrecord.RequireInteractionTranscript(ctx, h.repository, stopped.Message)
	if err != nil {
		writeGenerationError(response, err)
		return continuedTurn{}, false
	}
	messages := responseMessages(transcript.Messages)
	last := len(messages) - 1
	if last < 1 || messages[last].Role != inference.ChatRoleAssistant {
		sessionRefusal(response, "response "+id+" holds no assistant turn to continue")
		return continuedTurn{}, false
	}
	ids := make([]tokenizer.TokenID, len(tokens.Tokens))
	for index, token := range tokens.Tokens {
		ids[index] = tokenizer.TokenID(token)
	}
	if vocabulary, ok := h.generator.(SamplingVocabulary); ok && slices.Contains(vocabulary.SamplingEOGTokens(), ids[len(ids)-1]) {
		sessionRefusal(response, "response "+id+" already ended its turn")
		return continuedTurn{}, false
	}
	return continuedTurn{
		turn: messages[:last], parent: stopped.Parent, prompt: nativePrompt{TokenIDs: ids},
		session: turnSession{parameters: parameters, prefix: messages[last].Content, continues: stopped.ID, held: tokens.Held, keep: true},
	}, true
}
