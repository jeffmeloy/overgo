package server

import (
	"context"
	"errors"
	"maps"
	"net/http"
	"slices"
	"strings"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// Durable conversations (professional GUI campaign, gui-conversations/
// durable-sessions). A conversation is a chain of stored responses: every
// turn is an interaction record whose parent is the previous turn, so the
// server, not the browser, owns the thread. The routes here list those
// chains as conversations, materialize one chain's messages, label a
// conversation (title, archived) as a new record, and let a page reattach
// to a turn still being generated.

// conversationSummary is one chain as the front page lists it.
type conversationSummary struct {
	Root     string      `json:"root"`
	Latest   string      `json:"latest"`
	Title    string      `json:"title"`
	Turns    int         `json:"turns"`
	Model    artifact.ID `json:"model"`
	Archived bool        `json:"archived,omitzero"`
}

type conversationListResponse struct {
	Conversations []conversationSummary `json:"conversations"`
}

type conversationMessagesResponse struct {
	Response string                `json:"response"`
	Root     string                `json:"root"`
	Status   string                `json:"status"`
	Previous string                `json:"previous,omitzero"`
	Failure  string                `json:"failure,omitzero"`
	Messages []conversationMessage `json:"messages"`
}

type conversationLabelRequest struct {
	Root     string `json:"root"`
	Title    string `json:"title"`
	Archived bool   `json:"archived"`
}

// conversations lists the store's response chains newest first: a chain
// is reported once, by its latest response, with the title its label holds
// or the first line of its first user message.
func (h *Handler) conversations(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	if h.repository == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "interaction repository is unavailable")
		return
	}
	interactions, err := listInteractions(request.Context(), h.repository, h.config.MaxStoredResponses)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	byID := make(map[artifact.ID]runrecord.Interaction, len(interactions))
	isParent := make(map[artifact.ID]bool, len(interactions))
	for _, interaction := range interactions {
		byID[interaction.ID] = interaction
		if interaction.Parent.Valid() {
			isParent[interaction.Parent] = true
		}
	}
	result := conversationListResponse{Conversations: []conversationSummary{}}
	for _, interaction := range interactions {
		if isParent[interaction.ID] {
			continue // not the latest turn of its chain
		}
		root, turns := conversationRoot(byID, interaction)
		summary := conversationSummary{Root: root.Response, Latest: interaction.Response, Turns: turns, Model: interaction.Model}
		if label, found, err := resolveConversationLabel(request.Context(), h.repository, root.Response); err == nil && found {
			summary.Title, summary.Archived = label.Title, label.Archived
		}
		if summary.Title == "" {
			summary.Title = h.conversationTitle(request.Context(), root)
		}
		result.Conversations = append(result.Conversations, summary)
	}
	writeJSON(response, http.StatusOK, result)
}

// conversationTitle derives a title from the chain's first user message.
func (h *Handler) conversationTitle(ctx context.Context, root runrecord.Interaction) string {
	transcript, err := runrecord.RequireInteractionTranscript(ctx, h.repository, root.Message)
	if err != nil {
		return root.Response
	}
	for _, message := range transcript.Messages {
		if message.Role == string(inference.ChatRoleUser) && strings.TrimSpace(message.Content) != "" {
			for line := range strings.Lines(strings.TrimSpace(message.Content)) {
				return strings.TrimSpace(line)
			}
		}
	}
	return root.Response
}

// conversationMessages materializes one chain's visible messages up to the
// named response, the transcript a resumed conversation renders.
func (h *Handler) conversationMessages(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	responseID := request.URL.Query().Get("response")
	interaction, found, err := runrecord.ResolveInteraction(request.Context(), h.repository, responseID)
	if err != nil || !found {
		writeError(response, http.StatusNotFound, "not_found", "conversation not found")
		return
	}
	root := interaction
	for parent, ok := h.parentInteraction(request, root); ok; parent, ok = h.parentInteraction(request, root) {
		root = parent
	}
	previous := ""
	if parent, found := h.parentInteraction(request, interaction); found {
		previous = parent.Response
	}
	status, failure := h.responseTerminal(request.Context(), interaction)
	if turn, found := h.inflight.lookup(responseID); found {
		_, _, final, failed, _ := turn.snapshot()
		status, failure = final.Status, failed
	}
	writeJSON(response, http.StatusOK, conversationMessagesResponse{
		Status: status, Previous: previous, Failure: failure, Response: responseID, Root: root.Response, Messages: h.chainMessages(request, interaction),
	})
}

// conversationLabel records a title or an archival for the conversation
// rooted at the named response.
func (h *Handler) conversationLabel(response http.ResponseWriter, request *http.Request) {
	var body conversationLabelRequest
	if !requireMethod(response, request, http.MethodPost) || !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	if h.repository == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "interaction repository is unavailable")
		return
	}
	if _, found, err := runrecord.ResolveInteraction(request.Context(), h.repository, body.Root); err != nil || !found {
		writeError(response, http.StatusNotFound, "not_found", "conversation not found")
		return
	}
	label, err := publishConversationLabel(context.WithoutCancel(request.Context()), h.repository, conversationLabelRecord{
		Root: body.Root, Title: strings.TrimSpace(body.Title), Archived: body.Archived,
	})
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	writeJSON(response, http.StatusOK, label)
}

// Stored turns retain execution independently of their first HTTP connection.
// This registry owns cancellation and wakes followers; the interaction trace
// owns durable terminal state once an entry is evicted or the server restarts.
type inflightTurn struct {
	mu     sync.Mutex
	text   strings.Builder
	done   bool
	failed string
	final  responsesResponse
	cancel context.CancelFunc
	notify chan struct{}
}

type inflightRegistry struct {
	mu     sync.Mutex
	turns  map[string]*inflightTurn
	closed bool
}

func (r *inflightRegistry) begin(responseID string, bound int, cancel context.CancelFunc) *inflightTurn {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil
	}
	if r.turns == nil {
		r.turns = map[string]*inflightTurn{}
	}
	for id, turn := range r.turns {
		if len(r.turns) < bound {
			break
		}
		_, done, _, _, _ := turn.snapshot()
		if done {
			delete(r.turns, id)
		}
	}
	if len(r.turns) >= bound {
		return nil
	}
	turn := &inflightTurn{
		notify: make(chan struct{}), cancel: cancel,
		final: responsesResponse{ID: responseID, Object: "response", Status: "in_progress"},
	}
	r.turns[responseID] = turn
	return turn
}

func (r *inflightRegistry) lookup(responseID string) (*inflightTurn, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	turn, found := r.turns[responseID]
	return turn, found
}

func (r *inflightRegistry) shutdown() {
	r.mu.Lock()
	r.closed = true
	turns := slices.Collect(maps.Values(r.turns))
	r.mu.Unlock()
	for _, turn := range turns {
		turn.stop()
	}
}

func (turn *inflightTurn) stop() responsesResponse {
	turn.mu.Lock()
	final, cancel := turn.final, turn.cancel
	turn.cancel = nil
	if !turn.done {
		final.Status = "cancelling"
	}
	turn.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	return final
}

func (turn *inflightTurn) append(piece string) {
	turn.mu.Lock()
	if turn.done {
		turn.mu.Unlock()
		return
	}
	turn.text.WriteString(piece)
	notify := turn.notify
	turn.notify = make(chan struct{})
	turn.mu.Unlock()
	close(notify)
}

func (turn *inflightTurn) finish(final responsesResponse, failure string) {
	turn.mu.Lock()
	if turn.done {
		turn.mu.Unlock()
		return
	}
	turn.done, turn.final, turn.failed = true, final, failure
	notify := turn.notify
	turn.mu.Unlock()
	close(notify)
}

// snapshot returns the text so far, terminal state, and a change notification.
func (turn *inflightTurn) snapshot() (string, bool, responsesResponse, string, <-chan struct{}) {
	turn.mu.Lock()
	defer turn.mu.Unlock()
	return turn.text.String(), turn.done, turn.final, turn.failed, turn.notify
}

// conversationCancel is explicit execution cancellation, independent of the
// stream's connection. A repeated request returns the current terminal state.
func (h *Handler) conversationCancel(response http.ResponseWriter, request *http.Request) {
	var body struct {
		Response string `json:"response"`
		Model    string `json:"model"`
	}
	if !requireMethod(response, request, http.MethodPost) ||
		!h.decodeProtocolJSON(response, request, &body, func() string { return body.Model }) {
		return
	}
	if body.Response == "" || body.Model == "" {
		writeInvalidRequestMessage(response, "response and model are required")
		return
	}
	if turn, found := h.inflight.lookup(body.Response); found {
		writeJSON(response, http.StatusOK, turn.stop())
		return
	}
	_, id, found := h.loadResponseInteraction(request.Context(), body.Response)
	if !found {
		writeError(response, http.StatusNotFound, "not_found", "turn not found")
		return
	}
	interaction, err := runrecord.RequireInteraction(request.Context(), h.repository, id)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	status, _ := h.responseTerminal(request.Context(), interaction)
	writeJSON(response, http.StatusOK, responsesProgress{ID: body.Response, Object: "response", Status: status})
}

// conversationFollow replays the current turn, then follows it to a confirmed
// terminal event. It never starts a new generation.
func (h *Handler) conversationFollow(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) || !h.requireModel(response, request.URL.Query().Get("model")) {
		return
	}
	responseID := request.URL.Query().Get("response")
	turn, inflight := h.inflight.lookup(responseID)
	if !inflight {
		_, id, found := h.loadResponseInteraction(request.Context(), responseID)
		if !found {
			writeError(response, http.StatusNotFound, "not_found", "turn not found")
			return
		}
		interaction, err := runrecord.RequireInteraction(request.Context(), h.repository, id)
		if err != nil {
			writeGenerationError(response, err)
			return
		}
		transcript, err := runrecord.RequireInteractionTranscript(request.Context(), h.repository, interaction.Message)
		if err != nil {
			writeGenerationError(response, err)
			return
		}
		status, failure := h.responseTerminal(request.Context(), interaction)
		flusher, ok := beginSSE(response)
		if !ok {
			return
		}
		stream := newSSEEmitter(request.Context(), response, flusher)
		final := responsesProgress{ID: responseID, Object: "response", Status: status}
		_ = stream.named("response.created", responsesStreamEvent{Type: "response.created", Response: final})
		// Only this turn's assistant text is output; an interrupted initial
		// record holds the user's prompt and must never replay it as an answer.
		for _, message := range transcript.Messages {
			if message.Role == string(inference.ChatRoleAssistant) {
				if err := stream.named("response.output_text.delta", responsesStreamEvent{Type: "response.output_text.delta", ResponseID: responseID, Delta: message.Content}); err != nil {
					return
				}
			}
		}
		_ = stream.named("response."+status, responsesStreamEvent{Type: "response." + status, Response: final, Delta: failure})
		return
	}
	flusher, ok := beginSSE(response)
	if !ok {
		return
	}
	stream := newSSEEmitter(request.Context(), response, flusher)
	if err := stream.named("response.created", responsesStreamEvent{Type: "response.created", Response: responsesProgress{ID: responseID, Object: "response", Status: "in_progress"}}); err != nil {
		return
	}
	sent := 0
	for {
		text, done, final, failed, changed := turn.snapshot()
		if len(text) > sent {
			if err := stream.named("response.output_text.delta", responsesStreamEvent{Type: "response.output_text.delta", ResponseID: responseID, Delta: text[sent:]}); err != nil {
				return
			}
			sent = len(text)
		}
		if done {
			_ = stream.named("response."+final.Status, responsesStreamEvent{Type: "response." + final.Status, Response: final, Delta: failed})
			return
		}
		select {
		case <-changed:
		case <-request.Context().Done():
			return
		}
	}
}

const (
	// conversationLabelMediaType identifies a conversation label document.
	conversationLabelMediaType = "application/vnd.overgo.conversation-label+json"
	// conversationLabelSchema identifies the label contract.
	conversationLabelSchema = "overgo/conversation-label/v1"
	// conversationLabelAliasRoot scopes the current label of a conversation
	// by the response identifier of its root interaction.
	conversationLabelAliasRoot = "conversation/label/"
)

// conversationLabelRecord is the operator-facing state of one conversation:
// a title and whether it is archived. Interactions are append-only
// evidence, so a rename or an archival is a new label document aliased to
// the conversation's root response, never an edit of a record.
type conversationLabelRecord struct {
	Version  uint16      `json:"version"`
	Root     string      `json:"root"`
	Title    string      `json:"title,omitzero"`
	Archived bool        `json:"archived,omitzero"`
	ID       artifact.ID `json:"-"`
}

var conversationLabelCodec = artifact.JSONDocumentCodec(
	"conversation label", artifact.KindEvidence, conversationLabelMediaType, conversationLabelSchema,
	func(value *conversationLabelRecord) error {
		if value.Version != artifact.InitialDocumentVersion || strings.TrimSpace(value.Root) != value.Root || value.Root == "" ||
			strings.TrimSpace(value.Title) != value.Title {
			return errors.New("conversation label: invalid document")
		}
		return nil
	},
	func(value conversationLabelRecord) artifact.ID { return value.ID },
	func(value *conversationLabelRecord, id artifact.ID) { value.ID = id },
	func(value conversationLabelRecord) conversationLabelRecord { return value },
)

// publishConversationLabel commits a label for the conversation rooted at
// the named response and points the conversation's alias at it.
func publishConversationLabel(ctx context.Context, repository artifact.Repository, value conversationLabelRecord) (conversationLabelRecord, error) {
	if ctx == nil || repository == nil {
		return conversationLabelRecord{}, errors.New("conversation label: repository is absent")
	}
	value.Version = artifact.InitialDocumentVersion
	value, err := conversationLabelCodec.New(value)
	if err != nil {
		return conversationLabelRecord{}, err
	}
	batch, err := conversationLabelCodec.Batch("conversation-label/"+value.ID.String(), value, nil,
		[]artifact.AliasBinding{{Name: conversationLabelAliasRoot + value.Root, Target: value.ID}})
	if err != nil {
		return conversationLabelRecord{}, err
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return conversationLabelRecord{}, err
	}
	return value, nil
}

// resolveConversationLabel reads the current label of the conversation
// rooted at the named response.
func resolveConversationLabel(ctx context.Context, reader artifact.Reader, root string) (conversationLabelRecord, bool, error) {
	return conversationLabelCodec.Resolve(ctx, reader, conversationLabelAliasRoot+root)
}

// listInteractions visits the response interactions the store holds,
// newest first, up to limit.
func listInteractions(ctx context.Context, store overgodb.DocumentReader, limit int) ([]runrecord.Interaction, error) {
	var result []runrecord.Interaction
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts:     []artifact.DocumentContract{{Kind: artifact.KindEvidence, MediaType: runrecord.InteractionMediaType, Schema: runrecord.InteractionSchema}},
		AliasPrefixes: []string{runrecord.InteractionResponseAliasRoot},
		Order:         overgodb.DocumentNewestFirst, MaxResults: limit,
	}, runrecord.ParseInteraction, func(_ overgodb.DocumentView, value runrecord.Interaction) error {
		result = append(result, value)
		return nil
	})
	return result, err
}

// conversationRoot walks a turn's parents to the first turn of its chain
// and reports how many turns the chain holds.
func conversationRoot(byID map[artifact.ID]runrecord.Interaction, latest runrecord.Interaction) (root runrecord.Interaction, turns int) {
	for current, found := latest, true; found; current, found = byID[current.Parent] {
		root = current
		turns++
	}
	return root, turns
}
