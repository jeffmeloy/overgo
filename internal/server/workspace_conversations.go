package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"maps"
	"net/http"
	"slices"
	"strconv"
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
	Recipe   artifact.ID `json:"recipe"`
	Archived bool        `json:"archived,omitzero"`
}

type conversationListResponse struct {
	Conversations []conversationSummary `json:"conversations"`
	Next          string                `json:"next,omitzero"`
}

type conversationMessagesResponse struct {
	Response string                `json:"response"`
	Root     string                `json:"root"`
	Model    artifact.ID           `json:"model"`
	Recipe   artifact.ID           `json:"recipe"`
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
	if !requireMethod(response, request, http.MethodGet) || !h.conversationRepository(response, request) {
		return
	}
	values := request.URL.Query()
	search, view := strings.ToLower(strings.TrimSpace(values.Get("q"))), values.Get("view")
	if view != "" && view != "active" && view != "archived" {
		writeInvalidRequest(response, errors.New("conversation view must be active or archived"))
		return
	}
	limit := h.config.MaxStoredResponses
	if value := values.Get("limit"); value != "" {
		parsed, err := strconv.Atoi(value)
		if err != nil || parsed <= 0 || parsed > limit {
			writeInvalidRequest(response, errors.New("conversation limit is outside the configured bound"))
			return
		}
		limit = parsed
	}
	query := overgodb.DocumentQuery{
		Contracts:     []artifact.DocumentContract{{Kind: artifact.KindEvidence, MediaType: runrecord.InteractionMediaType, Schema: runrecord.InteractionSchema}},
		AliasPrefixes: []string{runrecord.InteractionResponseAliasRoot}, Order: overgodb.DocumentNewestFirst,
	}
	if value := values.Get("cursor"); value != "" {
		var continuation conversationCursor
		data, err := base64.RawURLEncoding.DecodeString(value)
		if err != nil || json.Unmarshal(data, &continuation) != nil || continuation.Search != search || continuation.View != view {
			writeInvalidRequest(response, errors.New("conversation cursor does not match this search"))
			return
		}
		cursor, err := overgodb.ParseQueryCursor(continuation.Page)
		if err != nil {
			writeInvalidRequest(response, err)
			return
		}
		query.Cursor = &cursor
	}
	head, _ := h.repository.Head()
	if query.Cursor != nil && query.Cursor.Head != head {
		writeError(response, http.StatusConflict, "history_changed", "History changed. Reload the list to continue.")
		return
	}
	result := conversationListResponse{Conversations: []conversationSummary{}}
	known := map[artifact.ID]runrecord.Interaction{}
	for {
		query.MaxResults = limit - len(result.Conversations)
		page, err := overgodb.VisitDecodedDocuments(request.Context(), h.repository, query, runrecord.ParseInteraction,
			func(_ overgodb.DocumentView, interaction runrecord.Interaction) error {
				leaf, err := h.conversationLeaf(request.Context(), interaction)
				if err != nil || !leaf {
					return err
				}
				chain, err := h.conversationChain(request.Context(), interaction, known)
				if err != nil {
					return err
				}
				root := chain[len(chain)-1]
				summary := conversationSummary{Root: root.Response, Latest: interaction.Response, Turns: len(chain), Model: interaction.Model, Recipe: interaction.Recipe}
				label, found, err := resolveConversationLabel(request.Context(), h.repository, root.Response)
				if err != nil {
					return err
				}
				if found {
					summary.Title, summary.Archived = label.Title, label.Archived
				}
				if summary.Title == "" {
					summary.Title = h.conversationTitle(request.Context(), root)
				}
				if (view == "active" && summary.Archived) || (view == "archived" && !summary.Archived) || !strings.Contains(strings.ToLower(summary.Title), search) {
					return nil
				}
				result.Conversations = append(result.Conversations, summary)
				return nil
			})
		current, _ := h.repository.Head()
		if current != head {
			writeError(response, http.StatusConflict, "history_changed", "History changed. Reload the list to continue.")
			return
		}
		if err != nil {
			writeGenerationError(response, err)
			return
		}
		query.Cursor = page.Next
		if page.Next == nil || len(result.Conversations) == limit {
			break
		}
	}
	if query.Cursor != nil {
		cursor, err := encodeNextCursor(query.Cursor)
		if err != nil {
			writeGenerationError(response, err)
			return
		}
		data, err := json.Marshal(conversationCursor{Search: search, View: view, Page: cursor})
		if err != nil {
			writeGenerationError(response, err)
			return
		}
		result.Next = base64.RawURLEncoding.EncodeToString(data)
	}
	writeJSON(response, http.StatusOK, result)
}

// Bind the store continuation to the server-side filters as well.
type conversationCursor struct {
	Search string `json:"q"`
	View   string `json:"view"`
	Page   string `json:"page"`
}

func (h *Handler) conversationRepository(response http.ResponseWriter, request *http.Request) bool {
	if h.repository == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "interaction repository is unavailable")
		return false
	}
	_, ok := h.requireBrowseStore(response, request)
	return ok
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
	if !requireMethod(response, request, http.MethodGet) || !h.conversationRepository(response, request) {
		return
	}
	responseID := request.URL.Query().Get("response")
	interaction, found, err := runrecord.ResolveInteraction(request.Context(), h.repository, responseID)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	if !found {
		writeError(response, http.StatusNotFound, "not_found", "conversation not found")
		return
	}
	chain, err := h.conversationChain(request.Context(), interaction, map[artifact.ID]runrecord.Interaction{})
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	previous := ""
	if len(chain) > 1 {
		previous = chain[1].Response
	}
	messages, err := h.chainMessages(request.Context(), chain)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	status, failure := h.responseTerminal(request.Context(), interaction)
	if turn, found := h.inflight.lookup(responseID); found {
		_, _, final, failed, _ := turn.snapshot()
		status, failure = final.Status, failed
	}
	writeJSON(response, http.StatusOK, conversationMessagesResponse{
		Status: status, Previous: previous, Failure: failure, Response: responseID, Root: chain[len(chain)-1].Response,
		Model: interaction.Model, Recipe: interaction.Recipe, Messages: messages,
	})
}

// conversationLabel records a title or an archival for the conversation
// rooted at the named response.
func (h *Handler) conversationLabel(response http.ResponseWriter, request *http.Request) {
	var body conversationLabelRequest
	if !requireMethod(response, request, http.MethodPost) || !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	if !h.conversationRepository(response, request) {
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
	current, found, err := resolveConversationLabel(ctx, repository, value.Root)
	if err != nil {
		return conversationLabelRecord{}, err
	}
	if found && current.ID == value.ID {
		return value, nil
	}
	binding := artifact.AliasBinding{Name: conversationLabelAliasRoot + value.Root, Target: value.ID}
	if found {
		binding.Previous = &current.ID
	}
	// A title/archive state may recur. Key the transition by the observed
	// store head so replay of an earlier toggle cannot suppress this write.
	head, _ := repository.Head()
	batch, err := conversationLabelCodec.Batch("conversation-label/"+value.ID.String()+"/"+head.String(), value, nil, []artifact.AliasBinding{binding})
	if err != nil {
		return conversationLabelRecord{}, err
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return conversationLabelRecord{}, err
	}
	return value, nil
}

// resolveConversationLabel reads the current label of the conversation
// rooted at the named response.
func resolveConversationLabel(ctx context.Context, reader artifact.Reader, root string) (conversationLabelRecord, bool, error) {
	return conversationLabelCodec.Resolve(ctx, reader, conversationLabelAliasRoot+root)
}

// Parent membership is global, not relative to the current list page.
// A terminal revision also depends on its reservation; only actual active
// child turns make a response cease to be a conversation leaf.
func (h *Handler) conversationLeaf(ctx context.Context, interaction runrecord.Interaction) (bool, error) {
	edges, err := h.repository.Children(ctx, interaction.ID)
	if err != nil {
		return false, err
	}
	for _, edge := range edges {
		descriptor, found, err := h.repository.Artifact(ctx, edge.Child)
		if err != nil {
			return false, err
		}
		if !found || descriptor.MediaType != runrecord.InteractionMediaType || descriptor.Schema != runrecord.InteractionSchema {
			continue
		}
		child, err := runrecord.RequireInteraction(ctx, h.repository, edge.Child)
		if err != nil {
			return false, err
		}
		if child.Parent != interaction.ID {
			continue
		}
		active, found, err := runrecord.ResolveInteraction(ctx, h.repository, child.Response)
		if err != nil {
			return false, err
		}
		if found && active.ID == child.ID {
			return false, nil
		}
	}
	return true, nil
}

// conversationChain returns the exact immutable branch, newest first.
func (h *Handler) conversationChain(ctx context.Context, latest runrecord.Interaction, known map[artifact.ID]runrecord.Interaction) ([]runrecord.Interaction, error) {
	var chain []runrecord.Interaction
	seen := map[artifact.ID]bool{}
	for current := latest; ; {
		if seen[current.ID] {
			return nil, errors.New("conversation parent chain contains a cycle")
		}
		seen[current.ID] = true
		known[current.ID] = current
		chain = append(chain, current)
		if !current.Parent.Valid() {
			return chain, nil
		}
		parent, found := known[current.Parent]
		if !found {
			var err error
			parent, err = runrecord.RequireInteraction(ctx, h.repository, current.Parent)
			if err != nil {
				return nil, err
			}
		}
		current = parent
	}
}
