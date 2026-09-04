package server

import (
	"context"
	"errors"
	"net/http"
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
	Response string                         `json:"response"`
	Root     string                         `json:"root"`
	Messages []runrecord.InteractionMessage `json:"messages"`
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
	messages, _, found := h.loadResponseInteraction(request.Context(), responseID)
	if !found {
		writeError(response, http.StatusNotFound, "not_found", "conversation not found")
		return
	}
	interaction, _, _ := runrecord.ResolveInteraction(request.Context(), h.repository, responseID)
	root := interaction
	for root.Parent.Valid() {
		parent, err := runrecord.RequireInteraction(request.Context(), h.repository, root.Parent)
		if err != nil {
			break
		}
		root = parent
	}
	writeJSON(response, http.StatusOK, conversationMessagesResponse{
		Response: responseID, Root: root.Response, Messages: interactionMessages(messages),
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

// ---- in-flight turns ----
// A stored streamed response keeps generating when its client drops: the
// text so far and its completion are held here, keyed by response id, so
// a page that reloads reattaches through /interactions/follow, receives
// what it missed as one delta, then follows live until the turn completes
// and its interaction is durable. Entries leave once followed after
// completion, bounded by the response store size.
type inflightTurn struct {
	mu     sync.Mutex
	text   strings.Builder
	done   bool
	failed string
	final  any
	notify chan struct{}
}

type inflightRegistry struct {
	mu    sync.Mutex
	turns map[string]*inflightTurn
}

func (r *inflightRegistry) begin(responseID string, bound int) *inflightTurn {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.turns == nil {
		r.turns = map[string]*inflightTurn{}
	}
	for id, turn := range r.turns {
		if len(r.turns) < bound {
			break
		}
		if turn.done {
			delete(r.turns, id)
		}
	}
	turn := &inflightTurn{notify: make(chan struct{})}
	r.turns[responseID] = turn
	return turn
}

func (r *inflightRegistry) lookup(responseID string) (*inflightTurn, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	turn, found := r.turns[responseID]
	return turn, found
}

func (turn *inflightTurn) append(piece string) {
	turn.mu.Lock()
	turn.text.WriteString(piece)
	notify := turn.notify
	turn.notify = make(chan struct{})
	turn.mu.Unlock()
	close(notify)
}

func (turn *inflightTurn) finish(final any, failure string) {
	turn.mu.Lock()
	turn.done, turn.final, turn.failed = true, final, failure
	notify := turn.notify
	turn.mu.Unlock()
	close(notify)
}

// snapshot returns the text so far, whether the turn is done, and a channel
// that closes at the next change.
func (turn *inflightTurn) snapshot() (string, bool, any, string, <-chan struct{}) {
	turn.mu.Lock()
	defer turn.mu.Unlock()
	return turn.text.String(), turn.done, turn.final, turn.failed, turn.notify
}

// conversationFollow reattaches a page to a turn: the text generated so far
// arrives as one delta, further deltas follow live, and the completion
// event closes the stream; a turn already durable replays from its record.
func (h *Handler) conversationFollow(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	responseID := request.URL.Query().Get("response")
	turn, inflight := h.inflight.lookup(responseID)
	if !inflight {
		messages, _, found := h.loadResponseInteraction(request.Context(), responseID)
		if !found {
			writeError(response, http.StatusNotFound, "not_found", "turn not found")
			return
		}
		flusher, ok := beginSSE(response)
		if !ok {
			return
		}
		stream := newSSEEmitter(request.Context(), response, flusher)
		text := ""
		if len(messages) != 0 {
			text = messages[len(messages)-1].Content
		}
		_ = stream.named("response.output_text.delta", responsesStreamEvent{Type: "response.output_text.delta", ResponseID: responseID, Delta: text})
		_ = stream.named("response.completed", responsesStreamEvent{Type: "response.completed", Response: responsesProgress{ID: responseID, Object: "response", Status: "completed"}})
		return
	}
	flusher, ok := beginSSE(response)
	if !ok {
		return
	}
	stream := newSSEEmitter(request.Context(), response, flusher)
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
			if failed != "" {
				_ = stream.named("response.failed", responsesStreamEvent{Type: "response.failed", ResponseID: responseID, Delta: failed})
				return
			}
			_ = stream.named("response.completed", responsesStreamEvent{Type: "response.completed", Response: final})
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
