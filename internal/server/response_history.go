package server

import (
	"container/list"
	"slices"
	"sync"

	"llamacpp2go/internal/inference"
)

const (
	defaultStoredResponses    = 128
	defaultResponseStoreBytes = 64 << 20
)

type responseHistoryEntry struct {
	id       string
	messages []inference.ChatMessage
	size     int
}

type responseHistoryStore struct {
	mu         sync.Mutex
	entries    map[string]*list.Element
	order      *list.List
	maxEntries int
	maxBytes   int
	bytes      int
}

func newResponseHistoryStore(maxEntries, maxBytes int) *responseHistoryStore {
	return &responseHistoryStore{
		entries: make(map[string]*list.Element, maxEntries),
		order:   list.New(), maxEntries: maxEntries, maxBytes: maxBytes,
	}
}

func (s *responseHistoryStore) get(id string) ([]inference.ChatMessage, bool) {
	if s == nil {
		return nil, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	element, ok := s.entries[id]
	if !ok {
		return nil, false
	}
	entry := element.Value.(*responseHistoryEntry)
	return cloneResponseMessages(entry.messages), true
}

func (s *responseHistoryStore) put(id string, messages []inference.ChatMessage) bool {
	if s == nil || id == "" || len(messages) == 0 {
		return false
	}
	cloned := cloneResponseMessages(messages)
	size := responseMessagesSize(id, cloned)
	if size > s.maxBytes {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if existing, ok := s.entries[id]; ok {
		entry := existing.Value.(*responseHistoryEntry)
		s.bytes -= entry.size
		s.order.Remove(existing)
		delete(s.entries, id)
	}
	entry := &responseHistoryEntry{id: id, messages: cloned, size: size}
	s.entries[id] = s.order.PushBack(entry)
	s.bytes += size
	for len(s.entries) > s.maxEntries || s.bytes > s.maxBytes {
		oldest := s.order.Front()
		if oldest == nil {
			break
		}
		removed := oldest.Value.(*responseHistoryEntry)
		delete(s.entries, removed.id)
		s.bytes -= removed.size
		s.order.Remove(oldest)
	}
	return true
}

func cloneResponseMessages(messages []inference.ChatMessage) []inference.ChatMessage {
	cloned := slices.Clone(messages)
	for index := range cloned {
		cloned[index].Media = slices.Clone(messages[index].Media)
		cloned[index].ToolCalls = slices.Clone(messages[index].ToolCalls)
	}
	return cloned
}

func responseMessagesSize(id string, messages []inference.ChatMessage) int {
	size := len(id)
	for _, message := range messages {
		size += len(message.Role) + len(message.Content) + len(message.ReasoningContent) +
			len(message.Name) + len(message.ToolCallID)
		for _, media := range message.Media {
			size += len(media.Type) + len(media.Data) + len(media.Format) + 8
		}
		for _, call := range message.ToolCalls {
			size += len(call.ID) + len(call.Type) + len(call.Function.Name) +
				len(call.Function.Arguments)
		}
	}
	return size
}
