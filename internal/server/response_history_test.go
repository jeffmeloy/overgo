package server

import (
	"testing"

	"overgo/internal/inference"
)

func TestResponseHistoryStoreEvictsOldestAndClones(t *testing.T) {
	store := newResponseHistoryStore(2, 1024)
	first := []inference.ChatMessage{{Role: "user", Content: "one"}}
	if !store.put("resp_1", first) {
		t.Fatal("first response was not stored")
	}
	first[0].Content = "changed"
	if !store.put("resp_2", []inference.ChatMessage{{Role: "user", Content: "two"}}) ||
		!store.put("resp_3", []inference.ChatMessage{{Role: "user", Content: "three"}}) {
		t.Fatal("response was not stored")
	}
	if _, ok := store.get("resp_1"); ok {
		t.Fatal("oldest response was retained")
	}
	messages, ok := store.get("resp_2")
	if !ok || len(messages) != 1 || messages[0].Content != "two" {
		t.Fatalf("stored messages = %+v, %v", messages, ok)
	}
	messages[0].Content = "mutated"
	messages, _ = store.get("resp_2")
	if messages[0].Content != "two" {
		t.Fatalf("store returned aliased messages: %+v", messages)
	}
}

func TestResponseHistoryStoreRejectsOversizedEntry(t *testing.T) {
	store := newResponseHistoryStore(2, 8)
	if store.put("resp_1", []inference.ChatMessage{{Role: "user", Content: "oversized"}}) {
		t.Fatal("oversized response was stored")
	}
}
