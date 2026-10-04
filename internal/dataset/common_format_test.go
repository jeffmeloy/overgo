package dataset

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"overgo/internal/overgodb"
)

// TestCommonFormatsConvert: alpaca, ShareGPT and OpenAI-messages records,
// as JSON Lines or a JSON array (a byte order mark allowed), convert to one
// OpenAI messages conversation each, naming the formats met; a record in no
// format, an unknown speaker or a non-text part is refused with its
// position; an import writes each conversation as its own file under the
// root and registers the directory as structured, and refuses an existing
// directory or a name that is not one path element.
func TestCommonFormatsConvert(t *testing.T) {
	t.Parallel()
	alpaca := `{"instruction":"Add","input":"2 and 3","output":"5","system":"Be brief"}`
	shareGPT := `{"conversations":[{"from":"human","value":"Hi"},{"from":"gpt","value":"Hello"}]}`
	openAI := `{"messages":[{"role":"user","content":[{"type":"text","text":"Name "},{"type":"text","text":"a color"}]},{"role":"assistant","content":"Blue"}]}`
	want := []Conversation{
		{Messages: []ChatMessage{{Role: "system", Content: "Be brief"}, {Role: "user", Content: "Add\n\n2 and 3"}, {Role: "assistant", Content: "5"}}},
		{Messages: []ChatMessage{{Role: "user", Content: "Hi"}, {Role: "assistant", Content: "Hello"}}},
		{Messages: []ChatMessage{{Role: "user", Content: "Name a color"}, {Role: "assistant", Content: "Blue"}}},
	}
	for name, source := range map[string]string{
		"lines": alpaca + "\n" + shareGPT + "\r\n\n" + openAI + "\n",
		"array": byteOrderMark + "  [" + alpaca + "," + shareGPT + "," + openAI + "]",
	} {
		conversations, formats, err := convertCommonFormat(strings.NewReader(source))
		if err != nil || !reflect.DeepEqual(conversations, want) || !slices.Equal(formats, []string{"alpaca", "sharegpt", "openai-messages"}) {
			t.Fatalf("%s: conversations=%+v formats=%v err=%v", name, conversations, formats, err)
		}
	}
	for source, refusal := range map[string]string{
		alpaca + "\n" + `{"text":"plain"}`:                                "record 2: the record is none of",
		`{"conversations":[{"from":"tool","value":"x"}]}`:                 "record 1: conversation speaker \"tool\"",
		`{"messages":[{"role":"user","content":[{"type":"image_url"}]}]}`: "record 1: message content part \"image_url\" is not text",
		"  \n": "the file holds no records",
	} {
		if _, _, err := convertCommonFormat(strings.NewReader(source)); err == nil || !strings.Contains(err.Error(), refusal) {
			t.Fatalf("source %q: err=%v, want %q", source, err, refusal)
		}
	}

	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	source := filepath.Join(t.TempDir(), "chats.jsonl")
	if err := os.WriteFile(source, []byte(alpaca+"\n"+shareGPT+"\n"+openAI+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	registered, formats, err := ImportCommonFormat(t.Context(), store, "chats", source, root)
	if err != nil || registered.Files != 3 || !registered.Dataset.Valid() || len(formats) != 3 {
		t.Fatalf("import = %+v formats=%v err=%v", registered, formats, err)
	}
	var second Conversation
	data, err := os.ReadFile(filepath.Join(root, "chats", "2.json"))
	if err != nil || json.Unmarshal(data, &second) != nil || !reflect.DeepEqual(second, want[1]) {
		t.Fatalf("second record = %s, %v", data, err)
	}
	catalog, found, err := ResolveCatalog(t.Context(), store)
	if err != nil || !found || !slices.ContainsFunc(catalog.Catalog, func(entry CatalogEntry) bool {
		return entry.Name == "chats" && entry.Modality == ModalityStructured
	}) {
		t.Fatalf("catalog = %+v found=%v err=%v", catalog, found, err)
	}
	if _, _, err := ImportCommonFormat(t.Context(), store, "chats", source, root); err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("a second import into the same directory: %v", err)
	}
	if _, _, err := ImportCommonFormat(t.Context(), store, "../escape", source, root); err == nil || !strings.Contains(err.Error(), "one path element") {
		t.Fatalf("an import outside the root: %v", err)
	}
}
