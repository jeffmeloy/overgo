package server

import (
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserDatasetImport drives a dataset file import from the
// library: an alpaca and ShareGPT file imports as conversations, registers
// and validates in one chain, names the formats it met, and previews under
// Datasets as a conversation; a file in no common format is refused with
// the record that failed.
func TestWebUIBrowserDatasetImport(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the dataset import leg uses Chromium through cmd/webui-lane")
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	sources := t.TempDir()
	chats := filepath.Join(sources, "chats.jsonl")
	if err := os.WriteFile(chats, []byte(`{"instruction":"Add","input":"2 and 3","output":"5"}`+"\n"+
		`{"conversations":[{"from":"human","value":"Hi"},{"from":"gpt","value":"Hello"}]}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	plain := filepath.Join(sources, "plain.jsonl")
	if err := os.WriteFile(plain, []byte(`{"text":"not a conversation"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	datasets := t.TempDir()
	handler := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{}), func(config *Config) {
		config.LibraryIntake.DatasetRoot = datasets
	})
	server := httptest.NewServer(handler)
	defer server.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
	settle(`!!document.querySelector('.composer textarea')`)
	check(`(() => { location.hash = 'library'; return true; })()`)
	settle(`!!document.querySelector('#panel-library.active input[aria-label="Chat dataset file on disk"]')`)
	check(`(() => {
  window.importPanel = () => document.querySelector('#panel-library');
  window.importRow = (name) => [...importPanel().querySelectorAll('tr')].find(row => row.firstChild && row.firstChild.textContent === name);
  window.importFile = (source, name) => {
    const field = (label) => importPanel().querySelector('input[aria-label="' + label + '"]');
    field('Chat dataset file on disk').value = source; field('Dataset name').value = name;
    [...importPanel().querySelectorAll('button')].find(button => button.textContent === 'Import a dataset file').click();
    return true;
  };
  return true;
})()`)

	// The file imports as conversations, registers and validates in one chain.
	check(`importFile(` + strconv.Quote(chats) + `, 'chats')`)
	settle(`!!importRow('chats') && importRow('chats').textContent.includes('2 conversations from alpaca, sharegpt') && importRow('chats').textContent.includes('validated: 1 example previewed')`)

	// Under Datasets it previews as a conversation.
	check(`(() => { location.hash = 'datasets'; return true; })()`)
	settle(`!![...document.querySelectorAll('#panel-datasets.active button')].find(button => button.textContent === 'chats')`)
	check(`(() => { [...document.querySelectorAll('#panel-datasets button')].find(button => button.textContent === 'chats').click(); return true; })()`)
	settle(`(() => {
  const turns = [...document.querySelectorAll('#panel-datasets .dataset-conversation .dataset-turn')];
  return turns.length === 2 && turns[0].querySelector('strong').textContent === 'user' && turns[0].querySelector('.preview-text').textContent === 'Add\n\n2 and 3' && turns[1].querySelector('strong').textContent === 'assistant';
})()`)

	// A file in no common format is refused with the record that failed.
	check(`(() => { location.hash = 'library'; return true; })()`)
	settle(`!!document.querySelector('#panel-library.active input[aria-label="Chat dataset file on disk"]')`)
	check(`importFile(` + strconv.Quote(plain) + `, 'plain')`)
	settle(`!!importRow('plain') && importRow('plain').textContent.includes('record 1: the record is none of')`)
	if _, err := os.Stat(filepath.Join(datasets, "plain")); !os.IsNotExist(err) {
		t.Fatalf("a refused import left its directory: %v", err)
	}
	webuilane.Leg(t, "dataset import leg", "an alpaca and ShareGPT file imported as conversations, registered and validated in one chain naming its formats, and previewed under Datasets as a conversation; a file in no common format was refused with the record that failed")
}
