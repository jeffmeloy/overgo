package server

import (
	"io/fs"
	"regexp"
	"strings"
	"testing"
)

// Composer ratchet (professional GUI campaign, gui-simplify/one-composer):
// every surface that asks the served model for something goes through the
// one composer and the one thread renderer in composer.js, over the one
// event vocabulary this package declares. The ceilings below are the
// measured values when the row landed; they only tighten.
const (
	webuiStreamReaderCeiling = 1    // response.body.getReader(): boot.js sseEvents, the one stream reader
	webuiRawFetchCeiling     = 4    // fetch(: boot.js api client only
	webuiAPIStreamCeiling    = 2    // api.stream(: chat and speech
	webuiJavaScriptCeiling   = 4796 // total lines under webui/
)

func webuiJavaScript(t *testing.T) map[string]string {
	t.Helper()
	sources := map[string]string{}
	if err := fs.WalkDir(webuiFS, ".", func(name string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !entry.IsDir() && strings.HasSuffix(name, ".js") {
			data, err := fs.ReadFile(webuiFS, name)
			if err != nil {
				return err
			}
			sources[name] = string(data)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	return sources
}

func TestWebUIComposerBudget(t *testing.T) {
	sources := webuiJavaScript(t)
	composer, ok := sources["composer.js"]
	if !ok {
		t.Fatal("composer.js is absent")
	}
	for _, needle := range []string{"overgo.thread = thread", "overgo.composer = composer", "overgo.streams = {", "function toolCard", "function mediaCard", "function errorRow", "function thinking"} {
		if !strings.Contains(composer, needle) {
			t.Errorf("composer.js lacks %s", needle)
		}
	}
	// The vocabulary the renderer documents is the one this package declares.
	for _, event := range streamEventTypes {
		if !strings.Contains(composer, `case "`+event+`":`) {
			t.Errorf("composer.js does not render stream event %q", event)
		}
	}
	// Every surface that talks to the served model uses the shared pieces.
	for _, module := range []string{"mod/chat.js", "mod/agent.js", "mod/image.js", "mod/video.js", "mod/speech.js"} {
		source := sources[module]
		if !strings.Contains(source, "overgo.generationTab(") && (!strings.Contains(source, "overgo.composer(") || !strings.Contains(source, "overgo.thread(")) {
			t.Errorf("%s does not use the shared composer and thread", module)
		}
		if strings.Contains(source, "getReader()") || strings.Contains(source, "FileReader") {
			t.Errorf("%s parses a wire format or files itself instead of through composer.js", module)
		}
	}
	// The loader lists composer.js after md.js: the thread renders markdown.
	boot := sources["boot.js"]
	if strings.Index(boot, `"/md.js"`) < 0 || strings.Index(boot, `"/composer.js"`) < strings.Index(boot, `"/md.js"`) {
		t.Error("boot.js must load composer.js after md.js")
	}

	readers, fetches, streams, lines := 0, 0, 0, 0
	fetchPattern := regexp.MustCompile(`\bfetch\(`)
	for name, source := range sources {
		readers += strings.Count(source, ".getReader()")
		fetches += len(fetchPattern.FindAllString(source, -1))
		streams += strings.Count(source, "api.stream(")
		lines += strings.Count(source, "\n")
		if name != "boot.js" && fetchPattern.MatchString(source) {
			t.Errorf("%s calls fetch directly; the API client in boot.js owns transport", name)
		}
	}
	if readers > webuiStreamReaderCeiling {
		t.Errorf("stream reader sites = %d, ceiling %d", readers, webuiStreamReaderCeiling)
	}
	if fetches > webuiRawFetchCeiling {
		t.Errorf("raw fetch sites = %d, ceiling %d", fetches, webuiRawFetchCeiling)
	}
	if streams > webuiAPIStreamCeiling {
		t.Errorf("api.stream sites = %d, ceiling %d", streams, webuiAPIStreamCeiling)
	}
	if lines > webuiJavaScriptCeiling {
		t.Errorf("webui JavaScript lines = %d, ceiling %d", lines, webuiJavaScriptCeiling)
	}
	t.Logf("composer budget: readers=%d fetch=%d api.stream=%d lines=%d", readers, fetches, streams, lines)
}
