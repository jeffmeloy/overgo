package webuilane

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"overgo/internal/processcontrol"
	"overgo/internal/testevidence"
	"path/filepath"
	"testing"
)

func TestWebUIBrowserDownloadIsolation(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testevidence.ShortIntegrationSkip + ": download isolation runs through cmd/webui-lane")
	}
	executable, err := FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	const payload = "isolated browser download"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/download" {
			w.Header().Set("Content-Disposition", `attachment; filename="fixture.txt"`)
			w.Write([]byte(payload))
			return
		}
		w.Header().Set("Content-Type", "text/html")
		w.Write([]byte(`<a id="download" href="/download" download="fixture.txt">Download</a>`))
	}))
	defer server.Close()
	browser, err := Open(t.Context(), executable, server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	if err := browser.Eventually(t.Context(), `!!document.querySelector('#download')`); err != nil {
		t.Fatal(err)
	}
	if err := browser.Evaluate(t.Context(), `(async()=>{const b=await (await fetch("/download")).blob();const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="fixture.txt";a.click()})()`, nil); err != nil {
		t.Fatal(err)
	}
	destination := filepath.Join(browser.profile, "downloads", "fixture.txt")
	if err := processcontrol.AwaitResource(t.Context(), func() error {
		data, err := os.ReadFile(destination)
		if errors.Is(err, os.ErrNotExist) {
			return processcontrol.ErrResourceBusy
		}
		if err != nil {
			return err
		}
		if string(data) != payload {
			return errors.New("download bytes differ")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	profile := browser.profile
	if err := browser.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(profile); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("download profile survives close: %v", err)
	}
}
