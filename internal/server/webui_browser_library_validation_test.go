package server

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/libraryintake"
	"overgo/internal/modelswap"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

// laneHubServer serves one repository whose single file is a real GGUF on
// disk, the shape the workbench's discovery surface consumes.
const laneHubRepository = "lane/validated"

func laneHubServer(t *testing.T, file string) *httptest.Server {
	t.Helper()
	handle, err := os.Open(file)
	if err != nil {
		t.Fatal(err)
	}
	defer handle.Close()
	hash := sha256.New()
	size, err := io.Copy(hash, handle)
	if err != nil {
		t.Fatal(err)
	}
	name := filepath.Base(file)
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/models", func(response http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(response).Encode([]map[string]any{{"id": laneHubRepository, "downloads": 1}})
	})
	mux.HandleFunc("GET /api/models/lane/validated", func(response http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(response).Encode(map[string]any{
			"id": laneHubRepository, "sha": "rev0",
			"siblings": []map[string]any{{"rfilename": name, "lfs": map[string]any{"sha256": hex.EncodeToString(hash.Sum(nil)), "size": size}}},
		})
	})
	mux.HandleFunc("GET /lane/validated/resolve/rev0/{file}", func(response http.ResponseWriter, request *http.Request) {
		if request.PathValue("file") != name {
			http.NotFound(response, request)
			return
		}
		http.ServeFile(response, request, file)
	})
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	return server
}

// TestModelJourneyLibraryValidation: from a cold proxy over an empty store
// the Library downloads a real small GGUF from the hub, registers it,
// validates it as an operation the workbench behind the shell runs, and
// the picker then serves it; a model registered with its projector
// validates its text path and keeps the projector as its projection.
func TestModelJourneyLibraryValidation(t *testing.T) {
	if os.Getenv(webuilane.ModelJourneyEnvironment) != "1" {
		t.Skip(testskip.ShortIntegration + ": library validation runs through cmd/webui-lane -journeys")
	}
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	small := roots.ResolveModelPath("overgo-hfconvert/Qwen2.5-0.5B-f16.gguf")
	if _, err := os.Stat(small); err != nil {
		t.Fatalf("library validation unavailable: the small text model is absent: %v", err)
	}
	ctx := t.Context()
	projectedName, projectedLocation, projectorPath := smallestDeclaredProjector(t, ctx, roots.Store)
	if projectedLocation == "" {
		t.Fatal("library validation unavailable: no model with a declared projector has bytes on disk")
	}
	binary := filepath.Join(t.TempDir(), "overgo-server.exe")
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "go", Args: []string{"build", "-o", binary, "./cmd/server"}, Dir: root, Stdout: os.Stderr, Stderr: os.Stderr,
	})
	if err != nil || receipt.ExitCode != 0 {
		t.Fatalf("library validation unavailable: the server binary did not build: %v (exit %d)", err, receipt.ExitCode)
	}
	hub := laneHubServer(t, small)
	// The private store starts as the data root's copy: the registered
	// architecture profiles the Library registers against live there, and
	// nothing the journey writes reaches the data root.
	storeDir := filepath.Join(t.TempDir(), "store")
	if err := prepareBrowserJourneyStore(ctx, roots.Store, storeDir); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(storeDir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	intake := LibraryIntake{ModelFiles: libraryintake.ModelFiles, Register: libraryintake.Register, Validate: libraryintake.Validate}
	workbench, err := New(Config{
		RuntimePolicy: testRuntimePolicy(), Repository: store, LibraryIntake: intake,
		HubEndpoint: hub.URL, HubDownloadRoot: filepath.Join(t.TempDir(), "downloads"),
	}, GenerationRefused{Reason: idleRefusal})
	if err != nil {
		t.Fatal(err)
	}
	defer workbench.Close()
	if err := os.MkdirAll(workbench.config.HubDownloadRoot, 0o755); err != nil {
		t.Fatal(err)
	}
	supervisor, err := modelswap.New(modelswap.ServerLauncher{Binary: binary, Store: storeDir, Dir: root}, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer supervisor.Close()
	catalog, err := overgodb.OpenReadOnly(storeDir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = catalog.Close() })
	resolver := &modelswap.CatalogResolver{Store: catalog, Limit: 256}
	proxy := &modelswap.Proxy{
		Supervisor: supervisor, Resolver: resolver, Keys: resolver,
		Idle: &IdleShell{Catalog: resolver.Catalog, Repository: store, Intake: intake, Workbench: workbench},
	}
	front := httptest.NewServer(proxy)
	defer front.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	browser, err := webuilane.Open(ctx, path, front.URL+"/")
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	if err := browser.SetViewport(ctx, webuilane.ScreenViewports[0].Width, webuilane.ScreenViewports[0].Height); err != nil {
		t.Fatal(err)
	}
	// API idleness does not imply shell readiness: workspace scripts load
	// outside the API request counter. Use the shared browser predicate wait.
	settle := func(what, expression string) {
		t.Helper()
		if err := browser.Eventually(ctx, expression); err != nil {
			var page string
			_ = browser.Evaluate(ctx, `JSON.stringify({errors: window.overgo && window.overgo.errors, banners: [...document.querySelectorAll('#panel-library .err-banner')].map((node) => node.textContent), pending: window.overgo.api.pending(), rows: [...document.querySelectorAll('#panel-library tr')].map((row) => row.outerHTML.slice(0, 400))})`, &page)
			t.Fatalf("%s: %v; page: %s", what, err, page)
		}
	}
	// rowButton: the lifecycle control of the Library row that names the model.
	rowButton := func(rowText, label string) string {
		return `(() => {
  const rows = [...document.querySelectorAll("#panel-library tr")].filter((candidate) => candidate.textContent.includes(` + strconv.Quote(rowText) + `));
  const button = rows.flatMap((row) => [...row.querySelectorAll("button")]).find((candidate) => candidate.textContent === ` + strconv.Quote(label) + ` && !candidate.disabled);
  if (!button) return false;
  button.click();
  return true;
})()`
	}
	rowNote := func(rowText, text string) string {
		return `[...document.querySelectorAll("#panel-library tr")].some((row) => row.textContent.includes(` + strconv.Quote(rowText) + `) && row.textContent.includes(` + strconv.Quote(text) + `))`
	}
	// A validation binds the committed Go source; the gate's candidate and a
	// developer's tree are modified, so the intake refuses there with its
	// source rule after admitting the model and its projector, and only a
	// clean checkout runs the operation to completion. Both outcomes are the
	// intake's own; a projector refusal or a route refusal is neither.
	const sourceRule = "a verification requires committed Go source"
	validated := func(rowText, what string) bool {
		t.Helper()
		settle(what+" admitted or refused by the source rule",
			`(typeof window.laneValidation === 'string' && window.laneValidation.length > 0) || `+rowNote(rowText, sourceRule))
		var admitted string
		if err := browser.Evaluate(ctx, `window.laneValidation`, &admitted); err != nil {
			t.Fatal(err)
		}
		if admitted == "" {
			t.Logf("%s: the modified tree refused with the source rule after admitting the model", what)
			return false
		}
		id, err := artifact.ParseID(admitted)
		if err != nil {
			t.Fatal(err)
		}
		assertBrowserPredicate(t, ctx, browser, `(() => { window.laneValidation = ''; return true; })()`)
		waitBrowserOperationState(t, workbench, id, operation.StateCompleted)
		return true
	}

	// 0. The cold page: no child, the Library is the way in, and the workbench's validation answers are captured.
	settle("cold page", `document.querySelector("#model-pill").textContent === "Choose a model" && !!document.querySelector("#cold-start") && window.overgo.errors.length === 0`)
	assertBrowserPredicate(t, ctx, browser, `(() => {
  window.laneValidation = '';
  const post = overgo.api.post;
  overgo.api.post = async function (path, body, options) {
    const result = await post.call(this, path, body, options);
    if (path === '/library/validate') window.laneValidation = result.operation;
    return result;
  };
  [...document.querySelectorAll("#cold-start button")].find((button) => button.textContent === "Open the Library").click();
  return true;
})()`)
	settle("library on the cold page", `!!document.querySelector("#panel-library.active input[placeholder='search the Hugging Face hub']")`)

	// 1. The hub download: search, download and register against the store behind the cold proxy.
	assertBrowserPredicate(t, ctx, browser, `(() => {
  const query = document.querySelector("#panel-library input[placeholder='search the Hugging Face hub']");
  query.value = 'validated';
  [...document.querySelectorAll("#panel-library button")].find((button) => button.textContent === "search").click();
  return true;
})()`)
	settle("hub search lists the lane repository", `[...document.querySelectorAll("#panel-library button")].some((button) => button.textContent === "download")`)
	assertBrowserPredicate(t, ctx, browser, `(() => { [...document.querySelectorAll("#panel-library button")].find((button) => button.textContent === "download").click(); return true; })()`)
	settle("download succeeded", rowNote(laneHubRepository, "done"))
	settle("register control of the row", rowButton(laneHubRepository, "register"))
	settle("downloaded model registered", rowNote(laneHubRepository, "registered"))
	t.Log("download leg: the hub download registered through the cold proxy's workbench")

	// 2. A model registered with its projector from disk keeps the projector as its projection candidate.
	assertBrowserPredicate(t, ctx, browser, `(() => {
  document.querySelector("#panel-library input[placeholder='model GGUF or directory on disk']").value = `+strconv.Quote(projectedLocation)+`;
  document.querySelector("#panel-library input[placeholder='projector GGUF (optional)']").value = `+strconv.Quote(projectorPath)+`;
  [...document.querySelectorAll("#panel-library button")].find((button) => button.textContent === "register a local model").click();
  return true;
})()`)
	settle("register control of the row", rowButton(projectedLocation, "register"))
	settle("local model registered with its projector", rowNote(projectedLocation, "with projector"))

	// 3. Both validate as operations the workbench behind the shell runs; the projector pair validates its text path.
	settle("validate control of the row", rowButton(projectedLocation, "validate"))
	projectorValidated := validated(projectedLocation, "projector validation")
	if projectorValidated {
		assertBrowserPredicate(t, ctx, browser, rowNote(projectedLocation, "projector"))
	}
	t.Logf("projector leg: %s admitted its text validation with projector %s registered (completed=%v)", projectedName, filepath.Base(projectorPath), projectorValidated)
	settle("validate control of the row", rowButton(laneHubRepository, "validate"))
	downloadValidated := validated(laneHubRepository, "downloaded model validation")
	t.Logf("validation leg: the downloaded model's validation admitted through the cold proxy's workbench (completed=%v)", downloadValidated)

	// 4. A model the store activates serves from the picker: the validated
	// download on a clean checkout, otherwise the projector pair the copied
	// store already activates; either way the first child launches from the
	// Library's page.
	name := filepath.Base(small)
	if !downloadValidated {
		name = projectedName
	}
	assertBrowserPredicate(t, ctx, browser, `(() => { history.replaceState(null, "", location.pathname); document.querySelector("#model-pill").click(); return true; })()`)
	settle("picker lists the validated model", `[...document.querySelectorAll(".topbar .card .row .mono")].some((node) => node.textContent === `+strconv.Quote(name)+`)`)
	assertBrowserPredicate(t, ctx, browser, `(() => {
  const row = [...document.querySelectorAll(".topbar .card .row")].find((node) => node.querySelector(".mono") && node.querySelector(".mono").textContent === `+strconv.Quote(name)+`);
  if (!row) return false;
  [...row.querySelectorAll("button")].find((button) => button.textContent === "serve").click();
  return true;
})()`)
	settle("the chosen model serves", `document.querySelector("#model-pill").textContent === `+strconv.Quote(name)+` && !!document.querySelector("#panel-chat.active .composer textarea") && !window.overgo.modelSwitching()`)
	t.Logf("library validation leg: downloaded, registered with a projector, validation admitted and %s served from the cold proxy", name)
}
