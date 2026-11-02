package server

import (
	"net/http/httptest"
	"testing"
)

// TestWebUIBrowserShowsActionErrors holds every action a person takes to saying what went wrong. A Cancel that failed
// left a disabled button and no word, a read of the workspace on a Chat click showed nothing, a page that was not JSON
// read as an empty answer, and a fault in the page's own code read as an unreachable server.
func TestWebUIBrowserShowsActionErrors(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the action errors leg uses Chromium through cmd/webui-lane")
	fixture := newSFTFixture(t)
	server := httptest.NewServer(newTestHandlerForRepository(t, fixture.store, responseRecipeGenerator(t, &fakeGenerator{}), func(config *Config) {
		config.Workflows, config.Evaluation = fixture.workspace, launchEvaluationWorkspace(t)
	}))
	defer server.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
	settle(`!!document.querySelector('.composer textarea')`)

	// What the server answers and what the page does with it.
	check(`(() => {
  const real = window.fetch;
  window.fetch = (url, init) => {
    if (String(url).endsWith('/stub/not-json')) return Promise.resolve(new Response('<html>gateway</html>', { status: 200 }));
    if (String(url).endsWith('/stub/empty')) return Promise.resolve(new Response('', { status: 200 }));
    if (String(url).endsWith('/stub/unanswered')) return Promise.reject(new TypeError('Failed to fetch'));
    return real(url, init);
  };
  window.settled = {};
  const note = (name, promise) => promise.then((value) => { settled[name] = { value }; }, (error) => { settled[name] = { error }; });
  note('notJSON', overgo.api.get('/stub/not-json'));
  note('empty', overgo.api.get('/stub/empty'));
  note('unanswered', overgo.api.get('/stub/unanswered'));
  return true;
})()`)
	settle(`!!settled.notJSON && !!settled.empty && !!settled.unanswered`)
	check(`!!settled.notJSON.error && settled.notJSON.error.message.includes('other than JSON')`)
	check(`!settled.empty.error && settled.empty.value === null`)
	check(`overgo.friendlyError(settled.unanswered.error).includes('could not be reached')`)
	check(`!overgo.friendlyError(new TypeError('x is not a function')).includes('could not be reached')`)

	// A Cancel the server refuses says why and can be pressed again, on a workflow and on an evaluation.
	check(`(() => {
  const get = overgo.api.get, post = overgo.api.post;
  const recipe = 'recipe:sha256:' + '0'.repeat(64), model = 'model:sha256:' + '0'.repeat(64);
  const suite = { plan: 'evidence:sha256:' + '1'.repeat(64), kind: 'exact', source: 'stub', cases: 1 };
  overgo.api.get = (path, options) => {
    if (path === '/training/capabilities') return Promise.resolve([{ task: 'training', recipe, name: 'stub', model, controls: [] }]);
    if (path === '/evaluations/capabilities') return Promise.resolve([{ model, suite }]);
    return get(path, options);
  };
  overgo.api.submit = () => Promise.resolve({ operation: 'evidence:stub' });
  overgo.api.post = (path, body, options) => {
    if (path === '/operations/cancel') return Promise.reject(new Error('Controlled cancellation failure'));
    if (path === '/evaluations/run') return Promise.resolve({ operation: 'evidence:stub' });
    return post(path, body, options);
  };
  overgo.waitOperation = () => new Promise(() => {});
  window.panelButton = (panel, label) => [...document.querySelectorAll(panel + ' button')].find((button) => button.textContent === label);
  location.hash = 'training-jobs';
  return true;
})()`)
	settle(`!!window.panelButton('#panel-training-jobs', 'Run')`)
	check(`(() => { panelButton('#panel-training-jobs', 'Run').click(); return true; })()`)
	settle(`!panelButton('#panel-training-jobs', 'Cancel').hidden`)
	check(`(() => { panelButton('#panel-training-jobs', 'Cancel').click(); return true; })()`)
	settle(`document.querySelector('#panel-training-jobs').textContent.includes('Controlled cancellation failure') && !panelButton('#panel-training-jobs', 'Cancel').disabled`)
	check(`(() => { location.hash = 'evaluations'; return true; })()`)
	settle(`!!document.querySelector('#panel-evaluations input[type="checkbox"]')`)
	check(`(() => { document.querySelector('#panel-evaluations input[type="checkbox"]').click(); panelButton('#panel-evaluations', 'Run').click(); return true; })()`)
	settle(`!panelButton('#panel-evaluations', 'Cancel').hidden`)
	check(`(() => { panelButton('#panel-evaluations', 'Cancel').click(); return true; })()`)
	settle(`document.querySelector('#panel-evaluations').textContent.includes('Controlled cancellation failure')`)

	// A Chat click whose workspace read fails opens Chat and says so.
	check(`(() => {
  const get = overgo.api.get;
  overgo.api.get = (path, options) => path === '/workspace/manifest' ? Promise.reject(new Error('Controlled manifest failure')) : get(path, options);
  location.hash = 'chat';
  return true;
})()`)
	settle(`document.querySelector('#panel-chat.active') && document.querySelector('#panel-chat').textContent.includes('Controlled manifest failure')`)
}
