//go:build windows

package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserListsShowNamesAndThumbnails holds moss-page/names (the owner's review of 2026-11-24, point 7). After
// a MOSS run, Runs leads each run with its model's name, the name the Generate page offers it by, its recipe's identity
// kept as the name's detail. Artifacts is a grid of cards, each led by what the item is ("PNG picture"), the run's
// picture a thumbnail that shows it whole in its card when pressed, the identity below. Runs had led with the recipe's identity, and Artifacts with
// identities and links to download. The Library and the model picker already lead with names. The store records no
// time a run ended, so no list shows a date.
func TestWebUIBrowserListsShowNamesAndThumbnails(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the lists-named leg uses Chromium through cmd/webui-lane")
	fixture := newMOSSGenerationFixture(t, nil)
	registered := serveTestRequest(fixture.handler, http.MethodPost, "/library/register", `{"kind":"model","path":`+string(testcheck.Must(json.Marshal(fixture.root))(t))+`}`)
	if registered.Code != http.StatusOK {
		t.Fatalf("register: %d %s", registered.Code, registered.Body)
	}
	server := httptest.NewServer(fixture.handler)
	defer server.Close()
	defer fixture.handler.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL+"/#generation")
	defer browser.Close()
	settle(`!!document.querySelector('#panel-generation.active .workflow-task + .control-grid > label.control')`)
	check(`(() => {
  const panel=document.querySelector('#panel-generation');
  // The model's name as the Generate page offers it: "image-gen / <name>".
  const offered=panel.querySelector('select[aria-label="capability"]');
  window.modelName=offered.options[offered.selectedIndex].text.split(' / ').slice(1).join(' / ').split(' (')[0].split(' · ')[0].trim();
  const prompt=[...panel.querySelectorAll('label.control')].find(label=>label.querySelector('span').textContent==='prompt').querySelector('textarea');
  prompt.value='a lighthouse at dusk'; prompt.dispatchEvent(new Event('input',{bubbles:true}));
  [...panel.querySelectorAll('button')].find(button=>button.textContent==='Run').click();
  return true;
})()`)
	settle(`!!document.querySelector('#panel-generation .workbench-output')`)
	check(`(() => { location.hash='#runs'; return true; })()`)
	settle(`!!document.querySelector('#panel-runs.active table.grid tr td')`)
	check(`(() => {
  const table=document.querySelector('#panel-runs table.grid');
  const heading=table.querySelector('th').textContent.trim().toLowerCase();
  const first=table.querySelector('tr td').textContent.trim();
  if (!window.modelName || heading!=='model' || first!==window.modelName) throw new Error('the run list leads with '+JSON.stringify([heading, first])+'; want its model\'s name '+JSON.stringify(window.modelName));
  location.hash='#artifacts';
  return true;
})()`)
	settle(`!!document.querySelector('#panel-artifacts.active .artifact-gallery article')`)
	check(`(() => {
  const cards=[...document.querySelectorAll('#panel-artifacts .artifact-gallery article')];
  const picture=cards.find(card=>card.querySelector('img'));
  if (!picture) throw new Error('the gallery holds no picture');
  const thumb=picture.querySelector('img');
  if (!thumb.classList.contains('artifact-thumb') || !thumb.closest('button') || thumb.alt!=='PNG picture') throw new Error('the picture is no thumbnail that opens it: '+picture.innerHTML.slice(0,200));
  thumb.closest('button').click();
  if (!picture.classList.contains('whole')) throw new Error('the pressed thumbnail does not show the picture whole');
  const led=cards.map(card=>[...card.children].find(child=>child.tagName!=='A' && child.textContent.trim())?.textContent.trim() || '');
  const byIdentity=led.filter(text=>/^[a-z]+:[0-9a-f]{6,}/.test(text));
  if (byIdentity.length) throw new Error(byIdentity.length+' cards lead with an identity: '+byIdentity[0]);
  if (getComputedStyle(document.querySelector('#panel-artifacts .artifact-gallery')).display!=='grid') throw new Error('the gallery is no grid');
  return window.overgo.errors.length===0;
})()`)
	webuilane.Leg(t, "lists-named", "After a MOSS run, Runs leads each run with its model's name and Artifacts is a grid of cards led by what each item is, a picture a thumbnail that opens it.")
}
