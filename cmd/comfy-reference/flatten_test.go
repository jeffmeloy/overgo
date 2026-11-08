package main

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestTheUltraWorkflowFlattensToItsAPIRequest holds the recorder's reading of MOSS Ultra's saved workflow to ComfyUI's:
// with each section switched on, the request built from the kept workflow names the same nodes, of the same classes,
// with the same inputs, as the request ComfyUI's own page built; and a link to a node the graph lacks is refused.
func TestTheUltraWorkflowFlattensToItsAPIRequest(t *testing.T) {
	ultra := testcheck.Must(readFixture(filepath.Join("..", "..", ultraFixture)))(t)
	if len(ultra.sections) == 0 {
		t.Fatal("the fixture holds no sections")
	}
	for _, title := range ultra.titles() {
		t.Run(slug(title), func(t *testing.T) {
			built := testcheck.Must(ultra.request(title))(t)
			var comfy map[string]apiNode
			testcheck.NoError(t, readJSON(filepath.Join(ultra.folder, filepath.FromSlash(ultra.requests[title])), &comfy))
			for _, difference := range differences(t, built, comfy) {
				t.Error(difference)
			}
			t.Logf("%s: %d nodes, as ComfyUI built it", title, len(comfy))
		})
	}
	t.Run("an unresolvable link is refused", func(t *testing.T) {
		var broken uiWorkflow
		testcheck.NoError(t, json.Unmarshal([]byte(`{"nodes":[{"id":1,"type":"VAEDecode","mode":0,"inputs":[{"name":"samples","type":"LATENT","link":1},{"name":"vae","type":"VAE","link":null}]}],"links":[[1,7,0,1,0,"LATENT"]]}`), &broken))
		if _, held := ultra.info["VAEDecode"]; !held {
			t.Fatal("the fixture's node definitions lack VAEDecode")
		}
		flattener := testcheck.Must(newFlattener(broken, ultra.info))(t)
		if _, err := flattener.request(); err == nil || !strings.Contains(err.Error(), "node 7") {
			t.Fatalf("a link from node 7, which the graph lacks, gave %v", err)
		}
	})
}

// differences are the ways built differs from ComfyUI's request: a node one has and the other lacks, or a class or input
// that differs, each compared as JSON reads it.
func differences(t *testing.T, built, comfy map[string]apiNode) []string {
	t.Helper()
	normal := func(request map[string]apiNode) map[string]apiNode {
		var read map[string]apiNode
		testcheck.NoError(t, json.Unmarshal(testcheck.Must(json.Marshal(request))(t), &read))
		return read
	}
	built, comfy = normal(built), normal(comfy)
	var ids []string
	for id := range comfy {
		ids = append(ids, id)
	}
	for id := range built {
		if _, held := comfy[id]; !held {
			ids = append(ids, id)
		}
	}
	slices.Sort(ids)
	var found []string
	for _, id := range ids {
		mine, held := built[id]
		theirs, theirsHeld := comfy[id]
		switch {
		case !held:
			found = append(found, "node "+id+" ("+theirs.ClassType+") is missing")
		case !theirsHeld:
			found = append(found, "node "+id+" ("+mine.ClassType+") is not in ComfyUI's request")
		case mine.ClassType != theirs.ClassType:
			found = append(found, "node "+id+" is "+mine.ClassType+", not "+theirs.ClassType)
		case !reflect.DeepEqual(mine.Inputs, theirs.Inputs):
			mineJSON, _ := json.Marshal(mine.Inputs)
			theirsJSON, _ := json.Marshal(theirs.Inputs)
			found = append(found, "node "+id+" ("+mine.ClassType+") has inputs "+string(mineJSON)+", not "+string(theirsJSON))
		}
	}
	return found
}
