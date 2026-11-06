//go:build modeltest

package main

import (
	"bytes"
	"encoding/json"
	"image/gif"
	"os"
	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"path/filepath"
	"slices"
	"testing"
)

func TestImageVideoWanGIFRejectsAlteredSequence(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	var bundle mediaWanBundle
	testcheck.NoError(t, jsonfile.DecodeStrict(filepath.Join(root, "docs/image_video_wan.json"), &bundle))
	if len(bundle.Cases) == 0 {
		t.Fatal("Wan bundle has no cases")
	}
	quality, err := os.ReadFile(filepath.Join(root, "docs/image_video_protocol.json"))
	testcheck.NoError(t, err)
	testcheck.NoError(t, checkImageVideoProtocolIdentity(quality))
	var protocol imageVideoProtocol
	testcheck.NoError(t, json.Unmarshal(quality, &protocol))
	index := slices.IndexFunc(protocol.Cases, func(c imageVideoCase) bool { return c.ID == bundle.Cases[0].Case })
	if index < 0 || len(protocol.Cases[index].Observations) != 1 {
		t.Fatal("Wan case lacks its protocol observation")
	}
	observation := protocol.Cases[index].Observations[0]
	roots, err := dataroot.Resolve(root)
	testcheck.NoError(t, err)
	store := overgodbtest.OpenReadOnly(t, retainedReferenceStore(roots.Store))
	content, found, err := artifact.ReadContent(t.Context(), store, bundle.Cases[0].Output)
	if err != nil || !found {
		t.Fatalf("Wan output %s: found=%t err=%v", bundle.Cases[0].Output, found, err)
	}
	if err := checkMediaWanGIF(content.Data, content.Data, observation); err != nil {
		t.Fatalf("unmodified control: %v", err)
	}
	for _, mutation := range []string{"truncate", "reorder", "timing", "pixel"} {
		t.Run(mutation, func(t *testing.T) {
			video, err := gif.DecodeAll(bytes.NewReader(content.Data))
			testcheck.NoError(t, err)
			switch mutation {
			case "truncate":
				video.Image = video.Image[:len(video.Image)-1]
				video.Delay = video.Delay[:len(video.Delay)-1]
				if len(video.Disposal) > 0 {
					video.Disposal = video.Disposal[:len(video.Disposal)-1]
				}
			case "reorder":
				slices.Reverse(video.Image)
			case "timing":
				video.Delay[0] += video.Delay[0]
			case "pixel":
				video.Image[0].Pix[0] = (video.Image[0].Pix[0] + 1) % uint8(len(video.Image[0].Palette)-1)
			}
			var encoded bytes.Buffer
			testcheck.NoError(t, gif.EncodeAll(&encoded, video))
			if err := checkMediaWanGIF(encoded.Bytes(), content.Data, observation); err == nil {
				t.Fatal("altered clip accepted")
			}
		})
	}
}
