package oscillatorimage

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
	"time"

	"overgo/internal/processmeasure"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

func TestArtifactThreadCountAndShapeResources(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip + ": resource transitions use the real Un-0 artifact")
	}
	root := testutil.RepoRoot(t)
	wantImage, err := os.ReadFile(filepath.Join(root, "docs/media_samples/a8bc7f12cb9a537b40bf150493a002251bbbc59620a69944e19778dd463da6f3.png"))
	if err != nil {
		t.Fatal(err)
	}
	wantClip, err := os.ReadFile(filepath.Join(root, "docs/media_samples/bbae395b6153b0f2581e72ba32f9c4b77c120ec04d47c8a127efb78fa8ed7448.gif"))
	if err != nil {
		t.Fatal(err)
	}
	previousThreads := runtime.GOMAXPROCS(0)
	t.Cleanup(func() { runtime.GOMAXPROCS(previousThreads) })
	requests := []VideoRequest{
		{Class: 1, Seed: 11, Frames: 12, Scale: 2},
		{Class: 1, Seed: 202, Frames: 6, Scale: 8},
		{Class: 1, Seed: 11, Frames: 12, Scale: 2},
	}
	outputs := map[VideoRequest][]byte{}
	for _, threads := range slices.Compact([]int{1, runtime.NumCPU()}) {
		runtime.GOMAXPROCS(threads)
		model := loadArtifactModel(t)
		plan, err := model.prepare(t.Context(), Request{Class: 1, Seed: 11})
		if err != nil {
			t.Fatal(err)
		}
		features, err := model.integrate(t.Context(), plan)
		if err != nil {
			t.Fatal(err)
		}
		image, err := model.decode(t.Context(), features)
		if err != nil || !bytes.Equal(image.Data, wantImage) {
			t.Fatalf("thread count %d changed the frozen image: %v", threads, err)
		}
		for index, request := range requests {
			runtime.GC()
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			started := time.Now()
			plan, err := model.prepareVideo(t.Context(), request)
			if err != nil {
				t.Fatal(err)
			}
			features, err := model.integrateVideo(t.Context(), plan)
			if err != nil {
				t.Fatal(err)
			}
			clip, err := model.decodeVideo(t.Context(), features)
			if err != nil {
				t.Fatal(err)
			}
			wall := time.Since(started)
			runtime.ReadMemStats(&after)
			if clip.Frames != request.Frames || clip.Width != model.Cfg.OutW()*request.Scale || clip.Height != model.Cfg.OutH()*request.Scale {
				t.Fatal("resource profile changed the requested clip geometry")
			}
			if want, found := outputs[request]; found && !bytes.Equal(want, clip.Data) {
				t.Fatal("thread or shape transition changed repeated clip bytes")
			}
			outputs[request] = slices.Clone(clip.Data)
			if request == requests[0] && !bytes.Equal(clip.Data, wantClip) {
				t.Fatal("resource profile changed the frozen clip")
			}
			peak, err := processmeasure.SelfPeakWorkingSet()
			if err != nil {
				t.Fatal(err)
			}
			row, err := json.Marshal(map[string]any{
				"gomaxprocs": runtime.GOMAXPROCS(0), "logical_cpus": runtime.NumCPU(), "case_index": index, "request": request,
				"gif_sha256": fmt.Sprintf("%x", sha256.Sum256(clip.Data)), "wall_ns": wall.Nanoseconds(),
				"allocated_bytes": after.TotalAlloc - before.TotalAlloc, "allocation_calls": after.Mallocs - before.Mallocs,
				"live_heap_bytes": after.HeapAlloc, "process_lifetime_peak_host_bytes": peak,
				"host_to_device_bytes": 0, "device_to_host_bytes": 0, "device_allocated_bytes": 0,
				"scope": "Complete host-only clip generation on one loaded real model per thread profile. GOMAXPROCS is a Go scheduler limit, not affinity or a hard memory cap; host peak includes preceding requests and model loads.",
			})
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("MEDIA_CAPACITY %s", row)
		}
	}
	t.Log("host resource transitions: frozen image/clip bytes and repeated changed-shape clip bytes match with one and all detected logical CPU threads")
}
