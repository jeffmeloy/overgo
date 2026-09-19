package sensenovarecipe

import (
	"strings"
	"testing"
	"time"

	"overgo/internal/routedlm"
	"overgo/internal/runrecord"
	"overgo/internal/worklease"
)

const capacityGiB = 1 << 30

// A representative model residency and per-token flow activation cost, measured
// elsewhere by the device runtime and retained; the capacity decision consumes
// them, it never re-executes the model.
const (
	modelResidencyBytes = 20 * capacityGiB
	perTokenDeviceBytes = 3 * 1024 * 1024
	perTokenHostBytes   = 256 * 1024
	baseHostBytes       = 4 * capacityGiB
)

// measuredPhases returns the retained per-phase serving measurements for an
// image request of the declared geometry: prepare loads the residency, the flow
// body holds the peak (residency plus a per-token activation working set), and
// decode reads results back. The numbers scale with the image token count, so
// geometry drives the retained demand.
func measuredPhases(plan routedlm.FlowImagePlan) []runrecord.ServingResources {
	activation := uint64(plan.Tokens) * perTokenDeviceBytes
	return []runrecord.ServingResources{
		{PeakDeviceBytes: modelResidencyBytes, PeakHostBytes: baseHostBytes, HostToDeviceBytes: modelResidencyBytes},
		{PeakDeviceBytes: modelResidencyBytes + activation, PeakHostBytes: baseHostBytes + uint64(plan.Tokens)*perTokenHostBytes, HostToDeviceBytes: activation, DeviceToHostBytes: activation / 4},
		{PeakDeviceBytes: modelResidencyBytes + activation/2, PeakHostBytes: baseHostBytes, DeviceToHostBytes: uint64(plan.Tokens) * perTokenHostBytes},
	}
}

func ceilGiB(bytes uint64) int { return int((bytes + capacityGiB - 1) / capacityGiB) }

// mediaRequestDemand derives a whole-request work-lease reservation from the
// declared geometry and the retained per-phase measurements. VRAM and host RAM
// are sized from the PEAK high-water mark (reduced by max), never from
// cumulative allocation traffic (reduced by sum); the request holds the GPU
// exclusively for the generation.
func mediaRequestDemand(plan routedlm.FlowImagePlan, phases []runrecord.ServingResources) worklease.Resources {
	var peakDevice, peakHost uint64
	for _, phase := range phases {
		peakDevice = max(peakDevice, phase.PeakDeviceBytes)
		peakHost = max(peakHost, phase.PeakHostBytes)
	}
	return worklease.Resources{
		CPUThreads:   4,
		HostRAMGiB:   ceilGiB(peakHost),
		VRAMGiB:      ceilGiB(peakDevice),
		GPUExclusive: true,
	}
}

// requestTraffic is the cumulative host/device transfer of a request, reduced by
// sum: it is the allocation traffic the row keeps distinct from retained peak.
func requestTraffic(phases []runrecord.ServingResources) uint64 {
	var traffic uint64
	for _, phase := range phases {
		traffic += phase.HostToDeviceBytes + phase.DeviceToHostBytes
	}
	return traffic
}

// TestMediaRequestCapacity chooses media execution from measured request
// capacity: it derives a whole-request RAM/VRAM demand from declared image
// geometry and retained phase measurements, then renders admission decisions
// through the pure worklease owner, with no model execution and no device.
func TestMediaRequestCapacity(t *testing.T) {
	t.Parallel()
	flow := routedlm.FlowPlan{VisionPatch: 16, ImageMerge: 2, NoiseScaleBase: 64, NoiseScale: 1, NoiseScaleMax: 4, NoiseScaleMode: "resolution"}
	plan, err := flow.ImagePlan(1024, 1024)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Tokens != 1024 {
		t.Fatalf("declared geometry tokens=%d", plan.Tokens)
	}
	phases := measuredPhases(plan)
	demand := mediaRequestDemand(plan, phases)
	now := time.Date(2026, 9, 19, 0, 0, 0, 0, time.UTC)
	request := worklease.Lease{Task: "media/generate", Worker: "media-lane", Worktree: "C:/repo/media", Resources: demand}

	// The demand is sized from the retained peak, not the cumulative traffic.
	if peakGiB := ceilGiB(modelResidencyBytes + uint64(plan.Tokens)*perTokenDeviceBytes); demand.VRAMGiB != peakGiB {
		t.Fatalf("VRAM demand %d GiB not sized from the peak %d GiB", demand.VRAMGiB, peakGiB)
	}
	if traffic := requestTraffic(phases); traffic <= uint64(demand.VRAMGiB)*capacityGiB {
		t.Fatal("fixture must exercise traffic that exceeds the peak reservation")
	}

	t.Run("sufficient capacity fits", func(t *testing.T) {
		capacity := worklease.Resources{CPUThreads: 16, HostRAMGiB: demand.HostRAMGiB + 8, VRAMGiB: demand.VRAMGiB + 8}
		advisory := worklease.AssessResources(now, capacity, []worklease.Lease{request})
		if !advisory.Fits || advisory.Reserved != demand || len(advisory.Conflicts) != 0 {
			t.Fatalf("advisory = %+v", advisory)
		}
	})

	t.Run("insufficient VRAM refuses", func(t *testing.T) {
		capacity := worklease.Resources{CPUThreads: 16, HostRAMGiB: demand.HostRAMGiB + 8, VRAMGiB: demand.VRAMGiB - 1}
		advisory := worklease.AssessResources(now, capacity, []worklease.Lease{request})
		if advisory.Fits {
			t.Fatal("insufficient VRAM fit")
		}
		if !hasConflict(advisory.Conflicts, "VRAM GiB reserved") {
			t.Fatalf("conflicts = %v", advisory.Conflicts)
		}
	})

	t.Run("second GPU request refuses", func(t *testing.T) {
		peer := worklease.Lease{Task: "media/peer", Worker: "peer-lane", Worktree: "C:/repo/peer", Resources: demand}
		capacity := worklease.Resources{CPUThreads: 16, HostRAMGiB: 4 * demand.HostRAMGiB, VRAMGiB: 4 * demand.VRAMGiB}
		advisory := worklease.AssessResources(now, capacity, []worklease.Lease{request, peer})
		if advisory.Fits {
			t.Fatal("two exclusive GPU requests fit")
		}
		if !hasConflict(advisory.Conflicts, "both reserve the exclusive GPU") {
			t.Fatalf("conflicts = %v", advisory.Conflicts)
		}
	})

	t.Run("release restores capacity", func(t *testing.T) {
		capacity := worklease.Resources{CPUThreads: 16, HostRAMGiB: demand.HostRAMGiB + 8, VRAMGiB: demand.VRAMGiB + 8}
		advisory := worklease.AssessResources(now, capacity, nil)
		if !advisory.Fits || advisory.Reserved != (worklease.Resources{}) || len(advisory.ActiveTasks) != 0 {
			t.Fatalf("released advisory = %+v", advisory)
		}
	})

	t.Run("exact reuse derives an identical demand", func(t *testing.T) {
		if again := mediaRequestDemand(plan, measuredPhases(plan)); again != demand {
			t.Fatalf("identical geometry derived a different demand: %+v != %+v", again, demand)
		}
	})

	t.Run("larger geometry demands more", func(t *testing.T) {
		bigger, err := flow.ImagePlan(2048, 2048)
		if err != nil {
			t.Fatal(err)
		}
		if bigger.Tokens <= plan.Tokens {
			t.Fatalf("larger image tokens=%d", bigger.Tokens)
		}
		if grown := mediaRequestDemand(bigger, measuredPhases(bigger)); grown.VRAMGiB <= demand.VRAMGiB {
			t.Fatalf("larger geometry did not grow the VRAM demand: %d <= %d", grown.VRAMGiB, demand.VRAMGiB)
		}
	})
}

func hasConflict(conflicts []string, want string) bool {
	for _, conflict := range conflicts {
		if strings.Contains(conflict, want) {
			return true
		}
	}
	return false
}
