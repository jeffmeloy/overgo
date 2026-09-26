package repoanalysis

import (
	"fmt"
	"reflect"
	"runtime"
	"testing"
	"unsafe"
)

// TestCensusSplitCopiesEachSiteOnce holds the census split and join to one
// copy of each site: the bytes a round trip allocates grow with the sites,
// not with packages times sites, and the round trip returns the census it
// was given without writing through the header's slices.
// Serial: it measures the process's allocated bytes.
func TestCensusSplitCopiesEachSiteOnce(t *testing.T) {
	const packages, findings, perPackage = 200, 8, 4
	census := ModernGoCensus{TargetGo: "1.26", SourceIdentity: "fixture"}
	for finding := range findings {
		entry := ModernGoFinding{ID: fmt.Sprintf("finding_%02d", finding), Measured: true}
		for pkg := range packages {
			for line := range perPackage {
				site := ModernGoSite{Path: fmt.Sprintf("internal/p%03d/f.go", pkg), Line: line + 1, Package: fmt.Sprintf("overgo/internal/p%03d", pkg), Symbol: "S", Kind: "k"}
				if line%2 == 0 {
					entry.Candidates = append(entry.Candidates, site)
				} else {
					entry.Adopted = append(entry.Adopted, site)
				}
			}
		}
		census.Findings = append(census.Findings, entry)
	}
	header, parts := census.SplitByPackage()
	if len(parts) != packages || len(header.Findings) != findings || header.Findings[0].Candidates != nil {
		t.Fatalf("split = %d parts, %d header findings", len(parts), len(header.Findings))
	}
	if joined := JoinModernGoCensus(header, parts); !reflect.DeepEqual(joined, census) {
		t.Fatal("the round trip changed the census")
	}
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	header, parts = census.SplitByPackage()
	_ = JoinModernGoCensus(header, parts)
	runtime.ReadMemStats(&after)
	sites := packages * findings * perPackage
	if allocated, bound := after.TotalAlloc-before.TotalAlloc, uint64(8*sites)*uint64(unsafe.Sizeof(ModernGoSite{})); allocated > bound {
		t.Fatalf("a round trip over %d sites allocated %d bytes, over the linear bound %d", sites, allocated, bound)
	}
}
