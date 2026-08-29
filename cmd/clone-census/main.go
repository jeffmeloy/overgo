// clone-census prints the repository's exact-clone assessment from the
// code profile: every group of functions whose bodies share one AST
// fingerprint, largest first, with the duplicate-excess node total.
// The analysis is the same codeprofile machinery the gate's advisory
// lines cite; this command makes it invocable and verifiable on its
// own.
package main

import (
	"cmp"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/codeprofile"
	"overgo/internal/repoanalysis"
)

func main() {
	clioptions.MainNamed("clone-census", run)
}

func run() error {
	summary := flag.Bool("summary", false, "print only the totals line")
	runtimeOnly := flag.Bool("runtime", false, "print only non-test clone groups")
	requireAbsent := flag.String("require-absent", "", "fail if any clone group still contains a function whose file:name contains this substring")
	maxExcess := flag.Int("max-excess", -1, "fail if duplicate-excess nodes exceed this ratchet; -1 disables")
	updateBaseline := flag.Bool("update-baseline", false, "write the measured duplicate-excess to "+codeprofile.CloneBaselineFile)
	flag.Parse()
	snapshot, err := repoanalysis.DiscoverGo(".", "cmd", "internal")
	if err != nil {
		return err
	}
	profile, err := codeprofile.Build(snapshot)
	if err != nil {
		return err
	}
	fmt.Printf("functions=%d clone_groups=%d duplicate_excess_nodes=%d\n",
		len(profile.Functions), len(profile.Clones), profile.DuplicateExcessNodes)
	if *updateBaseline {
		baseline := codeprofile.CloneBaseline{
			Version:              1,
			Doc:                  "Reviewed ceiling on duplicate-excess AST nodes; the gate refuses commits above it. Lower with clone-census -update-baseline after tightening.",
			DuplicateExcessNodes: profile.DuplicateExcessNodes,
		}
		baselinePath := filepath.FromSlash(codeprofile.CloneBaselineFile)
		previous, present, err := codeprofile.LoadCloneBaseline(baselinePath)
		if err != nil {
			return err
		}
		if present {
			baseline.Doc = previous.Doc
		}
		encoded, err := json.MarshalIndent(baseline, "", "  ")
		if err != nil {
			return err
		}
		if err := os.WriteFile(baselinePath, append(encoded, '\n'), 0o644); err != nil {
			return err
		}
		fmt.Printf("clone-census: baseline written to %s at %d\n", codeprofile.CloneBaselineFile, profile.DuplicateExcessNodes)
	}
	if *maxExcess >= 0 {
		if profile.DuplicateExcessNodes > *maxExcess {
			return fmt.Errorf("clone-census: duplicate-excess %d exceeds ratchet %d", profile.DuplicateExcessNodes, *maxExcess)
		}
		fmt.Printf("clone-census: duplicate-excess %d within ratchet %d\n", profile.DuplicateExcessNodes, *maxExcess)
	}
	if *requireAbsent != "" {
		for _, clone := range profile.Clones {
			for _, function := range clone.Functions {
				if strings.Contains(function, *requireAbsent) {
					return fmt.Errorf("clone-census: %q still clones as %v", *requireAbsent, clone.Functions)
				}
			}
		}
		fmt.Printf("clone-census: %q is absent from every clone group\n", *requireAbsent)
	}
	if *summary {
		return nil
	}
	clones := profile.Clones
	sort.Slice(clones, func(i, j int) bool { return clones[i].Nodes > clones[j].Nodes })
	for _, clone := range clones {
		if *runtimeOnly && clone.AdvisoryClass == "test" {
			continue
		}
		class := clone.AdvisoryClass
		class = cmp.Or(class, "runtime")
		fmt.Printf("%5d nodes  %-8s  %v\n", clone.Nodes, class, clone.Functions)
	}
	return nil
}
