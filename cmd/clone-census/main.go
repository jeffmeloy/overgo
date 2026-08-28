// clone-census prints the repository's exact-clone assessment from the
// code profile: every group of functions whose bodies share one AST
// fingerprint, largest first, with the duplicate-excess node total.
// The analysis is the same codeprofile machinery the gate's advisory
// lines cite; this command makes it invocable and verifiable on its
// own.
package main

import (
	"flag"
	"fmt"
	"sort"

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
		if class == "" {
			class = "runtime"
		}
		fmt.Printf("%5d nodes  %-8s  %v\n", clone.Nodes, class, clone.Functions)
	}
	return nil
}
