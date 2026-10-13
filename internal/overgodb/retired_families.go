package overgodb

import (
	"slices"

	"overgo/internal/artifact"
)

// A retired family is a record family that nothing in the tree writes and
// nothing reads. An open decodes its records all the same, so a rebuild asked
// to leave the retired families out carries none of them, and the census
// marks them in a store that holds them. The table is the one statement of
// which families those are: a family is retired by an entry here, with the
// commit that deleted its writer, and by nothing a command is told.

// The census keys a family by the projection the key belongs to. The schema
// of an artifact and the leading part of its commit's key name the family of
// the artifact and of its content, its locations and the lineage edges it
// declares. The leading part of a name is the family of an alias, and the
// leading part of a batch key the family of a commit.
const (
	artifactKeys = "artifacts"
	aliasNames   = "aliases"
	commitKeys   = "commits"
)

// The commits that deleted the writers of the retired families: the old gate,
// the closure scanner with its ledger, and the code manifest.
const (
	oldGateDeleted       = "d80326076"
	closureLedgerDeleted = "be70f7c1c"
	codeManifestDeleted  = "c69524039"
)

// retiredFamily is one retired family: the key that names it, written exactly
// as the census writes it, the commit that deleted its writer, and what its
// records were.
type retiredFamily struct {
	keyed, name, deleted, why string
}

// retiredFamilies are the families a rebuild leaves out when asked. Each is
// one the census of the installed store classes as the development process's
// with no writer in the tree (docs/verification/checkpoint-families.txt). A
// family that a test, a report or a command reads is not here, whatever wrote
// it.
var retiredFamilies = []retiredFamily{
	{artifactKeys, "overgo/agent-obligation/v1 by gate/package", oldGateDeleted, "the old gate's obligation for one requirement of a package it checked"},
	{artifactKeys, "overgo/gate-package-receipt/v1 by gate/package", oldGateDeleted, "the old gate's receipt for one package of a commit's checks"},
	{artifactKeys, "no schema: evidence bare by gate/package", oldGateDeleted, "an identity a package receipt's batch declared as an obligation's source and did not carry"},
	{artifactKeys, "overgo/gate-package-obligations/v1 by gate/package-obligations", oldGateDeleted, "the old gate's obligations document for one preparation of a package"},
	{artifactKeys, "no schema: evidence bare by gate/package-obligations", oldGateDeleted, "an identity an obligations document's batch declared and did not carry"},
	{aliasNames, "gate/package", oldGateDeleted, "the alias to a package's receipt"},
	{commitKeys, "gate/package", oldGateDeleted, "the batch that carried a package's receipt"},
	{artifactKeys, "overgo/closure-ledger/v3 by closure-scan", closureLedgerDeleted, "the closure scanner's ledger for one file it bound"},
	{artifactKeys, "overgo/closure-ledger/v3 by closure-scan/import", closureLedgerDeleted, "a closure ledger the scanner imported"},
	{artifactKeys, "overgo/closure-ledger/v3 by closure-scan/publish", closureLedgerDeleted, "a closure ledger the scanner published"},
	{artifactKeys, "overgo/closure-ledger/v3 by closure-scan/rebind", closureLedgerDeleted, "a closure ledger the scanner bound again"},
	{artifactKeys, "overgo/closure-ledger/v3 by rebuild", closureLedgerDeleted, "a closure ledger a store rebuild committed again"},
	{aliasNames, "closure/active", closureLedgerDeleted, "the alias to a file's current closure ledger"},
	{artifactKeys, "overgo.code-manifest-digest.v1 by code-manifest-digest/profile", codeManifestDeleted, "a code manifest's digest for one run of the old gate"},
	{artifactKeys, "no schema: profile bare by code-manifest-digest/profile", codeManifestDeleted, "a profile identity a digest's batch declared and did not carry"},
	{aliasNames, "code-manifest/digest", codeManifestDeleted, "the alias to a profile's code manifest digest"},
	{commitKeys, "code-manifest-digest/profile", codeManifestDeleted, "the batch that recorded a code manifest's digest"},
}

// retiredAt is the place in the table of the family that name is under one
// of the three keys, and less than zero for a family the table does not hold.
func retiredAt(keyed, name string) int {
	return slices.IndexFunc(retiredFamilies, func(family retiredFamily) bool {
		return family.keyed == keyed && family.name == name
	})
}

// keyedLike gives the key that names the families of a checkpoint member: a
// content locator, a lineage edge and a location carry the family of the
// artifact they belong to, and every other member is keyed as itself.
func keyedLike(member string) string {
	switch member {
	case "contents", "lineage", "locations":
		return artifactKeys
	}
	return member
}

// retiredCount is what a rebuild did with one retired family. leftOut and
// kept count the records the family's key names: artifacts, aliases or
// commits. For a family of artifacts the rest count what went with the ones
// left out: their content locators and the bytes of the content the source
// still held, the lineage edges they declare, and their locations.
type retiredCount struct {
	retiredFamily
	leftOut, kept              int
	contents, edges, locations int
	bytes                      uint64
}

// omission is what a rebuild leaves out of its source: the artifacts of the
// retired families that no kept record names, the aliases named for a retired
// family, and the count of every retired family in the table's order. The
// zero omission leaves nothing out.
type omission struct {
	artifacts map[artifact.ID]bool
	aliases   map[string]bool
	counts    []retiredCount
}

// leaveOutRetired decides, against the source's state, what a rebuild
// without the retired families leaves out. An artifact of a retired family
// stays when a kept record names it: as the lineage parent of a kept
// artifact, which a manifest's component is, as the target of a kept alias,
// or in a causal link, none of which is of a retired family. What it names in
// turn stays as well, so the kept records are closed under what they name. A
// rebuild carries no commit of its source, so every commit of a retired
// family is left out.
func leaveOutRetired(state *catalogState) omission {
	leave := omission{artifacts: map[artifact.ID]bool{}, aliases: map[string]bool{}, counts: make([]retiredCount, len(retiredFamilies))}
	for at, family := range retiredFamilies {
		leave.counts[at].retiredFamily = family
	}
	for _, commit := range state.commits.all() {
		if at := retiredAt(commitKeys, leadingPart(commit.key)); at >= 0 {
			leave.counts[at].leftOut++
		}
	}
	// unnamed holds the artifacts of retired families no kept record has
	// named yet, each with its family's place; kept holds the artifacts whose
	// own records are still to be read for what they name.
	family := artifactFamily(state)
	unnamed := map[artifact.ID]int{}
	var kept []artifact.ID
	for id := range state.artifacts.records {
		name, _ := family(id)
		if at := retiredAt(artifactKeys, name); at >= 0 {
			unnamed[id] = at
			continue
		}
		kept = append(kept, id)
	}
	keep := func(id artifact.ID) {
		if at, held := unnamed[id]; held {
			delete(unnamed, id)
			leave.counts[at].kept++
			kept = append(kept, id)
		}
	}
	state.aliases.each(func(name string, target artifact.ID) {
		if at := retiredAt(aliasNames, leadingPart(name)); at >= 0 {
			leave.counts[at].leftOut++
			leave.aliases[name] = true
			return
		}
		keep(target)
	})
	for _, link := range state.causality.records {
		keep(link.Execution)
		for _, evidence := range causalEvidence(link) {
			keep(evidence)
		}
	}
	for len(kept) > 0 {
		id := kept[len(kept)-1]
		kept = kept[:len(kept)-1]
		for _, edge := range state.lineage.parentsOf(id) {
			keep(edge.parent)
		}
	}
	for id, at := range unnamed {
		count := &leave.counts[at]
		leave.artifacts[id] = true
		count.leftOut++
		count.edges += len(state.lineage.parentsOf(id))
		count.locations += len(state.locations.of(id))
		if locator, held := state.contents.locator(id); held {
			count.contents++
			if locator.released == 0 {
				count.bytes += state.artifacts.records[id].descriptor.Size
			}
		}
	}
	return leave
}
