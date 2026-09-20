// Package storepolicy names which store documents are re-derivable caches:
// the one place retention, rebuild and release agree on what may lose its
// bytes. A cache is recomputable from git at its recorded source identity;
// everything else in the store is a claim, whatever reaches it.
package storepolicy

import (
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/codemanifest"
	"overgo/internal/codeprofile"
	"overgo/internal/runrecord"
)

// DerivedCacheSchemas are the document schemas the owner ruled re-derivable
// (2026-08-27): code manifests, code profiles and manifest analyses, which
// the gate recomputes from the tree at a source identity.
var DerivedCacheSchemas = []string{
	codemanifest.Schema,
	codeprofile.EvidenceSchema,
	automationcheck.ManifestAnalysisSchema,
}

// DerivedCache reports a descriptor whose bytes are a re-derivable cache.
func DerivedCache(descriptor artifact.Descriptor) bool {
	return slices.Contains(DerivedCacheSchemas, descriptor.Schema)
}

// FollowChildren reports a record whose dependents are discovered downward:
// the plan completion authority walks a gate preparation to its
// finalizations (children), a gate result to its attempts (children), and
// only then upward to runs, manifest analyses, recipes and profiles. A live
// set rooted at those records must follow their children or it drops the
// evidence every landed commit is proven by.
func FollowChildren(descriptor artifact.Descriptor) bool {
	return descriptor.Schema == runrecord.GateLifecycleSchema || descriptor.Schema == runrecord.GateSchema
}
