// Package acceptancelane is a declaration, not a library. A package whose
// tests import it says that those tests are a model acceptance suite: they
// run real models for minutes of whole-machine compute, and no amount of
// fixture sharing or scheduling shortens that. The gate runs such a package
// after the commit, in the deferred test lane and under the machine lease,
// beside the packages that need the device, instead of in the phase that
// blocks the commit. Its verdict still blocks the next gate, as every
// deferred lane's does.
package acceptancelane

// Declared is what a test file names to make the declaration an import the
// compiler keeps.
const Declared = true
