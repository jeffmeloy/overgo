package gate

// cloneGateContext builds a second context over the same gate state field by
// field: the context carries locks and is never copied as a value.
func cloneGateContext(g *gateContext) *gateContext {
	return &gateContext{
		repo: g.repo, candidateRoot: g.candidateRoot, candidateTree: g.candidateTree, paths: g.paths,
		planRef: g.planRef, messageFile: g.messageFile, storePath: g.storePath, start: g.start,
		environment: g.environment, preparation: g.preparation, preparationCommit: g.preparationCommit,
		source: g.source, baseSource: g.baseSource, profile: g.profile, profileDirty: g.profileDirty,
		preflight: g.preflight, stepEvidence: g.stepEvidence, cachePaths: g.cachePaths, retryCache: g.retryCache,
		checkpointMemos: g.checkpointMemos, verificationBatch: g.verificationBatch, structural: g.structural,
		packageGraph: g.packageGraph, selection: g.selection, selectionID: g.selectionID, manifestPlan: g.manifestPlan,
		terminal: g.terminal, baseManifest: g.baseManifest, candidateManifest: g.candidateManifest,
		manifestDelta: g.manifestDelta, manifestImpact: g.manifestImpact, manifestMetrics: g.manifestMetrics,
		strategy: g.strategy, diff: g.diff, completionAuthority: g.completionAuthority, indexBefore: g.indexBefore,
		mergeBefore: g.mergeBefore, planProjection: g.planProjection, mergeSourceStore: g.mergeSourceStore,
		mergeAuthority: g.mergeAuthority, planHead: g.planHead, committedHead: g.committedHead,
		commitInterrupted: g.commitInterrupted, acceptedTree: g.acceptedTree, runCommand: g.runCommand,
	}
}
