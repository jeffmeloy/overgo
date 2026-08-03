package model

// LayerPlan: derived layer execution contract.
type LayerPlan struct {
	Layer         uint32
	GraphFamily   ArchitectureFamily
	CatalogFamily ArchitectureFamily
	Attention     AttentionPolicy
	Position      PositionPolicy
	Residual      ResidualPolicy
	FeedForward   FeedForwardPolicy
	CacheExtent   CacheExtent
	Recurrent     bool
	Sliding       bool
	UsesRoPE      bool
	MultiAxis     bool
	HasKV         bool
}

// PlanLayer: derives graph and cache behavior once per layer.
func (s Spec) PlanLayer(layer uint32, recurrent bool) LayerPlan {
	profile := s.Profile()
	info := LayerWeights{Recurrent: recurrent}
	return LayerPlan{
		Layer:         layer,
		GraphFamily:   profile.GraphFamily,
		CatalogFamily: profile.CatalogFamily,
		Attention:     profile.Attention,
		Position:      profile.Position,
		Residual:      profile.Residual,
		FeedForward:   profile.FeedForward,
		CacheExtent:   PrimaryCacheExtent(s, int(layer), info),
		Recurrent:     recurrent || s.IsRecurrentLayer(layer),
		Sliding:       s.IsSlidingLayer(layer),
		UsesRoPE:      s.UsesRoPE(layer),
		MultiAxis:     profile.Has(ArchitectureMultiAxisPositions),
		HasKV:         s.LayerHasKV(layer),
	}
}
