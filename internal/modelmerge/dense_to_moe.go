package modelmerge

import (
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/evaluation"
	"overgo/internal/model"
	"overgo/internal/runrecord"
	"overgo/internal/tensor"
)

const (
	// DenseToMoEEvidenceScreenVersion is the immutable construction-screen version.
	DenseToMoEEvidenceScreenVersion = artifact.InitialDocumentVersion
	// DenseToMoEEvidenceScreenMediaType identifies admitted dense-to-MoE construction evidence.
	DenseToMoEEvidenceScreenMediaType = "application/vnd.overgo.dense-to-moe-evidence-screen+json"
	// DenseToMoEEvidenceScreenSchema identifies the construction-screen wire schema.
	DenseToMoEEvidenceScreenSchema = "overgo/dense-to-moe-evidence-screen/v1"
	// DenseToMoEPromotionVersion is the immutable promotion version.
	DenseToMoEPromotionVersion = artifact.InitialDocumentVersion
	// DenseToMoEPromotionMediaType identifies promoted dense-to-MoE models.
	DenseToMoEPromotionMediaType = "application/vnd.overgo.dense-to-moe-promotion+json"
	// DenseToMoEPromotionSchema identifies the promotion wire schema.
	DenseToMoEPromotionSchema = "overgo/dense-to-moe-promotion/v1"
)

// DenseToMoELayerTensors declares one exact dense input and MoE output seam.
type DenseToMoELayerTensors struct {
	Router     string `json:"router"`
	DenseGate  string `json:"dense_gate"`
	DenseUp    string `json:"dense_up"`
	DenseDown  string `json:"dense_down"`
	ExpertGate string `json:"expert_gate"`
	ExpertUp   string `json:"expert_up"`
	ExpertDown string `json:"expert_down"`
}

// DenseToMoEEvidenceScreen binds an exact generated artifact to its evidence,
// derived router, experts, target definition, recipe, and tensor seams.
type DenseToMoEEvidenceScreen struct {
	Version          uint16                   `json:"version"`
	Recipe           artifact.ID              `json:"recipe"`
	TargetDefinition artifact.ID              `json:"target_definition"`
	RouterEvidence   artifact.ID              `json:"router_evidence"`
	RouterPlan       artifact.ID              `json:"router_plan"`
	CandidateModel   artifact.ID              `json:"candidate_model"`
	Experts          []artifact.ID            `json:"experts"`
	Layers           []DenseToMoELayerTensors `json:"layers"`
	ID               artifact.ID              `json:"-"`
}

// DenseToMoECandidate is one exact generated snapshot and its admission facts.
type DenseToMoECandidate struct {
	Model   Snapshot
	Screen  DenseToMoEEvidenceScreen
	Router  model.DenseToMoERouterPlan
	Lineage []artifact.Lineage
	promote func(
		evaluation.OfflineArtifactGenerationPromotion,
		runrecord.Evaluation,
		runrecord.Evaluation,
	) (DenseToMoEPromotion, error)
}

// DenseToMoEPromotion binds generated-artifact evidence and held-out Pareto
// improvement to one exact dense-to-MoE candidate.
type DenseToMoEPromotion struct {
	Version             uint16      `json:"version"`
	Recipe              artifact.ID `json:"recipe"`
	CandidateModel      artifact.ID `json:"candidate_model"`
	EvidenceScreen      artifact.ID `json:"evidence_screen"`
	GenerationEvidence  artifact.ID `json:"generation_evidence"`
	BaselineEvaluation  artifact.ID `json:"baseline_evaluation"`
	CandidateEvaluation artifact.ID `json:"candidate_evaluation"`
	ID                  artifact.ID `json:"-"`
}

var denseToMoEEvidenceScreenCodec = artifact.JSONDocumentCodec(
	"dense-to-MoE evidence screen", artifact.KindEvidence,
	DenseToMoEEvidenceScreenMediaType, DenseToMoEEvidenceScreenSchema,
	canonicalizeDenseToMoEEvidenceScreen,
	func(value DenseToMoEEvidenceScreen) artifact.ID { return value.ID },
	func(value *DenseToMoEEvidenceScreen, id artifact.ID) { value.ID = id },
	func(value DenseToMoEEvidenceScreen) DenseToMoEEvidenceScreen {
		value.Experts = slices.Clone(value.Experts)
		value.Layers = slices.Clone(value.Layers)
		return value
	},
)

var denseToMoEPromotionCodec = artifact.JSONDocumentCodec(
	"dense-to-MoE promotion", artifact.KindEvidence,
	DenseToMoEPromotionMediaType, DenseToMoEPromotionSchema,
	canonicalizeDenseToMoEPromotion,
	func(value DenseToMoEPromotion) artifact.ID { return value.ID },
	func(value *DenseToMoEPromotion, id artifact.ID) { value.ID = id },
	func(value DenseToMoEPromotion) DenseToMoEPromotion { return value },
)

// DenseToMoEArtifact constructs expert tensors only after the derived router
// passes its held-out evidence screen and every expert inventory is exact.
func (compiler Compiler) DenseToMoEArtifact(
	recipe, targetDefinition artifact.ID,
	evidence model.DenseToMoERouterEvidence,
	experts []Snapshot,
	layers []DenseToMoELayerTensors,
) (DenseToMoECandidate, error) {
	if recipe.Kind() != artifact.KindRecipe || targetDefinition.Kind() != artifact.KindModelDefinition {
		return DenseToMoECandidate{}, errors.New("model merge: dense-to-MoE authority is invalid")
	}
	evidence, err := model.NewDenseToMoERouterEvidence(evidence)
	if err != nil {
		return DenseToMoECandidate{}, fmt.Errorf("model merge: seal dense-to-MoE evidence: %w", err)
	}
	router, err := model.CompileDenseToMoERouter(evidence)
	if err != nil {
		return DenseToMoECandidate{}, fmt.Errorf("model merge: dense-to-MoE router screen: %w", err)
	}
	if len(experts) != len(router.Experts) || len(layers) != len(router.Layers) {
		return DenseToMoECandidate{}, errors.New("model merge: dense-to-MoE expert or layer inventory differs")
	}
	for index, expert := range experts {
		if err := expert.ValidateIdentity(); err != nil {
			return DenseToMoECandidate{}, err
		}
		if expert.ID != router.Experts[index] {
			return DenseToMoECandidate{}, errors.New("model merge: dense-to-MoE expert order differs from router evidence")
		}
		if index > tensor.FirstOffset {
			if err := compatibleInventory(experts[tensor.FirstOffset], expert); err != nil {
				return DenseToMoECandidate{}, err
			}
		}
	}
	if targetDefinition == experts[tensor.FirstOffset].Definition {
		return DenseToMoECandidate{}, errors.New("model merge: dense-to-MoE target retains the dense model definition")
	}
	if err := validateDenseToMoELayers(experts[tensor.FirstOffset], router, layers); err != nil {
		return DenseToMoECandidate{}, err
	}
	weights := cloneWeights(experts[tensor.FirstOffset].Tensors)
	for layerIndex, names := range layers {
		delete(weights, names.DenseGate)
		delete(weights, names.DenseUp)
		delete(weights, names.DenseDown)
		for _, pair := range []struct{ source, target string }{
			{names.DenseGate, names.ExpertGate},
			{names.DenseUp, names.ExpertUp},
			{names.DenseDown, names.ExpertDown},
		} {
			stacked, stackErr := stackDenseToMoEWeight(experts, pair.source)
			if stackErr != nil {
				return DenseToMoECandidate{}, stackErr
			}
			weights[pair.target] = stacked
		}
		routerShape, shapeErr := tensor.NewShape(uint64(router.Width), uint64(len(experts)))
		if shapeErr != nil {
			return DenseToMoECandidate{}, shapeErr
		}
		weights[names.Router] = Weight{
			Layout: routerShape, Values: slices.Clone(router.Layers[layerIndex].Weights),
		}
	}
	candidate, err := compiler.Seal(targetDefinition, experts[tensor.FirstOffset].ID, weights)
	if err != nil {
		return DenseToMoECandidate{}, err
	}
	screen, err := denseToMoEEvidenceScreenCodec.New(DenseToMoEEvidenceScreen{
		Version: DenseToMoEEvidenceScreenVersion, Recipe: recipe, TargetDefinition: targetDefinition,
		RouterEvidence: evidence.ID, RouterPlan: router.ID, CandidateModel: candidate.ID,
		Experts: slices.Clone(router.Experts), Layers: slices.Clone(layers),
	})
	if err != nil {
		return DenseToMoECandidate{}, err
	}
	parents := []artifact.ID{recipe, targetDefinition, router.ID}
	parents = append(parents, router.Experts...)
	lineage := artifact.DependencyLineage(candidate.ID, parents...)
	lineage = append(lineage, screen.Lineage()...)
	result := DenseToMoECandidate{Model: candidate, Screen: screen, Router: router, Lineage: lineage}
	result.promote = func(
		generation evaluation.OfflineArtifactGenerationPromotion,
		baseline, composed runrecord.Evaluation,
	) (DenseToMoEPromotion, error) {
		if err := result.ValidateGeneratedArtifact(); err != nil {
			return DenseToMoEPromotion{}, err
		}
		if _, contentErr := generation.Content(); contentErr != nil {
			return DenseToMoEPromotion{}, errors.New("model merge: dense-to-MoE generation is not promoted")
		}
		if err := baseline.ValidateIdentity(); err != nil {
			return DenseToMoEPromotion{}, err
		}
		if err := composed.ValidateIdentity(); err != nil {
			return DenseToMoEPromotion{}, err
		}
		if generation.ProducedModel != result.Model.ID || baseline.Dataset != composed.Dataset ||
			baseline.Run == composed.Run || baseline.Recipe == result.Screen.Recipe ||
			composed.Recipe != result.Screen.Recipe || !strictMetricImprovement(baseline.Metrics, composed.Metrics) {
			return DenseToMoEPromotion{}, errors.New("model merge: dense-to-MoE promotion evidence is insufficient")
		}
		return denseToMoEPromotionCodec.New(DenseToMoEPromotion{
			Version: DenseToMoEPromotionVersion, Recipe: result.Screen.Recipe,
			CandidateModel: result.Model.ID, EvidenceScreen: result.Screen.ID,
			GenerationEvidence: generation.ID, BaselineEvaluation: baseline.ID, CandidateEvaluation: composed.ID,
		})
	}
	return result, nil
}

// ValidateGeneratedArtifact verifies the identities and exact cross-document
// bindings of a constructed dense-to-MoE candidate.
func (value DenseToMoECandidate) ValidateGeneratedArtifact() error {
	if err := value.Model.ValidateIdentity(); err != nil {
		return err
	}
	if err := value.Screen.ValidateIdentity(); err != nil {
		return err
	}
	if err := value.Router.ValidateIdentity(); err != nil {
		return err
	}
	if value.Screen.CandidateModel != value.Model.ID || value.Screen.TargetDefinition != value.Model.Definition ||
		value.Screen.RouterPlan != value.Router.ID || value.Screen.RouterEvidence != value.Router.Evidence ||
		!slices.Equal(value.Screen.Experts, value.Router.Experts) {
		return errors.New("model merge: dense-to-MoE generated artifact bindings differ")
	}
	return nil
}

// ValidateIdentity verifies exact evidence-screen identity.
func (value DenseToMoEEvidenceScreen) ValidateIdentity() error {
	return denseToMoEEvidenceScreenCodec.ValidateIdentity(value)
}

// Content returns exact evidence-screen content.
func (value DenseToMoEEvidenceScreen) Content() (artifact.Content, error) {
	return denseToMoEEvidenceScreenCodec.Content(value)
}

// Lineage binds the screen to the exact candidate and all construction authority.
func (value DenseToMoEEvidenceScreen) Lineage() []artifact.Lineage {
	parents := []artifact.ID{
		value.Recipe, value.TargetDefinition, value.RouterEvidence, value.RouterPlan, value.CandidateModel,
	}
	parents = append(parents, value.Experts...)
	return artifact.DependencyLineage(value.ID, parents...)
}

// Batch prepares atomic evidence-screen publication.
func (value DenseToMoEEvidenceScreen) Batch(key string) (artifact.Batch, error) {
	return denseToMoEEvidenceScreenCodec.Batch(key, value, value.Lineage(), nil)
}

// ValidateIdentity verifies exact promotion identity.
func (value DenseToMoEPromotion) ValidateIdentity() error {
	return denseToMoEPromotionCodec.ValidateIdentity(value)
}

// Content returns exact promotion content.
func (value DenseToMoEPromotion) Content() (artifact.Content, error) {
	return denseToMoEPromotionCodec.Content(value)
}

// Lineage binds promotion to construction, generation, and held-out evidence.
func (value DenseToMoEPromotion) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(
		value.ID, value.Recipe, value.CandidateModel, value.EvidenceScreen,
		value.GenerationEvidence, value.BaselineEvaluation, value.CandidateEvaluation,
	)
}

// Batch prepares atomic dense-to-MoE promotion publication.
func (value DenseToMoEPromotion) Batch(key string) (artifact.Batch, error) {
	return denseToMoEPromotionCodec.Batch(key, value, value.Lineage(), nil)
}

func validateDenseToMoELayers(reference Snapshot, router model.DenseToMoERouterPlan, layers []DenseToMoELayerTensors) error {
	claimed := make(map[string]bool, len(layers)*7)
	for layerIndex, names := range layers {
		allNames := []string{
			names.Router, names.DenseGate, names.DenseUp, names.DenseDown,
			names.ExpertGate, names.ExpertUp, names.ExpertDown,
		}
		for _, name := range allNames {
			if name == "" || claimed[name] {
				return errors.New("model merge: dense-to-MoE tensor seam is empty or duplicated")
			}
			claimed[name] = true
		}
		for _, output := range []string{names.Router, names.ExpertGate, names.ExpertUp, names.ExpertDown} {
			if _, collision := reference.Tensors[output]; collision {
				return fmt.Errorf("model merge: dense-to-MoE output tensor %q already exists", output)
			}
		}
		gate, gateFound := reference.Tensors[names.DenseGate]
		up, upFound := reference.Tensors[names.DenseUp]
		down, downFound := reference.Tensors[names.DenseDown]
		embeddingWidth := uint64(router.Width)
		if !gateFound || !upFound || !downFound || gate.Layout != up.Layout ||
			gate.Layout.Rank != tensor.PairedExtent || gate.Layout.Dims[tensor.FirstOffset] != embeddingWidth ||
			down.Layout.Rank != tensor.PairedExtent || down.Layout.Dims[tensor.FirstOffset] != gate.Layout.Dims[tensor.SingletonExtent] ||
			down.Layout.Dims[tensor.SingletonExtent] != embeddingWidth ||
			router.Layers[layerIndex].Layer != uint32(layerIndex) {
			return fmt.Errorf("model merge: dense-to-MoE layer %d geometry differs", layerIndex)
		}
	}
	return nil
}

func stackDenseToMoEWeight(experts []Snapshot, name string) (Weight, error) {
	base := experts[tensor.FirstOffset].Tensors[name]
	shape, err := tensor.PackBatchShape(base.Layout, uint64(len(experts)))
	if err != nil {
		return Weight{}, fmt.Errorf("model merge: stack dense-to-MoE tensor %q: %w", name, err)
	}
	capacity, ok := checked.MulInt(len(base.Values), len(experts))
	if !ok {
		return Weight{}, fmt.Errorf("model merge: stacked dense-to-MoE tensor %q overflows", name)
	}
	values := make([]float32, 0, capacity)
	for _, expert := range experts {
		values = append(values, expert.Tensors[name].Values...)
	}
	return Weight{Layout: shape, Values: values}, nil
}

func strictMetricImprovement(baseline, composed []runrecord.Metric) bool {
	if len(baseline) == 0 || len(baseline) != len(composed) {
		return false
	}
	improved := false
	for index, before := range baseline {
		after := composed[index]
		if before.Name != after.Name || before.Unit != after.Unit || before.Direction != after.Direction {
			return false
		}
		switch before.Direction {
		case runrecord.DirectionMaximize:
			if after.Value < before.Value {
				return false
			}
			improved = improved || after.Value > before.Value
		case runrecord.DirectionMinimize:
			if after.Value > before.Value {
				return false
			}
			improved = improved || after.Value < before.Value
		default:
			return false
		}
	}
	return improved
}

func canonicalizeDenseToMoEEvidenceScreen(value *DenseToMoEEvidenceScreen) error {
	if value == nil || value.Version != DenseToMoEEvidenceScreenVersion ||
		value.Recipe.Kind() != artifact.KindRecipe || value.TargetDefinition.Kind() != artifact.KindModelDefinition ||
		value.RouterEvidence.Kind() != artifact.KindEvidence || value.RouterPlan.Kind() != artifact.KindProfile ||
		value.CandidateModel.Kind() != artifact.KindModel || len(value.Experts) < tensor.PairedExtent || len(value.Layers) == 0 {
		return errors.New("model merge: invalid dense-to-MoE evidence screen")
	}
	seenExperts := make(map[artifact.ID]bool, len(value.Experts))
	nameCapacity, ok := checked.MulInt(len(value.Layers), len(denseToMoETensorNames(DenseToMoELayerTensors{})))
	if !ok {
		return errors.New("model merge: dense-to-MoE evidence-screen inventory overflows")
	}
	seenNames := make(map[string]bool, nameCapacity)
	for _, expert := range value.Experts {
		if expert.Kind() != artifact.KindModel || seenExperts[expert] {
			return errors.New("model merge: invalid dense-to-MoE evidence-screen expert")
		}
		seenExperts[expert] = true
	}
	for _, layer := range value.Layers {
		for _, name := range denseToMoETensorNames(layer) {
			if name == "" || seenNames[name] {
				return errors.New("model merge: invalid dense-to-MoE evidence-screen tensor seam")
			}
			seenNames[name] = true
		}
	}
	return nil
}

func denseToMoETensorNames(layer DenseToMoELayerTensors) []string {
	return []string{
		layer.Router, layer.DenseGate, layer.DenseUp, layer.DenseDown,
		layer.ExpertGate, layer.ExpertUp, layer.ExpertDown,
	}
}

func canonicalizeDenseToMoEPromotion(value *DenseToMoEPromotion) error {
	if value == nil || value.Version != DenseToMoEPromotionVersion ||
		value.Recipe.Kind() != artifact.KindRecipe || value.CandidateModel.Kind() != artifact.KindModel ||
		value.EvidenceScreen.Kind() != artifact.KindEvidence || value.GenerationEvidence.Kind() != artifact.KindEvidence ||
		value.BaselineEvaluation.Kind() != artifact.KindEvaluation || value.CandidateEvaluation.Kind() != artifact.KindEvaluation ||
		value.BaselineEvaluation == value.CandidateEvaluation {
		return errors.New("model merge: invalid dense-to-MoE promotion")
	}
	return nil
}
