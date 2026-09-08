#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/sortformer_diar/frontend.h"
#include "engine/models/sortformer_diar/graph.h"
#include "engine/models/sortformer_diar/postprocess.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>

namespace diar = engine::models::sortformer_diar;
ggml_tensor * origin(ggml_tensor *node) {
    while (node && (node->op==GGML_OP_RESHAPE || node->op==GGML_OP_REPEAT || node->op==GGML_OP_VIEW || node->op==GGML_OP_CONT)) node=node->src[0];
    return node;
}
void save(const std::filesystem::path & path, const std::vector<float> & values) {
    std::ofstream out(path, std::ios::binary); out.write(reinterpret_cast<const char *>(values.data()), values.size()*sizeof(float));
    if (!out) throw std::runtime_error("capture write failed");
}
int main(int argc, char **argv) {
    try {
        if (argc==1) {
            struct Case {const char *name;std::vector<float> values;int min_frames;int pad_frames;};
            const std::vector<Case> cases={
                {"strict-threshold-ties",{.5f,.75f,.5f,.25f},0,0},
                {"open-last-tick",{.25f,.75f,.75f},0,0},
                {"pad-merge",{.75f,.25f,.75f,.25f,.25f},0,1},
                {"minimum",{.75f,.25f,.25f},2,0},
                {"inactive",{.5f,.25f,.5f},0,0}
            };
            std::cout<<"[";
            bool first=true;
            for (const auto &c:cases) {
                if(!first)std::cout<<",";first=false;
                std::cout<<"{\"name\":\""<<c.name<<"\",\"probabilities\":[";
                for(size_t i=0;i<c.values.size();++i){if(i)std::cout<<",";std::cout<<c.values[i];}
                std::cout<<"],\"min_frames\":"<<c.min_frames<<",\"pad_frames\":"<<c.pad_frames<<",\"spans\":[";
                diar::SortformerPostprocessConfig config;config.min_frames=c.min_frames;config.pad_frames=c.pad_frames;
                auto turns=diar::decode_sortformer_speaker_turns(c.values,c.values.size(),c.values.size(),1,1280,config);
                for(size_t i=0;i<turns.size();++i){if(i)std::cout<<",";std::cout<<"{\"start\":"<<turns[i].span.start_sample<<",\"end\":"<<turns[i].span.end_sample<<"}";}
                std::cout<<"]}";
            }
            std::cout<<"]\n";return 0;
        }
        if (argc!=4) throw std::runtime_error("capture MODEL WAV OUTPUT_DIRECTORY");
        const std::filesystem::path output(argv[3]); std::filesystem::create_directories(output);
        auto assets=diar::load_sortformer_assets(argv[1]);
        engine::core::BackendConfig backend; backend.type=engine::core::BackendType::Cpu; backend.threads=8;
        engine::core::ExecutionContext execution(backend);
        auto weights=diar::load_sortformer_diar_weights(*assets,execution.backend(),execution.backend_type(),engine::assets::TensorStorageType::F32,engine::assets::TensorStorageType::F32,128ull*1024*1024);
        const auto wav=engine::audio::read_wav_f32(std::filesystem::path(argv[2]));
        engine::runtime::AudioBuffer audio{wav.sample_rate,wav.channels,wav.samples};
        const auto features=diar::compute_sortformer_features(audio,*assets,backend.threads);
        const auto context=diar::make_sortformer_fixed_context_contract_for_samples(wav.samples.size(),*assets);
        std::unique_ptr<diar::SortformerInferenceGraph> graph;
        diar::ensure_sortformer_inference_graph(graph,execution,*assets,*weights,512ull*1024*1024,context.feature_frames,context.encoder_frames);
        auto padded=features.time_major; padded.resize(context.feature_frames*assets->feature_config.num_mel_bins,0);
        engine::core::write_tensor_f32(graph->input,padded); save(output/"features.f32",features.time_major);
        const auto &fc=assets->model_config.fc_encoder;
        auto valid=features.valid_frames;
        std::vector<int32_t> mask;
        for (const auto target : {graph->mask1,graph->mask2,graph->encoder_keep_mask}) {
            valid=diar::sortformer_conv_valid_length(valid,fc.subsampling_conv_kernel_size,fc.subsampling_conv_stride,(fc.subsampling_conv_kernel_size-1)/2);
            diar::fill_sortformer_keep_mask(mask,target.shape.dims[1],valid); engine::core::write_tensor_i32(target,mask);
        }
        std::vector<float> attention;
        diar::fill_sortformer_transformer_attention_mask(attention,context.encoder_frames,valid);
        engine::core::write_tensor_f32(graph->transformer_mask,attention);
        engine::core::set_backend_threads(execution.backend(),backend.threads);
        if (engine::core::compute_backend_graph(execution.backend(),graph->graph,graph->plan)!=GGML_STATUS_SUCCESS) throw std::runtime_error("reference compute failed");
        std::vector<float> values;
        engine::core::read_tensor_f32_into(graph->output_probabilities.tensor,values); save(output/"probabilities.f32",values);
        std::ofstream nodes(output/"graph.tsv");
        for (int i=0;i<ggml_graph_n_nodes(graph->graph);++i) {
            auto node=ggml_graph_node(graph->graph,i);
            nodes<<i<<'\t'<<ggml_op_name(node->op)<<'\t'<<node->name;
            for (auto dim:node->ne) nodes<<'\t'<<dim;
            for (auto source:node->src) if (source) nodes<<'\t'<<source->name;
            nodes<<'\n';
            // Bind boundary identity to the loaded affine bias, not guessed graph indices.
            if (node->type==GGML_TYPE_F32 && ggml_is_contiguous(node) && node->op==GGML_OP_ADD) {
                for (size_t layer=0;layer<weights->conformer_layers.size();++layer) {
                    if (origin(node->src[1])==weights->conformer_layers[layer].norm_out.bias->tensor) {
                        engine::core::read_tensor_f32_into(node,values); save(output/("encoder-"+std::to_string(layer)+".f32"),values);
                    }
                }
                for (size_t layer=0;layer<weights->transformer_layers.size();++layer) {
                    if (origin(node->src[1])==weights->transformer_layers[layer].final_layer_norm.bias->tensor) {
                        engine::core::read_tensor_f32_into(node,values); save(output/("postnorm-"+std::to_string(layer)+".f32"),values);
                    }
                }
                if (origin(node->src[1])==weights->head.encoder_proj.bias->tensor) {
                    engine::core::read_tensor_f32_into(node,values); save(output/"projection.f32",values);
                }
            }
        }
        std::vector<float> probabilities; engine::core::read_tensor_f32_into(graph->output_probabilities.tensor,probabilities);
        auto turns=diar::decode_sortformer_speaker_turns(probabilities,context.encoder_frames,valid,assets->model_config.num_speakers,assets->feature_config.hop_length*fc.subsampling_factor,{});
        std::cout<<"features="<<features.frames<<" valid_features="<<features.valid_frames<<" encoder="<<context.encoder_frames<<" valid_encoder="<<valid<<"\n";
        for (auto turn:turns) std::cout<<turn.span.start_sample<<' '<<turn.span.end_sample<<' '<<turn.speaker_id<<'\n';
    } catch (const std::exception &e) {std::cerr<<e.what()<<'\n'; return 1;}
}
