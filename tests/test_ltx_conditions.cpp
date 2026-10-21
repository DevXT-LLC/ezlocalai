#include "pipeline/ltx_condition.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace {
using Json = nlohmann::json;
using namespace sd::pipeline;
void check(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
void near(float actual, float expected, float tolerance = 2e-6f) {
    if (std::abs(actual-expected) > tolerance) throw std::runtime_error("float mismatch: " + std::to_string(actual) + " vs " + std::to_string(expected));
}
}
int main(int argc, char **argv) {
    try {
        check(argc == 2,"fixture required");
        std::ifstream input(argv[1]); Json fixture; input >> fixture;
        size_t count = 0;
        for (const auto &row : fixture.at("preprocess")) {
            const auto index = ltx_condition_index(row.at("index"), (row.at("frames").get<int64_t>()-1)/8+1);
            check(index.has_value() == !row.at("resolved_index").is_null(),"condition skip");
            if (index) check(*index == row.at("resolved_index"),"latent index");
            if (row.contains("pixels")) {
                const uint32_t sw = row.at("source_width"), sh = row.at("source_height");
                const int w = row.at("width"), h = row.at("height");
                std::vector<uint8_t> pixels(sw*sh*3);
                for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = (i*73+19)%256;
                const auto resized = ltx_condition_pixels({sw,sh,3,pixels.data()},w,h);
                size_t offset = 0;
                for (int c = 0; c < 3; ++c) for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x)
                    near(resized.index(x,y,c,0)*2.f-1.f,row.at("pixels").at(offset++));
            }
            ++count;
        }
        for (const auto &row : fixture.at("coordinates")) {
            const auto positions = build_ltxv_video_positions(3,2,16,1,row.at("pixel_index"),1,row.at("fps"),32,8,true);
            size_t offset = 0;
            for (int axis = 0; axis < 3; ++axis) for (int token = 0; token < 6; ++token) for (int edge = 0; edge < 2; ++edge)
                near(positions.index(edge,axis,96+token,0),row.at("positions").at(offset++),0.f);
            ++count;
        }
        // Multiple conditions must append independent masks and coordinates;
        // earlier keyframes must not be counted as generated video frames.
        auto latent = sd::full<float>({3,2,16,4,1},0.f);
        auto mask = sd::full<float>({3,2,16,1,1},1.f);
        sd::Tensor<float> positions;
        int64_t conditions = 0;
        for (int i = 0; i < 3; ++i) {
            const auto keyframe = sd::full<float>({3,2,1,4,1},float(i+1));
            check(append_ltx_keyframe(latent,mask,positions,16,conditions,keyframe,1+i*8,.25f*i,24,32),"append keyframe");
        }
        check(conditions == 3 && latent.shape()[2] == 19 && positions.shape()[2] == 114,"keyframe counts");
        for (int i = 0; i < 3; ++i) for (int y = 0; y < 2; ++y) for (int x = 0; x < 3; ++x) {
            near(latent.index(x,y,16+i,0,0),float(i+1),0);
            near(mask.index(x,y,16+i,0,0),.25f*i,0);
            const int token = (16+i)*6+y*3+x;
            near(positions.index(0,0,token,0),(1.f+i*8)/24,0);
            near(positions.index(1,0,token,0),(2.f+i*8)/24,0);
        }
        check(!append_ltx_keyframe(latent,mask,positions,16,conditions,sd::full<float>({2,2,1,4,1},0.f),1,0,24,32),"reject mismatched shape");
        check(count == 137,"all installed-pipeline cases");
        std::cout << "Verified " << count << " installed LTX condition cases and multiple-keyframe tensor assembly\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
