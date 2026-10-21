#include "tts_pool_test.hpp"
#include "ezlocalai/stt_pool.hpp"
void concurrency() {
    std::atomic<int> loads{0},closed{0}; TtsPoolOperations ops;
    ops.should_preload = [] { return true; };
    ops.load = [&](const auto &) { ++loads; auto model = std::make_shared<TtsPoolModel>(); model->close = [&] { ++closed; }; return model; };
    TtsPool pool(8,ops); std::vector<std::thread> threads; std::atomic<bool> same{true};
    for (uint64_t key = 1; key <= 32; ++key) threads.emplace_back([&,key] { const auto model = pool.get(key); if (pool.get(key) != model) same = false; });
    for (auto &thread : threads) thread.join();
    expect(same && loads == 8 && pool.snapshot().leases.size() == 32,"concurrent keyed leases and resident capacity");
    expect(!pool.unload_idle() && closed == 0,"media eviction preserves active leases");
    pool.destroy(100,false,true); expect(closed == 8 && pool.snapshot().leases.size() == 32,"force cleanup retains other lease entries");
    for (uint64_t key=1;key<=32;++key) pool.destroy(key,false);
    pool.warm();
    const auto before_idle_close=closed.load();
    expect(pool.unload_idle() && closed>before_idle_close && pool.snapshot().resident.empty(),"media eviction closes idle residents");
    expect(!pool.unload_idle(),"empty media eviction is a no-op");
    // A blocked first job makes the queue's serialization and FIFO order observable.
    auto release = std::make_shared<std::promise<void>>(); auto gate = release->get_future().share();
    auto started = std::make_shared<std::promise<void>>(); auto ready = started->get_future();
    auto done = std::make_shared<std::promise<void>>(); auto completed = done->get_future();
    auto order = std::make_shared<std::vector<int>>(); auto active = std::make_shared<std::atomic<int>>(0);
    schedule_stt_cleanup([=] { *active = 1; started->set_value(); gate.wait(); order->push_back(1); *active = 0; });
    ready.get();
    schedule_stt_cleanup([=] { if (*active) order->push_back(-1); order->push_back(2); throw std::runtime_error("expected queue failure"); });
    schedule_stt_cleanup([=] { order->push_back(3); done->set_value(); });
    release->set_value(); completed.get(); expect(*order == std::vector<int>({1,2,3}),"single FIFO cleanup worker survives failure");
}
int main(int argc,char **argv) { try {
    expect(argc == 3,"reference and pool fixture paths"); std::vector<Json> cases; gzFile file=gzopen(argv[2],"rb"); expect(file,"open compressed pool cases"); std::string raw; char buffer[65536]; int size; while((size=gzread(file,buffer,sizeof(buffer)))>0)raw.append(buffer,size); expect(size==0&&gzclose(file)==Z_OK,"read compressed pool cases"); std::istringstream input(raw); for(std::string line;std::getline(input,line);)if(!line.empty())cases.push_back(Json::parse(line));
    expect(hash(read(fs::path(argv[1])/"Pipes.py")) == cases[0].at("sha256"),"unchanged reference pool source");
    expect(hash(read(fs::path(argv[1])/"ezlocalai/LlamaTTS.py")) == cases[0].at("resolver_sha256"),"unchanged reference model resolver");
    expect(cases.size() == cases[0].at("cases").get<size_t>() + 1,"complete pool fixtures"); size_t actions = 0;
    for (size_t row = 1; row < cases.size(); ++row) {
        const auto &scenario = cases[row]; Environment env(scenario.at("config")); TtsPool pool(env.config.at("capacity"),env.operations());
        actions += run_pool_scenario(env,pool,scenario,"pool " + std::to_string(row));
    }
    {
        TtsPool *active=nullptr;bool transfer=false;TtsPoolOperations ops;ops.load=[](const auto &){return std::make_shared<TtsPoolModel>();};
        ops.mark_in_use=[&](bool in_use){if(in_use&&transfer){transfer=false;active->transfer(1,2);}};
        TtsPool pool(1,ops);active=&pool;const auto model=pool.get(1);transfer=true;
        expect(pool.get(1)==model&&pool.snapshot().leases.at(0).first==2,"reentrant mark retains resolved source lease");pool.destroy(2,false);
    }
    concurrency(); std::cout << cases.size() - 1 << " source pool scenarios, " << actions << " actions and concurrent/FIFO cleanup checks match\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
