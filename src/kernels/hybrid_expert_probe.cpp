// Offline, real-weight CPU/GPU block-partition probe. No online cache or dispatch changes.
// All quantized weight bytes are preserved. Workload: one real activation and ten routed
// experts, compared with 3 complete GPU experts + 7 CPU experts at equal active GPU weight bytes.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_moe.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

namespace k = strata::kernels;
namespace cpu = strata::kernels::cpu;
namespace fs = std::filesystem;
constexpr int H = 2560, FF = 640, G = 10;
using Clock = std::chrono::steady_clock;
using Blob = std::vector<uint8_t>;
using Blobs = std::vector<Blob>;
using Vec = std::vector<float>;
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
#define CU(expr) do { auto ec = (expr); if (ec != cudaSuccess) throw std::runtime_error(std::string(#expr)+": "+cudaGetErrorString(ec)); } while (0)
template<class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { CU(cudaMalloc(reinterpret_cast<void**>(&p), std::max<size_t>(n,1)*sizeof(T))); }
    ~Device() { if (p) cudaFree(p); }
    Device(const Device&) = delete;
};
struct Pinned {
    float* p = nullptr;
    explicit Pinned(size_t n) { CU(cudaMallocHost(reinterpret_cast<void**>(&p), n*sizeof(float))); }
    ~Pinned() { if (p) cudaFreeHost(p); }
    Pinned(const Pinned&) = delete;
};
struct Stream {
    cudaStream_t s = nullptr;
    Stream() { CU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
    ~Stream() { if (s) cudaStreamDestroy(s); }
    Stream(const Stream&) = delete;
};
template<class T> void read(std::istream& in, T* p, size_t n) {
    require(n <= 1000000, "input allocation limit");
    in.read(reinterpret_cast<char*>(p), n*sizeof(T)); require(bool(in), "truncated probe input");
}
struct Case {
    int32_t layer, pos, token;
    std::array<float,H> x;
    std::array<int32_t,G> ids, tiers;
    std::array<float,G> weights;
};
std::vector<Case> cases(const fs::path& path) {
    std::ifstream in(path,std::ios::binary); char magic[8]; uint32_t dims[4];
    read(in,magic,8); read(in,dims,4);
    require(!std::memcmp(magic,"SHBv0001",8) && dims[0]>=1 && dims[0]<=120 && dims[1]==H && dims[2]==G && dims[3]==FF,"invalid probe header");
    std::vector<Case> out(dims[0]);
    for (auto& c:out) {
        read(in,&c.layer,1); read(in,&c.pos,1); read(in,&c.token,1);
        read(in,c.x.data(),H); read(in,c.ids.data(),G); read(in,c.weights.data(),G); read(in,c.tiers.data(),G);
        require(c.layer==0 || c.layer==1 || c.layer==35 || c.layer==36,"unsupported layer");
        require(c.pos>=0 && c.token>=0,"invalid position/token");
        for (auto x:c.x) require(std::isfinite(x),"nonfinite input");
        std::set<int> ids;
        for (int e=0;e<G;++e) {
            require(c.ids[e]>=0 && c.ids[e]<512 && ids.insert(c.ids[e]).second,"invalid expert IDs");
            require(std::isfinite(c.weights[e]) && c.weights[e]>=0,"invalid routing weight");
            require(c.tiers[e]>=0 && c.tiers[e]<=2,"invalid source tier");
        }
    }
    require(in.peek()==std::char_traits<char>::eof(),"trailing probe input");
    return out;
}
cpu::NativeFmt format(int gu,int down,int ff) {
    cpu::NativeFmt f; std::string error;
    if (!cpu::native_fmt(gu,down,H,ff,f,error)) throw std::runtime_error(error);
    require(k::native_expert_supported(gu,down,H,ff),"GPU partial format unsupported");
    const auto l=k::native_expert_layout(gu,down,H,ff);
    require(l.bytes==f.bytes && l.gu_row==f.gu_row && l.d_row==f.d_row,"CPU/GPU weight geometry mismatch");
    return f;
}
Blobs load(const strata::GgufModel& model,const Case& c,cpu::NativeFmt& fmt) {
    const strata::TensorInfo* tensors[3]; size_t shard[3]; const char* roles[]={"gate","up","down"};
    for (int i=0;i<3;++i) {
        tensors[i]=model.find("blk."+std::to_string(c.layer)+".ffn_"+roles[i]+"_exps.weight",&shard[i]);
        require(tensors[i] && model.in_bounds(*tensors[i],shard[i]),"missing or out of bounds expert tensor");
        require(tensors[i]->shape.size()==3 && tensors[i]->shape[2]==512,"unexpected expert tensor shape");
    }
    require(tensors[0]->shape==tensors[1]->shape && tensors[0]->type==tensors[1]->type &&
            tensors[0]->shape[0]==H && tensors[0]->shape[1]==FF && tensors[2]->shape[0]==FF && tensors[2]->shape[1]==H,"unexpected expert geometry");
    fmt=format(int(tensors[0]->type),int(tensors[2]->type),FF);
    require((fmt.gu_type==16 || fmt.gu_type==17 || fmt.gu_type==18) && (fmt.d_type==20 || fmt.d_type==42),"probe supports only measured format pairs");
    const size_t stride[]={fmt.up_off,fmt.up_off,fmt.bytes-fmt.down_off};
    for (int i=0;i<3;++i) require(strata::tensor_payload_bytes(*tensors[i])==512*stride[i],"expert tensor byte size mismatch");
    Blobs blobs(G,Blob(fmt.bytes));
    for (int e=0;e<G;++e) for (int i=0;i<3;++i) {
        const size_t off=i==0?0:(i==1?fmt.up_off:fmt.down_off);
        std::memcpy(blobs[e].data()+off,model.shard(shard[i]).tensor_data(*tensors[i])+size_t(c.ids[e])*stride[i],stride[i]);
    }
    return blobs;
}
// Copy complete 64-neuron blocks: gate/up rows plus the corresponding block(s)
// from every down row. Q8 activation blocks stay intact as well.
Blobs pack(const Blobs& full,const cpu::NativeFmt& f,const std::vector<int>& blocks) {
    const auto p=format(f.gu_type,f.d_type,int(blocks.size())*64);
    Blobs out(full.size(),Blob(p.bytes));
    const size_t db=f.d_row/10;
    for (size_t e=0;e<full.size();++e) for (size_t b=0;b<blocks.size();++b) {
        const int original=blocks[b]; require(original>=0 && original<10,"bad neuron block");
        for (int role=0;role<2;++role)
            std::memcpy(out[e].data()+role*p.up_off+b*64*f.gu_row,
                        full[e].data()+role*f.up_off+original*64*f.gu_row,64*f.gu_row);
        for (int row=0;row<H;++row)
            std::memcpy(out[e].data()+p.down_off+row*p.d_row+b*db,
                        full[e].data()+f.down_off+row*f.d_row+original*db,db);
    }
    return out;
}
void check_pack(const Blobs& full,const cpu::NativeFmt& f,const Blobs& hot,const Blobs& cold,
                const std::vector<int>& hi,const std::vector<int>& ci) {
    std::set<int> seen(hi.begin(),hi.end()); seen.insert(ci.begin(),ci.end());
    require(seen.size()==10 && hi.size()+ci.size()==10,"partition is not disjoint and exhaustive");
    Blobs rebuilt(full.size(),Blob(f.bytes));
    for (int side=0;side<2;++side) {
        const auto& ids=side?ci:hi; const auto& src=side?cold:hot;
        const auto q=format(f.gu_type,f.d_type,int(ids.size())*64); const size_t db=f.d_row/10;
        for (size_t e=0;e<full.size();++e) for (size_t b=0;b<ids.size();++b) {
            for (int role=0;role<2;++role)
                std::memcpy(rebuilt[e].data()+role*f.up_off+ids[b]*64*f.gu_row,src[e].data()+role*q.up_off+b*64*f.gu_row,64*f.gu_row);
            for (int row=0;row<H;++row)
                std::memcpy(rebuilt[e].data()+f.down_off+row*f.d_row+ids[b]*db,src[e].data()+q.down_off+row*q.d_row+b*db,db);
        }
    }
    require(rebuilt==full,"packed weights do not reconstruct the original bytes");
}
struct CpuBatch {
    const cpu::NativeFmt fmt;
    alignas(64) std::array<uint8_t,cpu::kNativeActBytes> act{};
    Vec out;
    std::vector<cpu::ExpertJobMulti> jobs;
    explicit CpuBatch(const cpu::NativeFmt& f,const Blobs& blobs):fmt(f),out(blobs.size()*H+16,12345.f),jobs(blobs.size()) {
        for (size_t e=0;e<blobs.size();++e) { jobs[e].blob=blobs[e].data(); jobs[e].nt=1; jobs[e].nact[0]=act.data(); jobs[e].out[0]=out.data()+e*H; }
    }
    void run(cpu::ExpertPool& pool,const float* x) {
        cpu::native_quant_act(fmt,x,act.data());
        pool.run_split_multi_native(fmt,jobs.data(),int(jobs.size()));
    }
    Vec values() const {
        for (size_t i=jobs.size()*H;i<out.size();++i) require(out[i]==12345.f,"CPU output guard overwritten");
        return Vec(out.begin(),out.begin()+jobs.size()*H);
    }
};
struct GpuBatch {
    const k::NativeExpertLayout fmt; const int n;
    Stream stream, copy;
    Device<uint8_t> blob, xq, scratch;
    Device<float> x, parts, weights, cpu_sum, sum;
    Device<unsigned long long> ptrs;
    Device<int32_t> starts, count, dst, tok;
    Pinned input, add;
    GpuBatch(const cpu::NativeFmt& f,const Blobs& bs,const float* xhost,const std::vector<float>& ws):
        fmt(k::native_expert_layout(f.gu_type,f.d_type,H,f.n_ff)),n(int(bs.size())),
        blob(std::max<size_t>(1,bs.size())*f.bytes),xq(H/32*36),scratch(k::native_expert_scratch_bytes(std::max(1,n),f.n_ff)),
        x(H),parts(std::max(1,n)*H),weights(std::max(1,n)),cpu_sum(H),sum(H),
        ptrs(std::max(1,n)),starts(std::max(1,n)+1),count(1),dst(std::max(1,n)),tok(std::max(1,n)),input(H),add(H) {
        require(ws.size()==bs.size(),"GPU weight count mismatch");
        std::vector<unsigned long long> pp(n); std::vector<int32_t> ss(n+1),dd(n),tt(n,0);
        for (int e=0;e<n;++e) {
            CU(cudaMemcpy(blob.p+size_t(e)*f.bytes,bs[e].data(),f.bytes,cudaMemcpyHostToDevice));
            pp[e]=reinterpret_cast<unsigned long long>(blob.p+size_t(e)*f.bytes); ss[e]=dd[e]=e;
        }
        ss[n]=n;
        CU(cudaMemcpy(x.p,xhost,H*4,cudaMemcpyHostToDevice));
        if (n) {
            CU(cudaMemcpy(ptrs.p,pp.data(),n*8,cudaMemcpyHostToDevice)); CU(cudaMemcpy(starts.p,ss.data(),(n+1)*4,cudaMemcpyHostToDevice));
            CU(cudaMemcpy(dst.p,dd.data(),n*4,cudaMemcpyHostToDevice)); CU(cudaMemcpy(tok.p,tt.data(),n*4,cudaMemcpyHostToDevice));
            CU(cudaMemcpy(weights.p,ws.data(),n*4,cudaMemcpyHostToDevice)); CU(cudaMemcpy(count.p,&n,4,cudaMemcpyHostToDevice));
        }
    }
    void launch() {
        if (!n) return;
        k::quantize_q8_1_rows(x.p,1,H,xq.p,stream.s);
        k::native_expert_grouped(fmt,ptrs.p,starts.p,count.p,dst.p,tok.p,n,n,xq.p,scratch.p,parts.p,stream.s);
    }
    void pull_input() { CU(cudaMemcpyAsync(input.p,x.p,H*4,cudaMemcpyDeviceToHost,copy.s)); CU(cudaStreamSynchronize(copy.s)); }
    void finish(bool has_cpu) {
        if (has_cpu) CU(cudaMemcpyAsync(cpu_sum.p,add.p,H*4,cudaMemcpyHostToDevice,stream.s));
        if (n) k::native_moe_combine(parts.p,weights.p,has_cpu?cpu_sum.p:nullptr,sum.p,H,n,stream.s);
        else CU(cudaMemcpyAsync(sum.p,cpu_sum.p,H*4,cudaMemcpyDeviceToDevice,stream.s));
        CU(cudaStreamSynchronize(stream.s));
    }
    Vec get_parts() { Vec out(size_t(n)*H); if (n) CU(cudaMemcpy(out.data(),parts.p,out.size()*4,cudaMemcpyDeviceToHost)); return out; }
    Vec get_sum() { Vec out(H); CU(cudaMemcpy(out.data(),sum.p,H*4,cudaMemcpyDeviceToHost)); return out; }
};
double l2(const Vec& a,const Vec& b) {
    require(a.size()==b.size(),"L2 dimension mismatch"); double n=0,d=0;
    for (size_t i=0;i<a.size();++i) { require(std::isfinite(a[i]) && std::isfinite(b[i]),"nonfinite output"); n+=std::pow(double(a[i])-b[i],2); d+=double(b[i])*b[i]; }
    require(d>0,"zero reference output"); return std::sqrt(n/d);
}
Vec add(const Vec& a,const Vec& b) { require(a.size()==b.size(),"sum dimension mismatch"); Vec c(a.size()); for (size_t i=0;i<a.size();++i)c[i]=a[i]+b[i]; return c; }
Vec weighted(const Vec& parts,const std::vector<float>& weights) {
    require(parts.size()==weights.size()*H,"weighted output shape mismatch"); Vec out(H);
    for (int r=0;r<H;++r) { double v=0; for (size_t e=0;e<weights.size();++e) v+=double(weights[e])*parts[e*H+r]; out[r]=float(v); }
    return out;
}
void values(std::ostream& out,const std::vector<double>& a) { out<<'['; for (size_t i=0;i<a.size();++i) { if(i)out<<','; out<<a[i]; } out<<']'; }
struct Arm {
    std::string name; std::vector<double> us;
    Vec output;
    size_t gpu_bytes=0,cpu_bytes=0;
};
Arm measure(const std::string& name,const Blobs& gpu_blobs,const cpu::NativeFmt& gf,const std::vector<float>& gw,
            const Blobs& cpu_blobs,const cpu::NativeFmt& cf,const std::vector<float>& cw,const Case& c,
            cpu::ExpertPool& pool,int iterations,int rounds) {
    GpuBatch gpu(gf,gpu_blobs,c.x.data(),gw); CpuBatch host(cf,cpu_blobs);
    auto step=[&] {
        gpu.launch();
        if (!cpu_blobs.empty()) {
            gpu.pull_input(); host.run(pool,gpu.input.p);
            std::fill_n(gpu.add.p,H,0.f);
            for (size_t e=0;e<cw.size();++e) for (int r=0;r<H;++r) gpu.add.p[r]+=cw[e]*host.out[e*H+r];
        }
        gpu.finish(!cpu_blobs.empty());
    };
    for (int i=0;i<8;++i) step();
    Arm result{name,{}, {},gpu_blobs.size()*gf.bytes,cpu_blobs.size()*cf.bytes};
    for (int rep=0;rep<rounds;++rep) {
        const auto start=Clock::now(); for(int i=0;i<iterations;++i)step();
        result.us.push_back(std::chrono::duration<double,std::micro>(Clock::now()-start).count()/iterations);
    }
    result.output=gpu.get_sum(); const auto cp=host.values();
    Vec expected=weighted(cp,cw);
    if (!gpu_blobs.empty()) expected=add(expected,weighted(gpu.get_parts(),gw));
    require(l2(result.output,expected)<1e-5,"GPU final merge disagrees with independently weighted partials");
    return result;
}

int main(int argc,char** argv) {
    if (argc!=6 && argc!=7) { std::cerr<<"usage: hybrid_expert_probe input.shb model.gguf output.jsonl iterations rounds [paired]\n";return 2; }
    try {
        const bool paired=argc==7;
        // lab gate-first probe: per-token neuron selection differs between a partition and the whole expert
        const bool gate_probe=std::getenv("STRATA_GATE_KEEP")!=nullptr;
        require(!paired || std::string(argv[6])=="paired","unknown probe mode");
        require(!fs::exists(fs::u8path(argv[3])),"output already exists");
        const int iterations=std::stoi(argv[4]),rounds=std::stoi(argv[5]);
        require(iterations>=1 && iterations<=100 && rounds>=1 && rounds<=9,"invalid benchmark bounds");
        const auto inputs=cases(fs::u8path(argv[1]));
        const strata::GgufModel model(strata::gguf_split_paths(argv[2]));
        cpu::ExpertPool pool(7,true,true);
        const auto topology=cpu::detect_cpu_topology(true); const auto previous=cpu::pin_current_thread(topology.host_core);
        cudaDeviceProp properties{}; CU(cudaGetDeviceProperties(&properties,0));
        k::native_grouped_set_v1(false);
        std::ofstream out(fs::u8path(argv[3]),std::ios::binary); require(bool(out),"cannot create output"); out<<std::setprecision(12);
        out<<"{\"type\":\"metadata\",\"device\":\""<<properties.name<<"\",\"vram_bytes\":"<<properties.totalGlobalMem
           <<",\"workers\":"<<pool.workers()<<",\"host_works\":true,\"iterations\":"<<iterations<<",\"rounds\":"<<rounds
           <<",\"cases\":"<<inputs.size()<<",\"paired_rounds\":"<<(paired?"true":"false")<<",\"hot_blocks_64\":[0,4,9],\"scope\":\"warm single-token expert microbenchmark; frozen weights; no end-to-end t/s; original cache tiers ignored in paired arms\"}\n";
        int index=0;
        for (const auto& c:inputs) {
            cpu::NativeFmt full; const auto bs=load(model,c,full);
            const auto hf=format(full.gu_type,full.d_type,192),cf=format(full.gu_type,full.d_type,448);
            const auto began=Clock::now(); const std::vector<int> hi{0,4,9},ci{1,2,3,5,6,7,8};
            const auto hot=pack(bs,full,hi),cold=pack(bs,full,ci); check_pack(bs,full,hot,cold,hi,ci);
            const double packing_us=std::chrono::duration<double,std::micro>(Clock::now()-began).count();
            require(10*hf.bytes==3*full.bytes && hf.bytes+cf.bytes==full.bytes,"partition byte budget mismatch");
            const std::vector<float> ws(c.weights.begin(),c.weights.end());
            CpuBatch cpu_full(full,bs),cpu_hot(hf,hot),cpu_cold(cf,cold);
            cpu_full.run(pool,c.x.data());cpu_hot.run(pool,c.x.data());cpu_cold.run(pool,c.x.data());
            const Vec ref_cpu=cpu_full.values(), hot_cpu=cpu_hot.values(),cold_cpu=cpu_cold.values();
            const double cpu_split_error=l2(add(hot_cpu,cold_cpu),ref_cpu);
            require(gate_probe || cpu_split_error<1e-5,"CPU partition parity failed");
            Vec ref_gpu,hot_gpu,cold_gpu;
            { GpuBatch g(full,bs,c.x.data(),ws);g.launch();g.finish(false);ref_gpu=g.get_parts(); }
            { GpuBatch g(hf,hot,c.x.data(),ws);g.launch();g.finish(false);hot_gpu=g.get_parts(); }
            { GpuBatch g(cf,cold,c.x.data(),ws);g.launch();g.finish(false);cold_gpu=g.get_parts(); }
            const double gpu_split_error=l2(add(hot_gpu,cold_gpu),ref_gpu);
            require(gate_probe || gpu_split_error<1e-5,"GPU partition parity failed");
            const Vec hybrid_reference=weighted(add(hot_gpu,cold_cpu),ws);
            const Vec gpu_reference=weighted(ref_gpu,ws),cpu_reference=weighted(ref_cpu,ws);
            Blobs three(bs.begin(),bs.begin()+3),seven(bs.begin()+3,bs.end());
            const std::vector<float> w3(ws.begin(),ws.begin()+3),w7(ws.begin()+3,ws.end());
            std::vector<Arm> arms;
            if (paired) {
                // Equal active weight bytes, AB/BA on each case in every round.
                // Recreate both arms symmetrically, warm up, then time transfers +
                // compute + merge. Construction/upload remain outside steady state.
                for (int rep=0;rep<rounds;++rep) for (int step=0;step<2;++step) {
                    const int mode=((index+rep)%2)?1-step:step;
                    auto a=mode==0?measure("gpu3_cpu7",three,full,w3,seven,full,w7,c,pool,iterations,1)
                                  :measure("gpu30_cpu70",hot,hf,ws,cold,cf,ws,c,pool,iterations,1);
                    auto found=std::find_if(arms.begin(),arms.end(),[&](const Arm& item){return item.name==a.name;});
                    if (found==arms.end()) arms.push_back(std::move(a));
                    else { found->us.push_back(a.us.front());found->output=std::move(a.output); }
                }
            } else for (int step=0;step<4;++step) {
                // Report every round, not the best sample; alternate order by case.
                int mode=(index%2)?3-step:step;
                if (mode==0) arms.push_back(measure("cpu10",{},full,{},bs,full,ws,c,pool,iterations,rounds));
                if (mode==1) arms.push_back(measure("gpu3_cpu7",three,full,w3,seven,full,w7,c,pool,iterations,rounds));
                if (mode==2) arms.push_back(measure("gpu30_cpu70",hot,hf,ws,cold,cf,ws,c,pool,iterations,rounds));
                if (mode==3) arms.push_back(measure("gpu10",bs,full,ws,{},full,{},c,pool,iterations,rounds));
            }
            out<<"{\"type\":\"case\",\"index\":"<<index<<",\"layer\":"<<c.layer<<",\"position\":"<<c.pos
               <<",\"gu\":"<<full.gu_type<<",\"down\":"<<full.d_type<<",\"full_expert_bytes\":"<<full.bytes
               <<",\"pack_and_inverse_check_us\":"<<packing_us<<",\"pack_bitwise_inverse\":true,\"cpu_split_l2\":"<<cpu_split_error
               <<",\"gpu_split_l2\":"<<gpu_split_error<<",\"full_cpu_vs_gpu_routed_l2\":"<<l2(cpu_reference,gpu_reference)
               <<",\"hybrid_vs_gpu_routed_l2\":"<<l2(hybrid_reference,gpu_reference)<<",\"arms\":[";
            for (size_t i=0;i<arms.size();++i) {
                const auto& a=arms[i]; if(i)out<<',';
                const Vec& reference=a.name=="gpu30_cpu70"?hybrid_reference:(a.name=="cpu10"?cpu_reference:gpu_reference);
                if (a.name!="gpu3_cpu7" && !(gate_probe && a.name=="gpu30_cpu70")) require(l2(a.output,reference)<1e-5,"timed path parity failed");
                out<<"{\"name\":\""<<a.name<<"\",\"gpu_active_weight_bytes\":"<<a.gpu_bytes<<",\"cpu_active_weight_bytes\":"<<a.cpu_bytes
                   <<",\"routed_l2_vs_full_gpu\":"<<l2(a.output,gpu_reference)<<",\"round_us\":"; values(out,a.us);out<<'}';
            }
            out<<"]}\n"; out.flush();require(bool(out),"failed writing result");
            std::cout<<"CASE "<<index++<<" layer "<<c.layer<<" CPU split "<<cpu_split_error<<" GPU split "<<gpu_split_error<<" PASS\n"<<std::flush;
        }
        cpu::restore_thread_affinity(previous);
        std::cout<<"HYBRID_PROBE_PASS "<<inputs.size()<<"\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<"hybrid_expert_probe: "<<e.what()<<'\n';return 1; }
}
