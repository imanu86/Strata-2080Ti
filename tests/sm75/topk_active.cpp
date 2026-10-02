
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
namespace k = strata::kernels;
void ck(cudaError_t e) { if(e != cudaSuccess) { std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); std::exit(2); } }
int main(int argc, char** argv) {
    const int64_t ctx = std::atoll(argv[1]), nq = std::atoll(argv[2]), stride = 65538;
    const int mode = std::atoi(argv[3]), reps = 7;
    const auto s = k::qsa_real_shapes();
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells,s);
    std::vector<int32_t> steps(nq*k::kStepCount);
    std::vector<float> scores(nq*stride, -999.f);
    std::mt19937 rng(742913);
    std::uniform_real_distribution<float> random(0.f,200.f);
    for(int64_t i=0;i<nq;++i) {
        auto st=steps.data()+i*k::kStepCount;
        const int64_t pos=ctx-nq+i;
        st[k::kStepPos]=int32_t(pos); st[k::kStepNKv]=int32_t(pos+1);
        st[k::kStepNBid]=int32_t((pos+1)/s.idx_block);
        st[k::kStepWidth]=int32_t(k::qsa_selection_width(pos+1,s));
        for(int64_t j=0;j<=st[k::kStepNBid];++j)
            scores[i*stride+j]=mode==1 ? 0.f : mode==2 ? float(j%7) : random(rng);
        if((pos+1)%s.idx_block) scores[i*stride+st[k::kStepNBid]]+=1e9f;
    }
    int64_t active=steps[(nq-1)*k::kStepCount+k::kStepNBid]+1;
    if(mode==3) active=0; // safe fallback, captured decode must use original API
    if(mode==4) active=stride+1;
    float* ds=nullptr; int32_t *dt=nullptr,*da=nullptr,*db=nullptr;
    ck(cudaMalloc(&ds,scores.size()*4)); ck(cudaMalloc(&dt,steps.size()*4));
    ck(cudaMalloc(&da,nq*cap*4)); ck(cudaMalloc(&db,nq*cap*4));
    ck(cudaMemcpy(ds,scores.data(),scores.size()*4,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dt,steps.data(),steps.size()*4,cudaMemcpyHostToDevice));
    auto old=[&]{k::qsa_block_topk(ds,dt,nq,stride,cap,s,da,nullptr);};
    auto bounded=[&]{k::qsa_block_topk(ds,dt,nq,stride,cap,s,db,nullptr,active);};
    old(); bounded(); ck(cudaDeviceSynchronize());
    std::vector<int32_t> a(nq*cap),b(a.size());
    ck(cudaMemcpy(a.data(),da,a.size()*4,cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(b.data(),db,b.size()*4,cudaMemcpyDeviceToHost));
    int64_t same=0;
    for(int64_t i=0;i<nq;++i) {
        const int64_t w=steps[i*k::kStepCount+k::kStepWidth];
        same+=std::equal(a.begin()+i*cap,a.begin()+i*cap+w,b.begin()+i*cap);
    }
    cudaEvent_t e0,e1; ck(cudaEventCreate(&e0)); ck(cudaEventCreate(&e1));
    auto timed=[&](auto f) {
        ck(cudaEventRecord(e0)); for(int r=0;r<reps;++r) f(); ck(cudaEventRecord(e1));
        ck(cudaEventSynchronize(e1)); float ms=0; ck(cudaEventElapsedTime(&ms,e0,e1)); return ms/reps;
    };
    const float x=timed(old),y=timed(bounded),z=timed(old);
    std::printf("TOPK ctx %lld nq %lld stride %lld active %lld mode %d identical %lld/%lld ABA_ms %.6f %.6f %.6f\n",
        (long long)ctx,(long long)nq,(long long)stride,(long long)active,mode,(long long)same,(long long)nq,x,y,z);
    ck(cudaFree(ds)); ck(cudaFree(dt)); ck(cudaFree(da)); ck(cudaFree(db));
    return same==nq ? 0 : 1;
}
