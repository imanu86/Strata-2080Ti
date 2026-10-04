// Bounded synthetic CUDA checks for the default-off closed-cache experiment.
#include "strata/kernels/native_router.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace k=strata::kernels;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
template<class T> struct Dev {T* p=nullptr;explicit Dev(size_t n){ck(cudaMalloc((void**)&p,n*sizeof(T)));}~Dev(){cudaFree(p);}};
int main(){try{
    cudaStream_t cs;ck(cudaStreamCreateWithFlags(&cs,cudaStreamNonBlocking));
    Dev<float> logits(8*512),weights(80),counts(2*512);
    Dev<int32_t> ids(80),original(80),res(512),mode(1),keep(1);
    int checks=0;
    for(int n:{1,4,8})for(int pattern=0;pattern<3;++pattern){
        std::vector<float> h(n*512);
        for(int t=0;t<n;++t)for(int e=0;e<512;++e)
            h[t*512+e]=pattern==0?0.f:pattern==1?std::sin(float(e*7+t)) * 3.f:float(e)*40.f;
        ck(cudaMemcpy(logits.p,h.data(),h.size()*4,cudaMemcpyHostToDevice));ck(cudaDeviceSynchronize());
        k::native_router_top10_multi(logits.p,ids.p,weights.p,n,cs);ck(cudaStreamSynchronize(cs));
        std::vector<int32_t> baseline(n*10),got(n*10),raw(n*10);
        std::vector<float> basew(n*10),gotw(n*10);
        ck(cudaMemcpy(baseline.data(),ids.p,n*40,cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(basew.data(),weights.p,n*40,cudaMemcpyDeviceToHost));
        for(int mask=0;mask<3;++mask){
            std::vector<int32_t> r(512,-1);
            for(int e=0;e<512;++e)if(mask==0 || (mask==1?e%3==1:e<10))r[e]=e;
            ck(cudaMemcpy(res.p,r.data(),512*4,cudaMemcpyHostToDevice));ck(cudaDeviceSynchronize());
            for(int m:{0,1,2}){
                ck(cudaMemcpy(mode.p,&m,4,cudaMemcpyHostToDevice));ck(cudaDeviceSynchronize());
                k::native_router_top10_multi(logits.p,ids.p,weights.p,n,cs);
                k::closed_router_apply(logits.p,ids.p,weights.p,original.p,res.p,mode.p,n,cs);
                ck(cudaStreamSynchronize(cs));
                ck(cudaMemcpy(got.data(),ids.p,n*40,cudaMemcpyDeviceToHost));
                ck(cudaMemcpy(raw.data(),original.p,n*40,cudaMemcpyDeviceToHost));
                ck(cudaMemcpy(gotw.data(),weights.p,n*40,cudaMemcpyDeviceToHost));
                if(raw!=baseline){
                    std::fprintf(stderr,"shadow mismatch n=%d pattern=%d mask=%d mode=%d\n",n,pattern,mask,m);
                    for(int j=0;j<n*10;++j)if(raw[j]!=baseline[j])std::fprintf(stderr,"i=%d raw=%d base=%d\n",j,raw[j],baseline[j]);
                }
                need(raw==baseline,"shadow IDs changed");
                if(m==0 || mask==0){
                    need(got==baseline,"full/observe IDs differ");
                    need(std::memcmp(gotw.data(),basew.data(),n*40)==0,"full/observe weights not bitwise");
                }else if(m==2){
                    int first=0;while(r[first]<0)++first;
                    for(int i=0;i<n*10;++i){
                        bool hit=r[baseline[i]]>=0;
                        need(got[i]==(hit?baseline[i]:first),"drop ID incorrect");
                        need(gotw[i]==(hit?basew[i]:0.f),"drop weight changed/renormalized");
                    }
                }else{
                    for(int t=0;t<n;++t){
                        double sum=0;int distinct=0;
                        for(int j=0;j<10;++j){need(r[got[t*10+j]]>=0,"selected nonresident");sum+=gotw[t*10+j];
                            for(int q=0;q<j;++q)need(got[t*10+j]!=got[t*10+q],"duplicate selected expert");++distinct;}
                        need(std::abs(sum-1)<2.e-6 && distinct==10,"restricted normalization incorrect");
                        if(pattern<=1){
                            std::vector<int> order;for(int e=0;e<512;++e)if(r[e]>=0)order.push_back(e);
                            std::sort(order.begin(),order.end(),[&](int a,int b){return h[t*512+a]!=h[t*512+b]?h[t*512+a]>h[t*512+b]:a<b;});
                            double denom=0;for(int j=0;j<10;++j)denom+=std::exp(double(h[t*512+order[j]]-h[t*512+order[0]]));
                            for(int j=0;j<10;++j){
                                need(got[t*10+j]==order[j],"restricted ID differs from host reference");
                                const double expected=std::exp(double(h[t*512+order[j]]-h[t*512+order[0]]))/denom;
                                need(std::abs(gotw[t*10+j]-expected)<2.e-6,"restricted weight differs from host reference");
                            }
                        }
                    }
                }
                ++checks;
            }
        }
    }
    // Graph replay must read a changed mode, not retain a host-side branch from capture.
    std::vector<float> h(512);std::iota(h.begin(),h.end(),0.f);
    std::vector<int32_t> r(512,-1);for(int e=0;e<64;++e)r[e]=e;
    ck(cudaMemcpy(logits.p,h.data(),512*4,cudaMemcpyHostToDevice));ck(cudaDeviceSynchronize());
    ck(cudaMemcpy(res.p,r.data(),512*4,cudaMemcpyHostToDevice));ck(cudaDeviceSynchronize());
    ck(cudaStreamBeginCapture(cs,cudaStreamCaptureModeThreadLocal));
    k::native_router_top10_multi(logits.p,ids.p,weights.p,1,cs);
    k::closed_router_apply(logits.p,ids.p,weights.p,original.p,res.p,mode.p,1,cs);
    cudaGraph_t graph;cudaGraphExec_t exec;
    ck(cudaStreamEndCapture(cs,&graph));ck(cudaGraphInstantiate(&exec,graph,0));
    for(int m:{0,1,2,1,0}){
        ck(cudaMemcpy(mode.p,&m,4,cudaMemcpyHostToDevice));ck(cudaDeviceSynchronize());ck(cudaGraphLaunch(exec,cs));ck(cudaStreamSynchronize(cs));
        int out;float w;ck(cudaMemcpy(&out,ids.p,4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(&w,weights.p,4,cudaMemcpyDeviceToHost));
        need(out==(m==0?511:m==1?63:0),"captured policy stale");if(m==2)need(w==0,"captured drop not zero");++checks;
    }
    ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));
    // Committed-row counting excludes the two discarded speculative rows per layer.
    std::vector<int32_t> shadow(80);for(int l=0;l<2;++l)for(int t=0;t<4;++t)for(int j=0;j<10;++j)shadow[l*40+t*10+j]=t*10+j;
    ck(cudaMemcpy(original.p,shadow.data(),80*4,cudaMemcpyHostToDevice));ck(cudaDeviceSynchronize());ck(cudaMemset(counts.p,0,2*512*4));
    int two=2;ck(cudaMemcpy(keep.p,&two,4,cudaMemcpyHostToDevice));ck(cudaDeviceSynchronize());
    k::closed_router_count(original.p,counts.p,2,4,keep.p,0,cs);ck(cudaStreamSynchronize(cs));
    std::vector<float> c(1024);ck(cudaMemcpy(c.data(),counts.p,1024*4,cudaMemcpyDeviceToHost));
    for(int l=0;l<2;++l)for(int e=0;e<512;++e)need(c[l*512+e]==(e<20?1.f:0.f),"rejected row entered counts");
    ++checks;ck(cudaStreamDestroy(cs));std::printf("CLOSED_ROUTER_PARITY_OK %d cases\n",checks);return 0;
}catch(const std::exception&e){std::fprintf(stderr,"CLOSED_ROUTER_FAIL %s\n",e.what());return 1;}}
